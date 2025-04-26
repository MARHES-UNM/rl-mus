import argparse
from datetime import datetime
from time import time
from matplotlib import pyplot as plt
import numpy as np
import ray
from ray import air, tune
from uav_sim.envs.uav_rl_ren import UavRlRen
from uav_sim.envs.uav_sim import UavSim
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import get_trainable_cls
from uav_sim.utils.callbacks import TrainCallback
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog

import os
import logging
import json
from uav_sim.utils.safety_layer import SafetyLayer
from plot_results import plot_uav_states
import math

from uav_sim.networks.fix_model import TorchFixModel
from uav_sim.networks.cnn_model import TorchCnnModel
from uav_sim.utils.utils import get_git_hash


PATH = Path(__file__).parent.absolute().resolve()
RESULTS_DIR = PATH / "ray_results"
logger = logging.getLogger(__name__)


ModelCatalog.register_custom_model("torch_fix_model", TorchFixModel)
ModelCatalog.register_custom_model("torch_cnn_model", TorchCnnModel)


def setup_stream(logging_level=logging.DEBUG):
    # Turns on logging to console
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "<%(module)s:%(funcName)s:%(lineno)s> - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging_level)


def get_obs_act_space(config):

    # Need to create a temporary environment to get obs and action space
    env_config = config["env_config"]
    render = env_config["render"]
    env_config["render"] = False
    temp_env = UavRlRen(env_config)

    env_obs_space = temp_env.observation_space[0]
    env_action_space = temp_env.action_space[0]
    temp_env.close()
    env_config["render"] = render

    return env_obs_space, env_action_space


def get_algo_config(config, env_obs_space, env_action_space, env_task_fn=None):

    custom_model = config["exp_config"].setdefault("custom_model", "torch_cnn_model")

    algo_config = (
        get_trainable_cls(config["exp_config"]["run"])
        .get_default_config()
        .environment(
            env=config["env_name"],
            env_config=config["env_config"],
            env_task_fn=env_task_fn,
        )
        .framework(config["exp_config"]["framework"])
        .rollouts(
            num_rollout_workers=0,
            # observation_filter="MeanStdFilter",  # or "NoFilter"
        )
        .debugging(log_level="ERROR", seed=config["env_config"]["seed"])
        .resources(
            num_gpus=0,
            # placement_strategy=[{"cpu": 1}, {"cpu": 1}],
            num_gpus_per_learner_worker=0,
        )
        # See for changing model options https://docs.ray.io/en/latest/rllib/rllib-models.html
        # Must set learner_api to use custom model and most set rl_module to False
        # https://github.com/ray-project/ray/issues/40201
        # .training(
        #     model={
        #         "custom_model": custom_model,
        #         # Extra kwargs to be passed to your model's c'tor.
        #         "custom_model_config": {"n_agent_state": 6, "max_action_val": 5},
        #     },
        #     _enable_learner_api=False,
        # )
        # .rl_module(_enable_rl_module_api=False)
        .multi_agent(
            policies={
                "shared_policy": (
                    None,
                    env_obs_space,
                    env_action_space,
                    {},
                )
            },
            # Always use "shared" policy.
            policy_mapping_fn=(
                lambda agent_id, episode, worker, **kwargs: "shared_policy"
            ),
        )
    )

    return algo_config


def curriculum_fn(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to.

    Args:
        train_results: The train results returned by Algorithm.train().
        task_settable_env: A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx: The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.

    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    # Our env supports tasks 1 (default) to 5.
    # With each task, rewards get scaled up by a factor of 10, such that:
    # Level 1: Expect rewards between 0.0 and 1.0.
    # Level 2: Expect rewards between 1.0 and 10.0, etc..
    # We will thus raise the level/task each time we hit a new power of 10.0
    time_steps = train_results.get("timesteps_total")
    new_task = time_steps // 3000000
    # Clamp between valid values, just in case:
    new_task = max(min(new_task, 2), 0)
    print(
        f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
        f"\nR={train_results['episode_reward_mean']}"
        f"\ntimesteps={train_results['timesteps_total']}"
        f"\nSetting env to task={new_task}"
    )
    return new_task


def train(args):

    num_gpus = int(math.ceil(os.environ.get("RLLIB_NUM_GPUS", args.gpu)))
    # args.local_mode = True
    ray.init(local_mode=args.local_mode, num_gpus=num_gpus)

    # We get the spaces here before test vary the experiment treatments (factors)
    env_obs_space, env_action_space = get_obs_act_space(args.config)

    # # This worked for ffcdea1
    # # Vary treatments here
    # args.config["env_config"]["num_uavs"] = 4
    # args.config["env_config"]["uav_type"] = tune.grid_search(["UavBase"])
    # args.config["env_config"]["use_safe_action"] = tune.grid_search([False])
    # args.config["env_config"]["obstacle_collision_weight"] = 1.0
    # args.config["env_config"]["uav_collision_weight"] = 1.0
    # args.config["env_config"]["crash_penalty"] = 10
    # # args.config["env_config"]["beta"] = tune.loguniform(0.001, 0.3)
    # # args.config["env_config"]["max_time_penalty"] = tune.grid_search([25, 50])
    # # args.config["env_config"]["stp_penalty"] = tune.qloguniform(0.4, 10, 0.05)
    # args.config["env_config"]["stp_penalty"] = tune.grid_search([1.5, 0.0])
    # args.config["env_config"]["t_go_error_func"] = tune.grid_search(["sum"])
    # args.config["env_config"]["max_dt_std"] = tune.grid_search([0.05])
    # args.config["env_config"]["max_dt_go_error"] = tune.grid_search([0.2])
    # args.config["env_config"]["tgt_reward"] = 50
    # args.config["env_config"]["sa_reward"] = 50
    # args.config["env_config"]["beta"] = 0.10
    # args.config["env_config"]["early_done"] = tune.grid_search([False])
    # args.config["env_config"]["beta_vel"] = 0.1

    neg_penalty = 1
    # Vary treatments here
    args.config["env_config"]["beta_vel"] = 0
    args.config["env_config"]["beta"] = 0
    args.config["env_config"]["crash_penalty"] = neg_penalty
    args.config["env_config"]["early_done"] = tune.grid_search([True])
    args.config["env_config"]["max_dt_go_error"] = tune.grid_search([1.5, 3])
    args.config["env_config"]["max_dt_std"] = tune.grid_search([0.5])
    args.config["env_config"]["max_time_penalty"] = neg_penalty
    args.config["env_config"]["num_uavs"] = 4
    args.config["env_config"]["obstacle_collision_weight"] = neg_penalty
    args.config["env_config"]["sa_reward"] = tune.grid_search([1])
    # args.config["env_config"]["start_level"] = tune.grid_search([2, 0])
    args.config["env_config"]["stp_penalty"] = tune.grid_search([0])
    args.config["env_config"]["t_go_error_func"] = tune.grid_search(["mean"])
    args.config["env_config"]["tgt_reward"] = tune.grid_search([1])
    args.config["env_config"]["uav_collision_weight"] = neg_penalty
    args.config["env_config"]["uav_type"] = "UavBase"
    args.config["env_config"]["use_safe_action"] = False
    args.config["env_config"]["use_virtual_leader"] = tune.grid_search([False])
    # custom_model = tune.grid_search(
    #     [
    #         "torch_fix_model",
    #         "torch_cnn_model",
    #     ]
    # )
    # custom_model = tune.grid_search(["torch_cnn_model"])
    # args.config["env_config"]["target_pos_rand"] = True

    # args.config["env_config"]["d_thresh"] = tune.grid_search([0.15, 0.01])
    # args.config["env_config"]["d_thresh"] = tune.grid_search([0.15])
    # args.config["env_config"]["time_final"] = tune.grid_search([8.0, 20.0])
    # args.config["env_config"]["time_final"] = tune.grid_search([8.0])
    # args.config["env_config"]["t_go_max"] = tune.grid_search([2.0])

    # obs_filter = tune.grid_search(["NoFilter", "MeanStdFilter"])
    obs_filter = "NoFilter"
    task_fn = curriculum_fn if "curriculum" in args.config["env_name"] else None
    # task_fn = tune.grid_search([None, curriculum_fn])
    callback_list = [TrainCallback]
    # multi_callbacks = make_multi_callbacks(callback_list)
    # Common config params: https://docs.ray.io/en/latest/rllib/rllib-training.html#configuring-rllib-algorithms
    train_config = (
        get_algo_config(
            args.config, env_obs_space, env_action_space, env_task_fn=task_fn
        ).rollouts(
            num_rollout_workers=(
                1 if args.smoke_test else args.cpu
            ),  # set 0 to main worker run sim
            num_envs_per_worker=args.num_envs_per_worker,
            # create_env_on_local_worker=True,
            # rollout_fragment_length="auto",
            batch_mode="complete_episodes",
            observation_filter=obs_filter,  # or "NoFilter"
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # TODO: set num_learner_workers to 0 so to train train main env
        # https://docs.ray.io/en/latest/rllib/rllib-training.html#rllib-config-resources
        .resources(
            num_gpus=0 if args.smoke_test else num_gpus,
            num_learner_workers=1,
            num_gpus_per_learner_worker=0 if args.smoke_test else args.gpu,
        )
        # See for changing model options https://docs.ray.io/en/latest/rllib/rllib-models.html
        # .model()
        # See for specific ppo config: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo
        # See for more on PPO hyperparameters: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
        .training(
            # https://docs.ray.io/en/latest/rllib/rllib-models.html
            model={"fcnet_hiddens": [32, 64, 128, 128, 64, 32]},
            # model={
            #     "custom_model": custom_model,
            #     # Extra kwargs to be passed to your custorm model.
            #     "custom_model_config": {"n_agent_state": 6, "max_action_val": 5},
            # },
            # _enable_learner_api=False,
            # 5e-5
            lr=tune.grid_search([5e-5]),
            use_gae=True,
            use_critic=True,
            lambda_=tune.grid_search([0.95]),
            train_batch_size=65536,
            # train_batch_size=131072,
            gamma=tune.grid_search([0.9997]),
            num_sgd_iter=32,
            sgd_minibatch_size=8192,
            vf_clip_param=10.0,
            vf_loss_coeff=0.5,
            clip_param=0.2,
            grad_clip=1.0,
            # entropy_coeff=0.0,
            # # seeing if this solves the error:
            # # https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
            # # Expected parameter loc (Tensor of shape (4096, 3)) of distribution Normal(loc: torch.Size([4096, 3]), scale: torch.Size([4096, 3])) to satisfy the constraint Real(),
            # kl_coeff=1.0,
            # kl_target=0.0068,
        )
        # .rl_module(_enable_rl_module_api=False)
        # .reporting(keep_per_episode_custom_metrics=True)
        # .evaluation(
        #     evaluation_interval=10, evaluation_duration=10  # default number of episodes
        # )
    )

    multi_callbacks = make_multi_callbacks(callback_list)
    train_config.callbacks(multi_callbacks)

    stop = {
        "training_iteration": 1 if args.smoke_test else args.stop_iters,
        "timesteps_total": args.stop_timesteps,
    }

    # # # trainable_with_resources = tune.with_resources(args.run, {"cpu": 18, "gpu": 1.0})
    # # # If you have 4 CPUs and 1 GPU on your machine, this will run 1 trial at a time.
    # # trainable_with_cpu_gpu = tune.with_resources(algo, {"cpu": 2, "gpu": 1})
    tuner = tune.Tuner(
        args.config["exp_config"]["run"],
        # args.run,
        # trainable_with_cpu_gpu,
        param_space=train_config.to_dict(),
        # tune_config=tune.TuneConfig(num_samples=10),
        run_config=air.RunConfig(
            stop=stop,
            local_dir=args.log_dir,
            name=args.name,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=100,
                # checkpoint_score_attribute="",
                checkpoint_at_end=True,
                checkpoint_frequency=5,
            ),
        ),
    )

    results = tuner.fit()


def test(args):
    if args.tune_run:
        pass
    else:
        if args.seed:
            print(f"******************seed: {args.seed}")
            seed_val = none_or_int(args.seed)
            args.config["env_config"]["seed"] = seed_val

        if args.checkpoint:
            args.config["exp_config"]["checkpoint"] = args.checkpoint

        if args.uav_type:
            args.config["env_config"]["uav_type"] = args.uav_type

        if args.run:
            args.config["exp_config"]["run"] = args.run

        max_num_episodes = args.max_num_episodes
        experiment_num = args.experiment_num
        args.config["render"] = args.render
        args.config["plot_results"] = args.plot_results

        if args.write_exp:
            args.config["write_experiment"] = True

            output_folder = Path(args.log_dir)
            if not output_folder.exists():
                output_folder.mkdir(parents=True, exist_ok=True)

            args.config["fname"] = output_folder / "result.json"

        experiment(args.config, max_num_episodes, experiment_num)


def experiment(exp_config={}, max_num_episodes=1, experiment_num=0):
    fname = exp_config.setdefault("fname", None)
    write_experiment = exp_config.setdefault("write_experiment", False)
    env_config = exp_config["env_config"]
    render = exp_config["render"]
    plot_results = exp_config["plot_results"]

    algo_to_run = exp_config["exp_config"].setdefault("run", "PPO")
    if algo_to_run not in ["cc", "ren", "PPO"]:
        print("Unrecognized algorithm. Exiting...")
        exit(99)

    # env = UavSim(env_config)
    env = UavRlRen(env_config)
    if algo_to_run == "PPO":
        checkpoint = exp_config["exp_config"].setdefault("checkpoint", None)
        env_obs_space, env_action_space = get_obs_act_space(exp_config)

        # Reload the algorithm as is from training.
        # if checkpoint is not None:
        # algo = Algorithm.from_checkpoint(checkpoint)
        if checkpoint is not None:
            # use policy here instead of algorithm because it's more efficient
            use_policy = True
            from ray.rllib.policy.policy import Policy
            from ray.rllib.models.preprocessors import get_preprocessor

            algo = Policy.from_checkpoint(checkpoint)

            # need preprocesor here if using policy
            # https://docs.ray.io/en/releases-2.6.3/rllib/rllib-training.html
            prep = get_preprocessor(env_obs_space)(env_obs_space)
        else:
            use_policy = False
            env.close()

            algo = (
                get_algo_config(exp_config, env_obs_space, env_action_space)
            ).build()

            env = algo.workers.local_worker().env
            # restore algorithm if need be:
            # algo.restore(checkpoint)

    if exp_config["exp_config"]["safe_action_type"] == "nn_cbf":
        sl = SafetyLayer(env, exp_config["safety_layer_cfg"])

    time_step_list = []
    uav_collision_list = [[] for idx in range(env.num_uavs)]
    obstacle_collision_list = [[] for idx in range(env.num_uavs)]
    uav_done_list = [[] for idx in range(env.num_uavs)]
    uav_done_dt_list = [[] for idx in range(env.num_uavs)]
    uav_done_time_list = [[] for idx in range(env.num_uavs)]
    uav_dt_go_list = [[] for idx in range(env.num_uavs)]
    uav_t_go_list = [[] for idx in range(env.num_uavs)]
    rel_pad_dist = [[] for idx in range(env.num_uavs)]
    rel_pad_vel = [[] for idx in range(env.num_uavs)]
    uav_state = [[] for idx in range(env.num_uavs)]
    uav_reward = [[] for idx in range(env.num_uavs)]
    rel_pad_state = [[] for idx in range(env.num_uavs)]
    obstacle_state = [[] for idx in range(env.max_num_obstacles)]
    target_state = []

    results = {
        "num_episodes": 0.0,
        "uav_collision": 0.0,
        "obs_collision": 0.0,
        "uav_crashed": 0.0,
        "uav_reward": 0.0,
        "uav_done": [[] for idx in range(env.num_uavs)],
        "uav_done_dt": [[] for idx in range(env.num_uavs)],
        "uav_done_time": [[] for idx in range(env.num_uavs)],
        "uav_sa_sat": [[] for idx in range(env.num_uavs)],
        "episode_time": [],
        "episode_data": {
            "time_step_list": [],
            "uav_collision_list": [],
            "obstacle_collision_list": [],
            "uav_done_list": [],
            "uav_done_dt_list": [],
            "uav_dt_go_list": [],
            "uav_t_go_list": [],
            "rel_pad_dist": [],
            "rel_pad_vel": [],
            "uav_state": [],
            "uav_reward": [],
            "rel_pad_state": [],
            "obstacle_state": [],
            "target_state": [],
        },
    }

    num_episodes = 0
    env_out, done = env.reset(), {i.id: False for i in env.uavs.values()}
    obs, info = env_out
    done["__all__"] = False

    logger.debug("running experiment")
    num_episodes = 0
    start_time = time()

    while num_episodes < max_num_episodes:
        actions = {}
        for idx in range(env.num_uavs):
            # if env._early_done and env.uavs[idx].done:
            # continue
            # classic control
            if algo_to_run == "cc":
                actions[idx] = env.get_time_coord_action(env.uavs[idx])
            elif algo_to_run == "ren":
                actions[idx] = env.get_tc_controller(env.uavs[idx])
            elif algo_to_run == "PPO":
                if use_policy:
                    # https://docs.ray.io/en/latest/rllib/rllib-training.html#rllib-config-exploration
                    # https://discuss.ray.io/t/inconsistent-actions-from-algorithm-compute-single-action/11027/3
                    # trying to get deterministic results
                    actions[idx] = algo.compute_single_action(
                        prep.transform(obs[idx]), explore=False
                    )[0]
                else:
                    actions[idx] = algo.compute_single_action(
                        obs[idx], policy_id="shared_policy"
                    )

            if exp_config["exp_config"]["safe_action_type"] is not None:
                if exp_config["exp_config"]["safe_action_type"] == "cbf":
                    actions[idx] = env.get_safe_action(env.uavs[idx], actions[idx])
                elif exp_config["exp_config"]["safe_action_type"] == "nn_cbf":
                    actions[idx] = sl.get_action(obs[idx], actions[idx])
                elif exp_config["exp_config"]["safe_action_type"] == "sca":
                    actions[idx] = env.get_col_avoidance(env.uavs[idx], actions[idx])
                else:
                    print("unknow safe action type")

        obs, rew, done, truncated, info = env.step(actions)
        for k, v in info.items():
            results["uav_collision"] += v["uav_collision"]
            results["obs_collision"] += v["obstacle_collision"]
            results["uav_crashed"] += v["uav_crashed"]

        for k, v in rew.items():
            results["uav_reward"] += v

        if render:
            env.render()

        # only get for 1st episode
        if num_episodes == 0:
            for k, v in info.items():
                uav_collision_list[k].append(v["uav_collision"])
                obstacle_collision_list[k].append(v["obstacle_collision"])
                uav_done_list[k].append(v["uav_landed"])
                uav_done_dt_list[k].append(v["uav_done_dt"])
                uav_done_time_list[k].append(v["uav_done_time"])
                uav_dt_go_list[k].append(v["uav_dt_go"])
                uav_t_go_list[k].append(v["uav_t_go"])
                rel_pad_dist[k].append(v["uav_rel_dist"])
                rel_pad_vel[k].append(v["uav_rel_vel"])
                uav_reward[k].append(rew[k])

            for uav_idx in range(env.num_uavs):
                uav_state[uav_idx].append(env.uavs[uav_idx].state.tolist())
                rel_pad_state[uav_idx].append(env.uavs[uav_idx].pad.state.tolist())

            target_state.append(env.target.state.tolist())
            time_step_list.append(env.time_elapsed)

            for obs_idx in range(env.max_num_obstacles):
                obstacle_state[obs_idx].append(env.obstacles[obs_idx].state.tolist())

        if done["__all__"]:
            num_episodes += 1
            for k, v in info.items():
                results["uav_done"][k].append(v["uav_landed"])
                results["uav_done_dt"][k].append(v["uav_done_dt"])
                results["uav_done_time"][k].append(v["uav_done_time"])
                results["uav_sa_sat"][k].append(v["uav_sa_sat"])
            results["num_episodes"] = num_episodes
            results["episode_time"].append(env.time_elapsed)

            if num_episodes <= 1:
                results["episode_data"]["time_step_list"].append(time_step_list)
                results["episode_data"]["uav_collision_list"].append(uav_collision_list)
                results["episode_data"]["obstacle_collision_list"].append(
                    obstacle_collision_list
                )
                results["episode_data"]["uav_done_list"].append(uav_done_list)
                results["episode_data"]["uav_done_dt_list"].append(uav_done_dt_list)
                results["episode_data"]["uav_dt_go_list"].append(uav_dt_go_list)
                results["episode_data"]["uav_t_go_list"].append(uav_t_go_list)
                results["episode_data"]["rel_pad_dist"].append(rel_pad_dist)
                results["episode_data"]["rel_pad_vel"].append(rel_pad_vel)
                results["episode_data"]["uav_state"].append(uav_state)
                results["episode_data"]["target_state"].append(target_state)
                results["episode_data"]["uav_reward"].append(uav_reward)
                results["episode_data"]["rel_pad_state"].append(rel_pad_state)
                results["episode_data"]["obstacle_state"].append(obstacle_state)

            if render:
                env.render(mode="human", done=True, plot_results=plot_results)
                # im = env.render(mode="rgb_array", done=True, plot_results=plot_results)
            if plot_results:
                plot_uav_states(results, env_config, num_episodes - 1)

            if num_episodes == max_num_episodes:
                end_time = time() - start_time
                break
            env_out, done = env.reset(), {
                agent.id: False for agent in env.uavs.values()
            }
            obs, info = env_out
            done["__all__"] = False

            # reinitialize data arrays
            time_step_list = [[] for idx in range(env.num_uavs)]
            uav_collision_list = [[] for idx in range(env.num_uavs)]
            obstacle_collision_list = [[] for idx in range(env.num_uavs)]
            uav_done_list = [[] for idx in range(env.num_uavs)]
            uav_done_dt_list = [[] for idx in range(env.num_uavs)]
            uav_done_time_list = [[] for idx in range(env.num_uavs)]
            uav_dt_go_list = [[] for idx in range(env.num_uavs)]
            uav_t_go_list = [[] for idx in range(env.num_uavs)]
            rel_pad_dist = [[] for idx in range(env.num_uavs)]
            rel_pad_vel = [[] for idx in range(env.num_uavs)]
            uav_state = [[] for idx in range(env.num_uavs)]
            uav_reward = [[] for idx in range(env.num_uavs)]
            rel_pad_state = [[] for idx in range(env.num_uavs)]
            obstacle_state = [[] for idx in range(env.num_obstacles)]
            target_state = []

    env.close()

    if write_experiment:
        if fname is None:
            file_prefix = {
                "tgt_v": env_config["target_v"],
                "sa": env_config["use_safe_action"],
                "obs": env_config["num_obstacles"],
                "seed": env_config["seed"],
            }
            file_prefix = "_".join(
                [f"{k}_{str(int(v))}" for k, v in file_prefix.items()]
            )

            fname = f"exp_{experiment_num}_{file_prefix}_result.json"
        # writing too much data, for now just save the first experiment
        for k, v in results["episode_data"].items():
            results["episode_data"][k] = [
                v[0],
            ]

        results["env_config"] = env.env_config
        results["exp_config"] = exp_config["exp_config"]
        results["time_total_s"] = end_time
        with open(fname, "w") as f:
            json.dump(results, f)

    logger.debug("done")


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def get_default_env_config(path, config):
    env = UavSim(config["env_config"])

    config["env_config"].update(env.env_config)

    with open(path, "w") as f:
        json.dump(config, f)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_config", default=f"{PATH}/configs/sim_config.json")
    parser.add_argument("--get_config")
    parser.add_argument(
        "--log_dir",
    )
    parser.add_argument(
        "--run", type=str, help="The RLlib-registered algorithm to use."
    )
    parser.add_argument("--tf", type=float)

    parser.add_argument("--name", help="Name of experiment.", default="debug")
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument("--env_name", type=str, default="multi-uav-ren-v0")

    subparsers = parser.add_subparsers(dest="command")
    test_sub = subparsers.add_parser("test")
    test_sub.add_argument("--checkpoint")
    test_sub.add_argument("--uav_type", type=str)
    test_sub.add_argument("--max_num_episodes", type=int, default=1)
    test_sub.add_argument("--experiment_num", type=int, default=0)
    test_sub.add_argument("--render", action="store_true", default=False)
    test_sub.add_argument("--write_exp", action="store_true", default=False)
    test_sub.add_argument("--plot_results", action="store_true", default=False)
    test_sub.add_argument("--tune_run", action="store_true", default=False)
    test_sub.add_argument("--seed")

    test_sub.set_defaults(func=test)

    train_sub = subparsers.add_parser("train")
    train_sub.add_argument("--smoke_test", action="store_true", help="run quicktest")

    train_sub.add_argument(
        "--stop_iters", type=int, default=1000, help="Number of iterations to train."
    )
    train_sub.add_argument(
        "--stop_timesteps",
        type=int,
        default=int(30e6),
        help="Number of timesteps to train.",
    )

    train_sub.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    train_sub.add_argument("--checkpoint", type=str)
    train_sub.add_argument("--gpu", type=float, default=0.0)
    train_sub.add_argument("--num_envs_per_worker", type=int, default=12)
    train_sub.add_argument(
        "--cpu", type=int, default=1, help="num_rollout_workers default is 1"
    )
    train_sub.set_defaults(func=train)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    setup_stream()

    with open(args.load_config, "rt") as f:
        args.config = json.load(f)

    if args.get_config:
        get_default_env_config(args.get_config, args.config)
        return 0

    args.config["env_name"] = args.env_name

    logger.debug(f"config: {args.config}")

    if args.run is not None:
        args.config["exp_config"]["run"] = args.run

    if args.tf is not None:
        args.config["env_config"]["time_final"] = args.tf

    if not args.log_dir:
        branch_hash = get_git_hash()

        num_uavs = args.config["env_config"]["num_uavs"]
        num_obs = args.config["env_config"]["max_num_obstacles"]
        dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_dir = f"{args.func.__name__}/{args.config['exp_config']['run']}/{args.env_name}_{dir_timestamp}_{branch_hash}_{num_uavs}u_{num_obs}o/{args.name}"
        args.log_dir = RESULTS_DIR / log_dir

    args.log_dir = Path(args.log_dir).resolve()

    args.func(args)


if __name__ == "__main__":
    main()
