import argparse
from datetime import datetime
from time import time
from matplotlib import pyplot as plt
import numpy as np
import ray
from ray import air, tune
from uav_sim.envs.uav_sim import UavSim
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import get_trainable_cls
from uav_sim.utils.callbacks import TrainCallback
from ray.rllib.algorithms.callbacks import make_multi_callbacks

import os
import logging
import json
from uav_sim.utils.safety_layer import SafetyLayer
from ray import tune
from plot_results import plot_uav_states

from uav_sim.utils.utils import get_git_hash


PATH = Path(__file__).parent.absolute().resolve()
RESULTS_DIR = Path.home() / "ray_results"
logger = logging.getLogger(__name__)


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
    temp_env = UavSim(env_config)

    env_obs_space = temp_env.observation_space[0]
    env_action_space = temp_env.action_space[0]
    temp_env.close()
    env_config["render"] = render

    return env_obs_space, env_action_space


def get_algo_config(config, env_obs_space, env_action_space):

    algo_config = (
        get_trainable_cls(config["exp_config"]["run"])
        .get_default_config()
        .environment(env=config["env_name"], env_config=config["env_config"])
        .framework(config["exp_config"]["framework"])
        .rollouts(
            num_rollout_workers=0,
            # observation_filter="MeanStdFilter",  # or "NoFilter"
        )
        .debugging(log_level="ERROR", seed=config["env_config"]["seed"])
        .resources(
            num_gpus=0,
            placement_strategy=[{"cpu": 1}, {"cpu": 1}],
            num_gpus_per_learner_worker=0,
        )
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


def train(args):

    # args.local_mode = True
    ray.init(local_mode=args.local_mode, num_gpus=1)

    # We get the spaces here before test vary the experiment treatments (factors)
    env_obs_space, env_action_space = get_obs_act_space(args.config)

    # Vary treatments here
    num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", args.gpu))
    args.config["env_config"]["num_uavs"] = 4
    args.config["env_config"]["uav_type"] = tune.grid_search(["Uav", "UavBase"])
    args.config["env_config"]["use_safe_action"] = tune.grid_search([False])
    args.config["env_config"]["target_pos_rand"] = True

    args.config["env_config"]["tgt_reward"] = 100
    args.config["env_config"]["stp_penalty"] = 5
    args.config["env_config"]["beta"] = 0.3
    args.config["env_config"]["d_thresh"] = tune.grid_search([0.15, 0.01])
    # args.config["env_config"]["time_final"] = tune.grid_search([8.0, 20.0])
    args.config["env_config"]["time_final"] = tune.grid_search([20.0])
    args.config["env_config"]["t_go_max"] = 2.0
    args.config["env_config"]["obstacle_collision_weight"] = 0.1
    args.config["env_config"]["uav_collision_weight"] = 0.1

    args.config["env_config"]["crash_penalty"] = 10

    obs_filter = "NoFilter"
    callback_list = [TrainCallback]
    # multi_callbacks = make_multi_callbacks(callback_list)
    # Common config params: https://docs.ray.io/en/latest/rllib/rllib-training.html#configuring-rllib-algorithms
    train_config = (
        get_algo_config(args.config, env_obs_space, env_action_space).rollouts(
            num_rollout_workers=(
                1 if args.smoke_test else args.num_rollout_workers
            ),  # set 0 to main worker run sim
            num_envs_per_worker=args.num_envs_per_worker,
            # create_env_on_local_worker=True,
            # rollout_fragment_length="auto",
            batch_mode="complete_episodes",
            observation_filter=obs_filter,  # or "NoFilter"
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
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
            # model={"fcnet_hiddens": [512, 512, 512]},
            lr=5e-5,
            use_gae=True,
            use_critic=True,
            lambda_=0.95,
            # train_batch_size=65536,
            gamma=0.99,
            # num_sgd_iter=32,
            # sgd_minibatch_size=4096,
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
    if algo_to_run not in ["cc", "PPO"]:
        print("Unrecognized algorithm. Exiting...")
        exit(99)

    env = UavSim(env_config)
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
    uav_dt_go_list = [[] for idx in range(env.num_uavs)]
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
        "uav_done": [[] for idx in range(env.num_uavs)],
        "uav_done_dt": [[] for idx in range(env.num_uavs)],
        "episode_time": [],
        "episode_data": {
            "time_step_list": [],
            "uav_collision_list": [],
            "obstacle_collision_list": [],
            "uav_done_list": [],
            "uav_done_dt_list": [],
            "uav_dt_go_list": [],
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
            # classic control
            if algo_to_run == "cc":
                actions[idx] = env.get_time_coord_action(env.uavs[idx])
                # actions[idx] = env.get_tc_controller(env.uavs[idx])
            elif algo_to_run == "PPO":
                if use_policy:
                    actions[idx] = algo.compute_single_action(prep.transform(obs[idx]))[
                        0
                    ]
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

        # only get for 1st episode
        if num_episodes == 0:
            for k, v in info.items():
                uav_collision_list[k].append(v["uav_collision"])
                obstacle_collision_list[k].append(v["obstacle_collision"])
                uav_done_list[k].append(v["uav_landed"])
                uav_done_dt_list[k].append(v["uav_done_dt"])
                uav_dt_go_list[k].append(v["uav_dt_go"])
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

        if render:
            env.render()

        if done["__all__"]:
            num_episodes += 1
            for k, v in info.items():
                results["uav_done"][k].append(v["uav_landed"])
                results["uav_done_dt"][k].append(v["uav_done_dt"])
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
                results["episode_data"]["rel_pad_dist"].append(rel_pad_dist)
                results["episode_data"]["rel_pad_vel"].append(rel_pad_vel)
                results["episode_data"]["uav_state"].append(uav_state)
                results["episode_data"]["target_state"].append(target_state)
                results["episode_data"]["uav_reward"].append(uav_reward)
                results["episode_data"]["rel_pad_state"].append(rel_pad_state)
                results["episode_data"]["obstacle_state"].append(obstacle_state)

            if render:
                im = env.render(mode="rgb_array", done=True)
                # fig, ax = plt.subplots()
                # im = ax.imshow(im)
                # plt.show()
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
            uav_dt_go_list = [[] for idx in range(env.num_uavs)]
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_config", default=f"{PATH}/configs/sim_config.json")
    parser.add_argument(
        "--log_dir",
    )

    parser.add_argument(
        "--run", type=str, help="The RLlib-registered algorithm to use."
    )
    parser.add_argument("--name", help="Name of experiment.", default="debug")
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument("--env_name", type=str, default="multi-uav-sim-v0")

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
    train_sub.add_argument("--gpu", type=int, default=0.50)
    train_sub.add_argument("--num_envs_per_worker", type=int, default=12)
    train_sub.add_argument("--num_rollout_workers", type=int, default=1)
    train_sub.set_defaults(func=train)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    setup_stream()

    with open(args.load_config, "rt") as f:
        args.config = json.load(f)

    args.config["env_name"] = args.env_name

    logger.debug(f"config: {args.config}")

    if args.run is not None:
        args.config["exp_config"]["run"] = args.run

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
