import os

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
from datetime import datetime
import json
from ray import air, tune
from ray.tune.registry import register_env
import logging

from uav_sim.utils.utils import get_git_hash
from uav_sim.envs.uav_sim import UavSim
from pathlib import Path
import argparse
import ray
from ray.tune.registry import get_trainable_cls
from ray.rllib.utils import check_env

from ray.rllib.algorithms.callbacks import make_multi_callbacks
from uav_sim.utils.callbacks import TrainCallback
from ray.rllib.examples.env.multi_agent import FlexAgentsMultiAgent
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPO, PPOConfig


PATH = Path(__file__).parent.absolute().resolve()

formatter = "%(asctime)s: %(name)s - %(levelname)s - <%(module)s:%(funcName)s:%(lineno)d> - %(message)s"
logging.basicConfig(
    # filename=os.path.join(app_log_path, log_file_name),
    format=formatter
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_config", default=f"{PATH}/configs/sim_config.cfg")
    parser.add_argument(
        "--log_dir",
    )
    parser.add_argument("--name", help="Name of experiment.", default="debug")
    parser.add_argument(
        "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument(
        "--preload", type=str, help="preload policy with training weights"
    )

    parser.add_argument("--smoke_test", action="store_true", help="run quicktest")
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )

    parser.add_argument(
        "--stop-iters", type=int, default=1000, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=int(30e6),
        help="Number of timesteps to train.",
    )

    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    parser.add_argument("--env_name", type=str, default="multi-uav-sim-v0")

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0.0)
    parser.add_argument("--num_envs_per_worker", type=int, default=12)
    parser.add_argument("--num_rollout_workers", type=int, default=8)

    args = parser.parse_args()

    return args


def my_train_fn(config, reporter):
    iterations = config.pop("train-iterations", 1)

    config["env_config"]["use_safe_action"] = False
    config["env_config"]["tgt_reward"] = 0.0
    config["env_config"]["beta"] = 0.3
    config["env_config"]["d_thresh"] = 0.01
    config["env_config"]["uav_collision_weight"] = 0.0
    config["env_config"]["obstacle_collision_weight"] = 0.0

    config = PPOConfig().update_from_dict(config)

    # Train for n iterations with high LR.
    # config.lr = 0.01
    agent1 = config.build()
    for _ in range(iterations):
        result = agent1.train()
        result["phase"] = 1
        reporter(**result)
        phase1_time = result["timesteps_total"]
        if iterations % 5 == 0: 
            agent1.save()
    state = agent1.save()
    agent1.stop()


    config["env_config"]["use_safe_action"] = False
    config["env_config"]["tgt_reward"] = 10
    config["env_config"]["beta"] = 0.01
    config["env_config"]["d_thresh"] = 0.01
    config["env_config"]["uav_collision_weight"] = 0.0
    config["env_config"]["obstacle_collision_weight"] = 0.0

    config = PPOConfig().update_from_dict(config)
    # Train for n iterations with low LR
    # config.lr = 0.0001
    agent2 = config.build()
    agent2.restore(state)
    for _ in range(iterations):
        result = agent2.train()
        result["phase"] = 2
        result["timesteps_total"] += phase1_time  # keep time moving forward
        reporter(**result)
        if iterations % 5 == 0: 
            agent2.save()
    agent2.stop()


if __name__ == "__main__":
    args = parse_arguments()

    if args.load_config:
        with open(args.load_config, "rt") as f:
            args.config = json.load(f)
    if not args.log_dir:
        branch_hash = get_git_hash()

        dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        args.log_dir = (
            f"./results/{args.run}/{args.env_name}_{dir_timestamp}_{branch_hash}"
        )

        logdir = Path(args.log_dir)

        if not logdir.exists():
            logdir.mkdir(parents=True, exist_ok=True)

    env_config = args.config["env_config"]

    register_env(args.env_name, lambda env_config: UavSim(env_config))

    check_env(UavSim(env_config))

    # args.local_mode = True
    # ray.init(local_mode=args.local_mode, num_gpus=1)
    ray.init(num_gpus=1)

    temp_env = UavSim(args.config)
    num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", args.gpu))

    callback_list = [TrainCallback]
    # multi_callbacks = make_multi_callbacks(callback_list)
    # info on common configs: https://docs.ray.io/en/latest/rllib/rllib-training.html#specifying-rollout-workers
    train_config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(env=args.env_name, env_config=args.config["env_config"])
        .framework(args.framework)
        # .callbacks(multi_callbacks)
        .rollouts(
            num_rollout_workers=1
            if args.smoke_test
            else args.num_rollout_workers,  # set 0 to main worker run sim
            num_envs_per_worker=args.num_envs_per_worker,
            batch_mode="complete_episodes",
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .debugging(log_level="ERROR", seed=123)  # DEBUG, INFO
        .resources(
            num_gpus=0 if args.smoke_test else num_gpus,
            num_learner_workers=1,
            # num_gpus=args.gpu,
            # num_cpus_per_worker=args.cpu,
            # num_gpus_per_worker=args.gpu,
            num_gpus_per_learner_worker=0 if args.smoke_test else args.gpu,
        )
        # See for changing model options https://docs.ray.io/en/latest/rllib/rllib-models.html
        # .model()
        # See for specific ppo config: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo
        # See for more on PPO hyperparameters: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
        .training(
            lr=5e-5,
            use_gae=True,
            use_critic=True,
            # gae_lambda .90 seems to get better performance of uavs landing
            lambda_=0.95,
            train_batch_size=65536,
            gamma=0.99,
            num_sgd_iter=32,
            sgd_minibatch_size=4096,
            vf_clip_param=10.0,
            vf_loss_coeff=0.5,
            clip_param=0.2,
            # entropy_coeff=0.0,
            # # seeing if this solves the error:
            # # https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
            # # Expected parameter loc (Tensor of shape (4096, 3)) of distribution Normal(loc: torch.Size([4096, 3]), scale: torch.Size([4096, 3])) to satisfy the constraint Real(),
            grad_clip=1.0,
            # kl_coeff=1.0,
            # kl_target=0.0068,
        )
        .multi_agent(
            policies={
                "shared_policy": (
                    None,
                    temp_env.observation_space[0],
                    temp_env.action_space[0],
                    {},
                )
            },
            # Always use "shared" policy.
            policy_mapping_fn=(
                lambda agent_id, episode, worker, **kwargs: "shared_policy"
            ),
            # policies_to_train=[""]
        )
        # .reporting(keep_per_episode_custom_metrics=True)
        # .evaluation(
        #     evaluation_interval=10, evaluation_duration=10  # default number of episodes
        # )
    )

    if args.preload is not None:
        # use preload checpoint,
        # https://github.com/ray-project/ray/blob/master/rllib/examples/restore_1_of_n_agents_from_checkpoint.py
        policy_checkpoint = r"/home/prime/Documents/workspace/rl_multi_uav_sim/results/PPO/multi-uav-sim-v0_2023-10-30-05-59_6341c86/beta_0_3_pen_5/PPO_multi-uav-sim-v0_161e9_00003_3_beta=0.3000,d_thresh=0.2000,obstacle_collision_weight=5.0000,tgt_reward=300.0000,uav_collision__2023-10-30_05-59-59/checkpoint_000454/policies/shared_policy"
        policy_checkpoint = args.preload
        restored_policy = Policy.from_checkpoint(policy_checkpoint)
        restored_policy_weights = restored_policy.get_weights()

        class RestoreWeightsCallback(DefaultCallbacks):
            def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs) -> None:
                algorithm.set_weights({"shared_policy": restored_policy_weights})

        callback_list.append(RestoreWeightsCallback)
        # Make sure, the non-1st policies are not updated anymore.
        # config.policies_to_train = [pid for pid in policy_ids if pid != "policy_0"]

    multi_callbacks = make_multi_callbacks(callback_list)
    train_config.callbacks(multi_callbacks)

    stop = {
        "training_iteration": 1 if args.smoke_test else args.stop_iters,
        "timesteps_total": args.stop_timesteps,
    }

    # args.config["env_config"]["use_safe_action"] = tune.grid_search([False])
    # args.config["env_config"]["tgt_reward"] = tune.grid_search([48.0, 25.0, 100.0, 0.0])
    # args.config["env_config"]["beta"] = tune.grid_search([-2.3, 0.1, 0.01])
    # args.config["env_config"]["d_thresh"] = tune.grid_search([-2.01])
    # args.config["env_config"]["uav_collision_weight"] = tune.grid_search([-2.0])
    # args.config["env_config"]["obstacle_collision_weight"] = tune.grid_search([-2.0])

    config = {
        # Special flag signalling `my_train_fn` how many iters to do.
        "train-iterations": 2,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        # "num_workers": 0,
        # "framework": args.framework,
    }

    train_config = train_config.to_dict()
    train_config.update(config)

    resources = PPO.default_resource_request(train_config)
    tuner = tune.Tuner(
        tune.with_resources(my_train_fn, resources=resources),
        param_space=train_config,
        # run_config=air.RunConfig(
        #     stop=stop,
        #     local_dir=args.log_dir,
        #     name=args.name,
        #     # checkpoint_config=air.CheckpointConfig(
        #     #     # num_to_keep=150,
        #     #     # checkpoint_score_attribute="",
        #     #     # checkpoint_at_end=True,
        #     #     checkpoint_frequency=5,
        #     # ),
        # ),
    )

    tuner.fit()

    ray.shutdown()
