from datetime import datetime
import json
from ray import air, tune
from ray.tune.registry import register_env
import logging

from uav_sim.utils.utils import get_git_hash
from uav_sim.envs.uav_sim import UavSim
from pathlib import Path
import os
import argparse
import ray
from ray.tune.registry import get_trainable_cls
from ray.rllib.utils import check_env

from ray.rllib.algorithms.callbacks import make_multi_callbacks
from uav_sim.utils.callbacks import TrainCallback
from ray.rllib.examples.env.multi_agent import FlexAgentsMultiAgent


max_num_cpus = os.cpu_count() - 1

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
        default=3000000,
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
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    return args


def train(args):
    args.local_mode = True
    ray.init(local_mode=args.local_mode)
    # ray.init()
    temp_env = UavSim(args.config)
    # temp_env = FlexAgentsMultiAgent()
    # observer_space = temp_env.observation_space[0]
    # action_space = temp_env

    callback_list = [TrainCallback]
    multi_callbacks = make_multi_callbacks(callback_list)
    train_config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(env=args.env_name, env_config=args.config["env_config"])
        .framework(args.framework)
        .callbacks(multi_callbacks)
        .rollouts(
            num_rollout_workers=1,  # set 0 to main worker run sim
            batch_mode="complete_episodes",
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .debugging(log_level="ERROR", seed=123)  # DEBUG, INFO
        .resources(
            num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", args.gpu)),
            num_cpus_per_worker=args.cpu,
        )
        .training(
            lr=8e-5,
            # use_gae=True,
            # use_critic=True,
            # lambda_=0.95,
            # train_batch_size=65536,
            # gamma=0.99,
            # num_sgd_iter=32,
            # sgd_minibatch_size=4096,
            # vf_clip_param=10.0,
            # vf_loss_coeff=0.5,
        )
        .multi_agent(
            # policies={
            #     "shared_policy": (
            #         None,
            #         temp_env.observation_space,
            #         temp_env.action_space,
            #         {},
            #     )
            # },
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

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
    }

    tuner = tune.Tuner(
        args.run,
        param_space=train_config.to_dict(),
        run_config=air.RunConfig(
            stop=stop,
            local_dir=args.log_dir,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True, checkpoint_frequency=30
            ),
        ),
    )

    results = tuner.fit()

    ray.shutdown()


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
    # register_env(args.env_name, lambda _: FlexAgentsMultiAgent() )

    check_env(UavSim(env_config))
    train(args)
