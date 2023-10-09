import argparse
import json
import logging
import os
import pathlib
from datetime import datetime
import numpy as np

from ray import tune

# from ray.air import Checkpoint, session

from uav_sim.envs.uav_sim import UavSim
from uav_sim.utils import safety_layer
from uav_sim.utils.safety_layer import SafetyLayer
from uav_sim.utils.utils import get_git_hash

formatter = "%(asctime)s: %(name)s - %(levelname)s - <%(module)s:%(funcName)s:%(lineno)d> - %(message)s"
logging.basicConfig(
    # filename=os.path.join(app_log_path, log_file_name),
    format=formatter
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PATH = pathlib.Path(__file__).parent.absolute().resolve()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load_config",
        help="Load configuration for experiment.",
        default=f"{PATH}/configs/sim_config.cfg",
    )
    parser.add_argument("--name", type=str, help="experiment name", default="debug")
    parser.add_argument("--log_dir", type=str)

    subparsers = parser.add_subparsers(dest="command")
    train_sub = subparsers.add_parser("train")
    train_sub.add_argument("--duration", type=int, default=5 * 24 * 60 * 60)
    train_sub.add_argument("--num_timesteps", type=int, default=int(5e6))
    train_sub.add_argument("--training_iteration", type=int, default=int(5))
    train_sub.add_argument("--restore", type=str, default=None)
    train_sub.add_argument("--resume", action="store_true", default=False)
    train_sub.set_defaults(func=train)

    test_sub = subparsers.add_parser("test")
    checkpoint_or_experiment = test_sub.add_mutually_exclusive_group()
    checkpoint_or_experiment.add_argument("--checkpoint", type=str)
    checkpoint_or_experiment.add_argument("--experiment", type=str)
    test_sub.add_argument("--tune_run", action="store_true", default=False)
    test_sub.add_argument("--max_num_episodes", type=int, default=100)
    test_sub.set_defaults(func=test)

    args = parser.parse_args()

    return args


def train_safety_layer(config, checkpoint_dir=None):
    env = UavSim(config["env_config"])
    config["safety_layer_cfg"]["report_tune"] = True

    # # for use with ray air session
    # checkpoint = session.get_checkpoint()
    # config["safety_layer_cfg"]["checkpoint"] = checkpoint

    if checkpoint_dir:
        config["safety_layer_cfg"]["checkpoint_dir"] = os.path.join(
            checkpoint_dir, "checkpoint"
        )

    safe_action_layer = SafetyLayer(env, config["safety_layer_cfg"])

    safe_action_layer.train()


def train(args):
    args.config["safety_layer_cfg"]["device"] = "cuda"
    args.config["env_config"]["target_v"] = tune.grid_search([0.0])
    args.config["env_config"]["num_obstacles"] = 4
    args.config["env_config"]["max_num_obstacles"] = 4
    args.config["safety_layer_cfg"]["loss_action_weight"] = tune.grid_search(
        [0.1]
        # [1.0, 0.8, 0.5]
        # [1.0, 0.8, 0.5]
    )
    args.config["env_config"]["obstacle_radius"] = tune.grid_search([1.0])
    args.config["safety_layer_cfg"]["num_training_steps"] = 6000
    args.config["safety_layer_cfg"]["num_epochs"] = 200
    # args.config["safety_layer_cfg"]["num_iter_per_epoch"] = tune.grid_search([50])
    args.config["safety_layer_cfg"]["lr"] = 5e-4

    # args.config["safety_layer_cfg"]["eps"] = tune.loguniform(0.003, 0.2)
    # args.config["safety_layer_cfg"]["eps_deriv"] = tune.loguniform(0.003, 0.2)
    # args.config["safety_layer_cfg"]["loss_action_weight"] = tune.loguniform(1e-3, 0.2)

    args.config["safety_layer_cfg"]["eps"] = tune.grid_search([0.0, 0.01, 0.1])
    args.config["safety_layer_cfg"]["eps_deriv"] = tune.grid_search([0.0, 0.01, 0.1])
    args.config["safety_layer_cfg"]["eps_action"] = tune.grid_search([0.0, 0.01, 0.1])
    args.config["safety_layer_cfg"]["num_iter_per_epoch"] = 100
    args.config["safety_layer_cfg"]["batch_size"] = tune.grid_search([1024])

    # tune.grid_search(
    #     # [128, 256, 512, 1024]
    #     [256]
    # )

    # args.config["safety_layer_cfg"]["n_hidden"] = tune.grid_search([32])
    # args.config["safety_layer_cfg"]["eps"] = tune.loguniform(1e-5, 1.0)
    # args.config["safety_layer_cfg"]["eps_deriv"] = tune.loguniform(1e-5, 1.0)
    # args.config["safety_layer_cfg"]["num_epochs"] = 10
    # args.config["safety_layer_cfg"]["replay_buffer_size"] = 64 * 10
    # args.config["safety_layer_cfg"]["batch_size"] = 32
    # args.config["safety_layer_cfg"]["num_eval_steps"] = 64
    # args.config["safety_layer_cfg"]["num_training_steps"] = 64
    # args.config["safety_layer_cfg"]["eps"] = tune.grid_search([0.0001, 0.01])
    # args.config["safety_layer_cfg"]["eps_deriv"] = tune.grid_search([0.000042737, 0.01])
    # args.config["safety_layer_cfg"]["eps_action"] = 0.00019
    # args.config["safety_layer_cfg"]["lr"] = 0.00407
    # args.config["safety_layer_cfg"]["weight_decay"] = 0.0042277
    # args.config["safety_layer_cfg"]["loss_action_weight"] = 0.1
    # args.config["safety_layer_cfg"]["lr"] = tune.loguniform(5e-4, 0.01)
    # args.config["safety_layer_cfg"]["weight_decay"] = tune.loguniform(1e-5, 0.01)

    # TODO: implement with Ray session air
    # https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
    results = tune.run(
        train_safety_layer,
        stop={
            # "timesteps_total": args.num_timesteps,
            "training_iteration": args.config["safety_layer_cfg"]["num_epochs"],
            "time_total_s": args.duration,
        },
        # num_samples=32,
        resources_per_trial={"cpu": 1, "gpu": 0.25},
        config=args.config,
        # checkpoint_freq=5,
        # checkpoint_at_end=True,
        local_dir=args.log_dir,
        name=args.name,
        restore=args.restore,
        resume=args.resume,
    )


def test(args):
    args.config["tune_run"] = args.tune_run
    # args.config["use_safe_action"] = tune.grid_search([True, False])
    args.config["use_safe_action"] = True
    args.config["env_config"]["target_v"] = 0.0
    args.config["env_config"]["seed"] = 999
    args.config["safety_layer_cfg"]["seed"] = 999

    if args.tune_run:
        results = tune.run(
            test_safe_action,
            stop={
                # "timesteps_total": 20,
                #     "training_iteration": 100,
                #     "time_total_s": args.duration,
            },
            config=args.config,
            local_dir=args.log_dir,
            name=args.name,
            resources_per_trial={"cpu": 1, "gpu": 0},
        )
    else:
        test_safe_action(args.config)


def test_safe_action(config):
    num_iterations = int(config.get("num_iterations", 400))
    tune_run = config.get("tune_run", False)
    # config["env_config"]["seed"] = None
    # config["env_config"]["num_obstacles"] = 1

    env = UavSim(config["env_config"])
    # render = config["render"]
    # plot_results = config["plot_results"]

    # config["safety_layer_cfg"][
    # "checkpoint_dir"
    # ] = r"/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-08-27-17-47_07e3223/debug/train_safety_layer_50b66_00010_10_eps_action=0.0002,eps_dang=0.0007,eps_deriv=0.0001,eps_safe=0.0049,loss_action_weight=0.2435,lr=_2023-08-27_19-16-48/checkpoint_000035/checkpoint"
    config["safety_layer_cfg"][
        "checkpoint_dir"
        # ] = r"/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-08-28-23-23_b13e4e3/debug/train_safety_layer_7e25e_00072_72_eps_action=0.0002,eps_dang=0.1414,eps_deriv=0.0000,eps_safe=0.0185,loss_action_weight=0.7270,lr=_2023-08-29_15-00-23/checkpoint_000045/checkpoint"
        # ] = r"/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-09-01-06-54_6a6ba7e/debug/train_safety_layer_00757_00011_11_eps=0.0100,eps_deriv=0.0100,lr=0.0020,weight_decay=0.0001_2023-09-01_17-58-53/checkpoint_000244/checkpoint"
        # ] = r"/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-09-01-06-54_6a6ba7e/debug/train_safety_layer_00757_00017_17_eps=0.0100,eps_deriv=0.0000,lr=0.0013,weight_decay=0.0005_2023-09-01_23-56-57/checkpoint_000244/checkpoint"
        # ] = r"/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-09-04-00-12_fd3b073/debug/train_safety_layer_545fd_00007_7_num_obstacles=8,target_v=1.0000,loss_action_weight=0.0800_2023-09-04_04-48-25/checkpoint_000199/checkpoint"
    ] = r"/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-09-09-14-08_d056fbe/small_1_dist/train_safety_layer_d1e8e_00001_1_batch_size=128,eps=0.1000,eps_deriv=0.0300,loss_action_weight=0.0000,lr=0.0005,n_hidden=32,num_it_2023-09-09_14-08-05/checkpoint_000024/checkpoint"

    safe_layer = SafetyLayer(env, config["safety_layer_cfg"])

    obs, dones = env.reset(), {uav.id: False for uav in env.uavs}
    dones["__all__"] = False

    results = {
        "timesteps_total": 0.0,
        "uav_collision": 0.0,
        "obs_collision": 0.0,
        "uav_done": 0.0,
        "uav_done_time": 0.0,
    }

    logger.debug("running experiment")
    for _ in range(num_iterations):
        actions = env.action_space.sample()

        for uav_id, action in actions.items():
            action = env.get_time_coord_action(env.uavs[uav_id])

            # computer safe action with default from nominal action
            if config["use_safe_action"]:
                action = safe_layer.get_action(obs[uav_id], action)

            actions[uav_id] = action

        obs, rewards, dones, infos = env.step(actions)
        results["uav_collision"] += sum([v["uav_collision"] for k, v in infos.items()])
        results["obs_collision"] += sum(
            [v["obstacle_collision"] for k, v in infos.items()]
        )
        results["timesteps_total"] += 1

        env.render()

        if dones["__all__"]:
            for k, v in infos.items():
                results["uav_done"] += v["uav_landed"]
                results["uav_done_time"] += v["uav_done_time"]
            if tune_run:
                tune.report(**results)
            obs = env.reset()
            dones["__all__"] = False


def main():
    args = parse_arguments()

    if args.load_config:
        with open(args.load_config, "rt") as f:
            args.config = json.load(f)

    logger.debug(f"config: {args.config}")

    if not args.log_dir:
        branch_hash = get_git_hash()
        dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

        args.log_dir = (
            f"./results/safety_layer/safety_layer{dir_timestamp}_{branch_hash}"
        )

    # https://stackoverflow.com/questions/27529610/call-function-based-on-argparse
    args.func(args)


if __name__ == "__main__":
    main()
