import argparse
import json
import logging
import os
import pathlib
import subprocess
from datetime import datetime
import numpy as np

from ray import tune

from uav_sim.envs.uav_sim import UavSim
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
    # config["safety_layer_cfg"]["replay_buffer_size"] = 1000
    # config["safety_layer_cfg"]["episode_length"] = 10
    # config["safety_layer_cfg"]["num_epochs"] = 10

    if checkpoint_dir:
        config["safety_layer_cfg"]["checkpoint_dir"] = os.path.join(
            checkpoint_dir, "checkpoint"
        )

    safe_action_layer = SafetyLayer(env, config["safety_layer_cfg"])

    safe_action_layer.train()


# TODO: implement function to test safe action
# def test_safe_action(config):
#     num_iterations = int(config.get("num_iterations", 100))
#     tune_run = config.get("tune_run", False)
#     config["env_config"]["seed"] = None
#     env = CuasEnvMultiAgentV1(config["env_config"])

#     config["safety_layer_cfg"][
#         "checkpoint_dir"
#     ] = r"/home/marcus/Documents/workspace/cuas/results/safe_action/safe_action_2022-11-06-01-56_63424d7/safe_action_layer/train_safe_action_2163c_00000_0_2022-11-06_01-56-25/checkpoint_000045/checkpoint"

#     safe_action_layer = SafeActionLayer(env, config["safety_layer_cfg"])

#     obs, dones = env.reset(), {i.id: False for i in env.agents}
#     dones["__all__"] = False

#     results = {
#         "episode_reward": 0,
#         "timesteps_total": 0,
#         "agent_collisions": 0,
#         "target_collisions": 0,
#         "target_breached": 0,
#         "obstacle_collisions": 0,
#     }

#     logger.debug("running experiment")
#     for _ in range(num_iterations):
#         actions = env.action_space.sample()

#         if config["use_safe_action"]:
#             # print("using safe action")
#             for agent_id, action in actions.items():
#                 safe_action = safe_action_layer.get_action(
#                     obs[agent_id]["observations"]
#                 )
#                 action += safe_action

#                 actions[agent_id] = np.clip(action, -1, 1)

#         # actions = {}
#         # for agent in env.agents:
#         #     actions[agent.id] = ppo_agent.compute_single_action(
#         #         observation=obs[agent.id], policy_id="pursuer"
#         #     )

#         obs, rewards, dones, infos = env.step(actions)
#         results["episode_reward"] += sum([v for k, v in rewards.items()])
#         results["agent_collisions"] += sum(
#             [v["agent_collision"] for k, v in infos.items()]
#         )
#         results["target_collisions"] += sum(
#             [v["target_collision"] for k, v in infos.items()]
#         )
#         results["obstacle_collisions"] += sum(
#             [v["obstacle_collision"] for k, v in infos.items()]
#         )
#         results["target_breached"] += sum(
#             [v["target_breached"] for k, v in infos.items()]
#         )

#         results["timesteps_total"] += 1

#         env.render()

#         if dones["__all__"]:
#             if tune_run:
#                 # print("reporting tune")
#                 tune.report(**results)
#             obs, dones = env.reset(), {agent.id: False for agent in env.agents}
#             dones["__all__"] = False
#             results["episode_reward"] = 0


def train(args):
    results = tune.run(
        train_safety_layer,
        stop={
            # "timesteps_total": args.num_timesteps,
            "training_iteration": args.config["safety_layer_cfg"]["num_epochs"],
            "time_total_s": args.duration,
        },
        # num_samples=10,
        resources_per_trial={"cpu": 6, "gpu": 1},
        config=args.config,
        # checkpoint_freq=10,
        # checkpoint_at_end=True,
        local_dir=args.log_dir,
        name=args.name,
        restore=args.restore,
        resume=args.resume,
    )


def test(args):
    pass


# def test(args):
#     args.config["tune_run"] = args.tune_run
#     args.config["use_safe_action"] = tune.grid_search([True, False])
#     if args.tune_run:
#         results = tune.run(
#             test_safe_action,
#             stop={
#                 # "timesteps_total": 20,
#                 #     "training_iteration": 100,
#                 #     "time_total_s": args.duration,
#             },
#             config=args.config,
#             local_dir=args.log_dir,
#             name=args.name,
#             resources_per_trial={"cpu": 1, "gpu": 0},
#         )
#     else:
#         test_safe_action(args.config)


# # def test(args):
# #     args.config["tune_run"] = args.tune_run

# #     if args.experiment:
# #         args.config["analysis"] = ExperimentAnalysis(args.experiment)
# #         if not args.tune_run:
# #             args.config["trial"] = args.config["analysis"].trials[1]
# #         else:
# #             args.config["trial"] = tune.grid_search(
# #                 [trial for trial in args.config["analysis"].trials]
# #             )
# #     elif args.checkpoint:
# #         args.config["checkpoint"] = args.checkpoint
# #         args.config["restore_checkpoint"] = True
# #     else:
# #         args.config["restore_checkpoint"] = False

# #     # args.config["train_config"]["model"]["custom_model_config"][
# #     #     "use_safe_action"
# #     # ] = tune.grid_search([True, False])
# #     # args.config["env_config"]["use_safety_layer"] = tune.grid_search([True, False])
# #     # args.config["env_config"]["safety_layer_type"] = tune.grid_search(["hard", "soft"])
# #     args.config["env_config"]["use_safe_action"] = tune.grid_search([True, False])
# #     # args.config["env_config"]["use_safe_action"] = tune.grid_search([True])
# #     # args.config["env_config"]["constraint_k"] = 1
# #     # args.config["env_config"]["constraint_slack"] = 0.2

# #     # args.config["env_config"]["use_safe_action"] = tune.grid_search([False, True])
# #     # args.config["env_config"]["constraint_slack"] = tune.loguniform(0, 10)
# #     # args.config["env_config"]["constraint_k"] = tune.loguniform(0, 100)
# #     if args.tune_run:
# #         results = tune.run(
# #             experiment,
# #             stop={
# #                 # "timesteps_total": 20,
# #                 #     "training_iteration": 100,
# #                 #     "time_total_s": args.duration,
# #             },
# #             config=args.config,
# #             # checkpoint_freq=10,
# #             checkpoint_at_end=True,
# #             local_dir=args.log_dir,
# #             name=args.name,
# #             resources_per_trial={"cpu": 1, "gpu": 0},
# #         )
# #     else:
# #         experiment(args.config)


def main():
    args = parse_arguments()

    if args.load_config:
        with open(args.load_config, "rt") as f:
            args.config = json.load(f)

    logger.debug(f"config: {args.config}")
    # config = Config(args.config)

    # env_config = config.env_config

    # env_suffix = (
    #     f"{env_config.num_pursuers}v{env_config.num_evaders}o{env_config.num_obstacles}"
    # )
    # env_name = f"cuas_multi_agent-v1_{env_suffix}"

    if not args.log_dir:
        branch_hash = get_git_hash()
        dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

        args.log_dir = (
            f"./results/safety_layer/safety_layer{dir_timestamp}_{branch_hash}"
        )

    # # Must register by passing env_config if wanting to grid search over environment variables
    # args.config["env_name"] = env_name
    # tune.register_env(env_name, lambda env_config: CuasEnvMultiAgentV1(env_config))

    # https://stackoverflow.com/questions/27529610/call-function-based-on-argparse
    args.func(args)


if __name__ == "__main__":
    main()
