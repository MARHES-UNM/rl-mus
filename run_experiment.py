import argparse
import cmath
from datetime import datetime
from time import time
from turtle import color
from matplotlib import pyplot as plt
import numpy as np
import ray
from uav_sim.envs.uav_sim import UavSim
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig

import os
import logging
import json
from uav_sim.utils.safety_layer import SafetyLayer
from ray import tune
from plot_results import plot_uav_states

from uav_sim.utils.utils import get_git_hash

PATH = Path(__file__).parent.absolute().resolve()

formatter = "%(asctime)s: %(name)s - %(levelname)s - <%(module)s:%(funcName)s:%(lineno)d> - %(message)s"
logging.basicConfig(
    # filename=os.path.join(app_log_path, log_file_name),
    format=formatter
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

max_num_cpus = os.cpu_count() - 1


def experiment(exp_config={}, max_num_episodes=1, experiment_num=0):
    fname = exp_config.setdefault("fname", None)
    write_experiment = exp_config.setdefault("write_experiment", False)
    env_config = exp_config["env_config"]
    render = exp_config["render"]
    plot_results = exp_config["plot_results"]

    env = UavSim(env_config)

    algo_to_run = exp_config["exp_config"].setdefault("run", "PPO")
    if algo_to_run == "PPO":
        checkpoint = exp_config["exp_config"].setdefault("checkpoint", None)

        # Reload the algorithm as is from training.
        # if checkpoint is not None:
        # algo = Algorithm.from_checkpoint(checkpoint)

        # ray.init(num_cpus=1, num_gpus=0)
        algo = (
            PPOConfig()
            .environment(
                env=exp_config["env_name"], env_config=exp_config["env_config"]
            )
            .framework("torch")
            .rollouts(num_rollout_workers=0)
            .debugging(log_level="ERROR", seed=exp_config["env_config"]["seed"])
            .resources(num_gpus=0)
            .multi_agent(
                policies={
                    "shared_policy": (
                        None,
                        env.observation_space[0],
                        env.action_space[0],
                        {},
                    )
                },
                # Always use "shared" policy.
                policy_mapping_fn=(
                    lambda agent_id, episode, worker, **kwargs: "shared_policy"
                ),
            )
            .build()
        )

        if checkpoint is not None:
            algo.restore(checkpoint)

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
    obstacle_state = [[] for idx in range(env.num_obstacles)]
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

            for obs_idx in range(env.num_obstacles):
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_config", default=f"{PATH}/configs/sim_config.cfg")
    parser.add_argument(
        "--log_dir",
    )
    parser.add_argument("-d", "--debug")
    parser.add_argument("-v", help="version number of experiment")
    parser.add_argument("--max_num_episodes", type=int, default=1)
    parser.add_argument("--experiment_num", type=int, default=0)
    parser.add_argument("--env_name", type=str, default="multi-uav-sim-v0")
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--write_exp", action="store_true")
    parser.add_argument("--plot_results", action="store_true", default=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    if args.load_config:
        with open(args.load_config, "rt") as f:
            args.config = json.load(f)

    if not args.config["exp_config"]["run"] == "cc":
        if args.env_name == "multi-uav-sim-v0":
            args.config["env_name"] = args.env_name
            tune.register_env(
                args.config["env_name"],
                lambda env_config: UavSim(env_config=env_config),
            )

    logger.debug(f"config: {args.config}")

    if not args.log_dir:
        branch_hash = get_git_hash()

        dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        args.log_dir = f"./results/test_results/exp_{dir_timestamp}_{branch_hash}"

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


if __name__ == "__main__":
    main()
