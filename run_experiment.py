import argparse
from datetime import datetime
import subprocess
from time import time
from matplotlib import pyplot as plt
import numpy as np
from uav_sim.envs.uav_sim import UavSim
from pathlib import Path

import os
import logging
import json

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
    # max_num_episodes = exp_config.setdefault("max_num_episodes", 1)
    write_experiment = exp_config.setdefault("write_experiment", False)
    # experiment_num = exp_config.setdefault("experiment_num", 0)
    env_config = exp_config["env_config"]
    render = exp_config["render"]

    env = UavSim(env_config)
    N = env.t_go_n
    tf = env.time_final

    time_step_list = [[] for idx in range(env.num_uavs)]
    uav_collision_list = [[] for idx in range(env.num_uavs)]
    obstacle_collision_list = [[] for idx in range(env.num_uavs)]
    uav_done_list = [[] for idx in range(env.num_uavs)]
    rel_pad_dist = [[] for idx in range(env.num_uavs)]
    rel_pad_vel = [[] for idx in range(env.num_uavs)]

    results = {
        "num_episodes": 0.0,
        "uav_collision": 0.0,
        "obs_collision": 0.0,
        "uav_done": [[] for idx in range(env.num_uavs)],
        "uav_done_time": [[] for idx in range(env.num_uavs)],
        "episode_time": [],
        "episode_data": {
            "time_step_list": [],
            "time_step_list": [],
            "uav_collision_list": [],
            "obstacle_collision_list": [],
            "uav_done_list": [],
            "rel_pad_dist": [],
            "rel_pad_vel": [],
        },
    }

    num_episodes = 0
    obs, done = env.reset(), {i.id: False for i in env.uavs}
    done["__all__"] = False

    logger.debug("running experiment")
    num_episodes = 0
    start_time = time()

    while num_episodes < max_num_episodes:
        pos_er = np.zeros((env.num_uavs, 12))
        t = env.time_elapsed
        actions = {}
        for idx in range(env.num_uavs):
            des_pos = np.zeros(12)
            des_pos[0:6] = env.uavs[idx].pad.state[0:6]
            pos_er[idx, :] = des_pos[0:12] - env.uavs[idx].state

            t0 = min(t, tf - 0.1)
            t_go = (tf - t0) ** N
            p = env.uavs[idx].get_p_mat(tf, N, t0)
            B = np.zeros((2, 1))
            B[1, 0] = 1.0
            actions[idx] = t_go * np.array(
                [
                    B.T @ p[-1].reshape((2, 2)) @ pos_er[idx, [0, 3]],
                    B.T @ p[-1].reshape((2, 2)) @ pos_er[idx, [1, 4]],
                    B.T @ p[-1].reshape((2, 2)) @ pos_er[idx, [2, 5]],
                ]
            )

        obs, rew, done, info = env.step(actions)
        for k, v in info.items():
            # results["uav_done"][k] = v["uav_landed"]
            results["uav_collision"] += v["uav_collision"]
            results["obs_collision"] += v["obstacle_collision"]

            time_step_list[k].append(v["time_step"])
            uav_collision_list[k].append(v["uav_collision"])
            obstacle_collision_list[k].append(v["obstacle_collision"])
            uav_done_list[k].append(v["uav_landed"])
            rel_dist = np.linalg.norm(pos_er[k, 0:3])
            rel_vel = np.linalg.norm(pos_er[k, 3:6])
            rel_pad_dist[k].append(rel_dist)
            rel_pad_vel[k].append(rel_vel)

        if render:
            env.render()

        if done["__all__"]:
            num_episodes += 1
            for k, v in info.items():
                results["uav_done"][k].append(v["uav_landed"])
                results["uav_done_time"][k].append(v["uav_done_time"])
            results["num_episodes"] = num_episodes
            results["episode_time"].append(env.time_elapsed)
            results["episode_data"]["time_step_list"].append(time_step_list)
            results["episode_data"]["uav_collision_list"].append(uav_collision_list)
            results["episode_data"]["obstacle_collision_list"].append(
                obstacle_collision_list
            )
            results["episode_data"]["uav_done_list"].append(uav_done_list)
            results["episode_data"]["rel_pad_dist"].append(rel_pad_dist)
            results["episode_data"]["rel_pad_vel"].append(rel_pad_vel)

            if num_episodes == max_num_episodes:
                end_time = time() - start_time
                break
            obs, done = env.reset(), {agent.id: False for agent in env.uavs}
            done["__all__"] = False
            env.reset()

            # reinitialize data arrays
            time_step_list = [[] for idx in range(env.num_uavs)]
            uav_collision_list = [[] for idx in range(env.num_uavs)]
            obstacle_collision_list = [[] for idx in range(env.num_uavs)]
            uav_done_list = [[] for idx in range(env.num_uavs)]
            rel_pad_dist = [[] for idx in range(env.num_uavs)]
            rel_pad_vel = [[] for idx in range(env.num_uavs)]

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
        results["env_config"] = env.env_config
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
    parser.add_argument("--render", action="store_true", default=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    if args.load_config:
        with open(args.load_config, "rt") as f:
            args.config = json.load(f)

    logger.debug(f"config: {args.config}")

    # branch_hash = (
    #     subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    #     .strip()
    #     .decode()
    # )

    # dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    # if not args.log_dir:
    #     args.log_dir = f"./results/uas_{dir_timestamp}_{branch_hash}"

    max_num_episodes = args.max_num_episodes
    experiment_num = args.experiment_num
    args.config["render"] = args.render

    # output_folder = Path(args.log_dir)

    # exp_config = {
    #     "env_config": {
    #         "target_v": 0.0,
    #         "num_uavs": 4,
    #         "use_safe_action": False,
    #         "num_obstacles": 30,
    #         "max_time": 30.0,
    #         "seed": 0,
    #     }
    # }
    experiment(args.config, max_num_episodes, experiment_num)

    # experiment(
    #     exp_config, output_folder, max_num_episodes, experiment_num=experiment_num
    # )


if __name__ == "__main__":
    main()
