import argparse
from datetime import datetime
import subprocess
from time import time
from matplotlib import pyplot as plt
import numpy as np
import ray
from uav_sim.envs.uav_sim import UavSim
from pathlib import Path
import mpl_toolkits.mplot3d.art3d as art3d
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

import os
import logging
import json
from uav_sim.utils.safety_layer import SafetyLayer
from ray import tune

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
        # if not ray.is_initialized():
        #     ray.init(include_dashboard=False)
        checkpoint = exp_config["exp_config"].setdefault("checkpoint", None)
        
        # Reload the algorithm as is from training. 
        # if checkpoint is not None:
            # algo = Algorithm.from_checkpoint(checkpoint)

        algo = (
            PPOConfig()
            .framework("torch")
            .environment(env=exp_config["env_name"])
            .resources(num_gpus=0)
            .rollouts(num_rollout_workers=0)
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

            time_step_list[k].append(v["time_step"])
            uav_collision_list[k].append(v["uav_collision"])
            obstacle_collision_list[k].append(v["obstacle_collision"])
            uav_done_list[k].append(v["uav_landed"])
            uav_done_dt_list[k].append(v["uav_done_dt"])
            uav_dt_go_list[k].append(v["uav_dt_go"])
            rel_pad_dist[k].append(v["uav_rel_dist"])
            rel_pad_vel[k].append(v["uav_rel_vel"])
            uav_reward[k].append(rew[k])

        for k, v in obs.items():
            uav_state[k].append(v["state"].tolist())
            target_state.append(v["target"].tolist())

        if render:
            env.render()

        if done["__all__"]:
            num_episodes += 1
            for k, v in info.items():
                results["uav_done"][k].append(v["uav_landed"])
                results["uav_done_dt"][k].append(v["uav_done_dt"])
            results["num_episodes"] = num_episodes
            results["episode_time"].append(env.time_elapsed)
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

            if render:
                fig = env.render(done=True)
            if plot_results:
                plot_uav_states(env.num_uavs, results, num_episodes - 1)

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


def plot_uav_states(num_uavs, results, num_episode=0):
    uav_collision_list = results["episode_data"]["uav_collision_list"][num_episode]
    obstacle_collision_list = results["episode_data"]["obstacle_collision_list"][
        num_episode
    ]
    uav_done_list = results["episode_data"]["uav_done_list"][num_episode]
    uav_done_dt_list = results["episode_data"]["uav_done_dt_list"][num_episode]
    uav_dt_go_list = results["episode_data"]["uav_dt_go_list"][num_episode]
    rel_pad_dist = results["episode_data"]["rel_pad_dist"][num_episode]
    rel_pad_vel = results["episode_data"]["rel_pad_vel"][num_episode]
    uav_reward = results["episode_data"]["uav_reward"][num_episode]

    uav_state = np.array(results["episode_data"]["uav_state"])[num_episode]
    target_state = np.array(results["episode_data"]["target_state"])[num_episode]

    axs = []
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    fig = plt.figure(figsize=(10, 6))
    ax21 = fig.add_subplot(411)
    ax22 = fig.add_subplot(412)
    ax23 = fig.add_subplot(413)
    ax24 = fig.add_subplot(414)
    fig = plt.figure(figsize=(10, 6))
    ax3 = fig.add_subplot(211)
    ax4 = fig.add_subplot(212)
    fig = plt.figure(figsize=(10, 6))
    ax5 = fig.add_subplot(111, projection="3d")
    axs.extend([ax, ax21, ax3, ax4])
    for idx in range(num_uavs):
        ax.plot(uav_collision_list[idx], label=f"uav_id:{idx}")
        ax.title.set_text("uav collision")
        ax1.plot(obstacle_collision_list[idx], label=f"uav_id:{idx}")
        ax1.title.set_text("obstacle collision")
        ax21.plot(uav_done_list[idx], label=f"uav_id:{idx}")
        ax21.title.set_text("uav done")
        ax22.plot(uav_done_dt_list[idx], label=f"uav_id:{idx}")
        ax22.title.set_text("uav done delta time")
        ax23.plot(uav_dt_go_list[idx], label=f"uav_id:{idx}")
        ax23.title.set_text("relative vs time_elapsed")
        ax24.plot(uav_reward[idx], label=f"uav_id:{idx}")
        ax24.title.set_text("uav rewrad")
        ax3.plot(rel_pad_dist[idx], label=f"uav_id:{idx}")
        ax3.title.set_text("uav relative dist to target")
        ax4.plot(rel_pad_vel[idx], label=f"uav_id:{idx}")
        # ax4.title.set_text("uav relative velocity to target")
        ax5.plot(
            uav_state[idx, :, 0],
            uav_state[idx, :, 1],
            uav_state[idx, :, 2],
            label=f"uav_id:{idx}",
        )
    ax5.plot(target_state[:, 0], target_state[:, 1], target_state[:, 2], label="target")
    for ax_ in axs:
        ax_.legend()
    plt.legend()
    plt.show()


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
