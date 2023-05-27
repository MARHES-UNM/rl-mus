from time import time
from matplotlib import pyplot as plt
import numpy as np
from uav_sim.envs.uav_sim import UavSim
from pathlib import Path

import os
import concurrent.futures
from functools import partial
import logging
import json


formatter = "%(asctime)s: %(name)s - %(levelname)s - <%(module)s:%(funcName)s:%(lineno)d> - %(message)s"
logging.basicConfig(
    # filename=os.path.join(app_log_path, log_file_name),
    format=formatter
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

max_num_episodes = 3
max_num_cpus = os.cpu_count() - 1


def experiment(config={}, output_folder="", max_num_episodes=1, experiment_num=0):
    tf = 20.0
    N = 1.0
    env = UavSim(config)

    obs, done = env.reset(), False

    actions = {}
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
        "uav_done": 0.0,
    }

    num_episodes = 0
    obs, dones = env.reset(), {i.id: False for i in env.uavs}
    dones["__all__"] = False

    logger.debug("running experiment")
    # logger.debug(f"{config}")
    num_episodes = 0
    start_time = time()

    # while num_episodes < max_num_episodes:
    #     actions = {}
    #     pos_er = np.zeros((env.num_uavs, 12))
    #     t = env.time_elapsed
    #     for idx in range(env.num_uavs):
    #         des_pos = np.zeros(12)
    #         des_pos[0:6] = env.uavs[idx].pad.state[0:6]
    #         pos_er[idx, :] = des_pos[0:12] - env.uavs[idx].state

    #         t0 = min(t, tf - 0.1)
    #         t_go = (tf - t0) ** N
    #         p = env.uavs[idx].get_p_mat(tf, N, t0)
    #         B = np.zeros((2, 1))
    #         B[1, 0] = 1.0
    #         actions[idx] = t_go * np.array(
    #             [
    #                 B.T @ p[-1].reshape((2, 2)) @ pos_er[idx, [0, 3]],
    #                 B.T @ p[-1].reshape((2, 2)) @ pos_er[idx, [1, 4]],
    #                 B.T @ p[-1].reshape((2, 2)) @ pos_er[idx, [2, 5]],
    #             ]
    #         )

    #     obs, rew, done, info = env.step(actions)
    #     for k, v in info.items():
    #         results["uav_done"] += v["uav_landed"]
    #         results["uav_collision"] += v["uav_collision"]
    #         results["obs_collision"] += v["obstacle_collision"]

    #         time_step_list[k].append(v["time_step"])
    #         uav_collision_list[k].append(v["uav_collision"])
    #         obstacle_collision_list[k].append(v["obstacle_collision"])
    #         uav_done_list[k].append(v["uav_landed"])
    #         rel_dist = np.linalg.norm(pos_er[k, 0:3])
    #         rel_vel = np.linalg.norm(pos_er[k, 3:6])
    #         rel_pad_dist[k].append(rel_dist)
    #         rel_pad_vel[k].append(rel_vel)
    #     # env.render()

    #     if done["__all__"]:
    #         num_episodes += 1
    #         results["num_episodes"] = num_episodes
    #         if num_episodes == max_num_episodes:
    #             end_time = time() - start_time
    #             break
    #         obs, dones = env.reset(), {agent.id: False for agent in env.uavs}
    #         dones["__all__"] = False
    #         env.reset()

    logger.debug("close env")
    env.close()

    # time_step_list = np.array(time_step_list)
    # uav_collision_list = np.array(uav_collision_list)
    # obstacle_collision_list = np.array(obstacle_collision_list)
    # uav_done_list = np.array(uav_done_list)
    # rel_pad_dist = np.array(rel_pad_dist)
    # rel_pad_vel = np.array(rel_pad_vel)

    # all_axes = []
    # all_figs = []
    # fig = plt.figure(figsize=(10, 6))
    # all_axes.append(fig.add_subplot(121))
    # all_axes.append(fig.add_subplot(122))
    # all_figs.append(fig)

    # fig = plt.figure(figsize=(10, 6))
    # all_axes.append(fig.add_subplot(111))
    # all_figs.append(fig)

    # fig = plt.figure(figsize=(10, 6))
    # all_axes.append(fig.add_subplot(121))
    # all_axes.append(fig.add_subplot(122))
    # all_figs.append(fig)

    # for idx in range(env.num_uavs):
    #     all_axes[0].plot(
    #         time_step_list[idx],
    #         uav_collision_list[idx],
    #         label=f"uav_{env.uavs[idx].id}",
    #     )
    #     all_axes[1].plot(
    #         time_step_list[idx],
    #         obstacle_collision_list[idx],
    #         label=f"uav_{env.uavs[idx].id}",
    #     )
    #     all_axes[2].plot(
    #         time_step_list[idx],
    #         uav_done_list[idx],
    #         label=f"uav_{env.uavs[idx].id}",
    #     )
    #     all_axes[3].plot(
    #         time_step_list[idx],
    #         rel_pad_dist[idx],
    #         label=f"uav_{env.uavs[idx].id}",
    #     )
    #     all_axes[4].plot(
    #         time_step_list[idx],
    #         rel_pad_vel[idx],
    #         label=f"uav_{env.uavs[idx].id}",
    #     )

    # all_axes[0].set_ylabel("UAV collisions")
    # all_axes[1].set_ylabel("NCFO collisions")
    # all_axes[2].set_ylabel("UAV landed")
    # all_axes[3].set_ylabel("$\parallel \Delta \mathbf{r} \parallel$")
    # all_axes[4].set_ylabel("$\parallel \Delta \mathbf{v} \parallel$")

    # for ax_ in all_axes:
    #     ax_.set_xlabel("t (s)")

    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    plt_prefix = {
        "tgt_v": config["target_v"],
        "sa": config["use_safe_action"],
        "obs": config["num_obstacles"],
        "seed": config["seed"],
    }
    plt_prefix = "_".join([f"{k}_{str(int(v))}" for k, v in plt_prefix.items()])

    suffixes = ["collisions.png", "landed.png", "r_v_time.png"]

    # for fig_, suffix in zip(all_figs, suffixes):
    #     file_name = output_folder / f"{plt_prefix}_{suffix}"
    #     fig_.savefig(file_name)
    #     plt.close(fig_)

    # if write_experiment:
    fname = output_folder / f"exp_{experiment_num}_{plt_prefix}_result.json"
    results["config"] = config
    # results["time_total_s"] = end_time
    with open(fname, "w") as f:
        json.dump(results, f)
    # labels = [*all_axes[0].get_legend_handles_labels()]
    # return labels

    logger.debug("done")
    return 0


def main():
    target_v = [0.0, 1.0]
    use_safe_action = [False, True]
    num_obstacles = [20, 30]

    # target_v = [0.0]
    # use_safe_action = [False]
    # num_obstacles = [1]

    output_folder = Path(r"/home/prime/Documents/workspace/uav_sim/results")
    output_folder = output_folder / r"images"

    starter = partial(experiment)
    err_msg = "running "

    future_run_experiment = []
    experiment_num = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_cpus) as executor:
        for target in target_v:
            for action in use_safe_action:
                for num_obstacle in num_obstacles:
                    config = {
                        "target_v": target,
                        "num_uavs": 4,
                        "use_safe_action": action,
                        "num_obstacles": num_obstacle,
                        "max_time": 30.0,
                        "seed": 0,
                    }
                    # labels = experiment(config, output_folder)

                    future_run_experiment.append(
                        # executor.submit(
                        #     partial(
                        #         experiment,
                        #         config=config,
                        #         output_folder=output_folder,
                        #         max_num_episodes=1,
                        #         experiment_num=experiment_num
                        #     )
                        # )
                        executor.submit(
                            starter, config, output_folder, 1, experiment_num
                        )
                    )
                    experiment_num += 1

        for future in concurrent.futures.as_completed(future_run_experiment):
            labels = future.result()

    # figsize = (10, 3)
    # fig_leg = plt.figure(figsize=figsize)
    # ax_leg = fig_leg.add_subplot(111)
    # # add the legend from the previous axes
    # ax_leg.legend(*labels, loc="center", ncol=len(labels[1]))
    # # hide the axes frame and the x/y labels
    # ax_leg.axis("off")
    # fig_leg.savefig(output_folder / "labels.png")

    # plt.show()


if __name__ == "__main__":
    main()
