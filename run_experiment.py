from matplotlib import pyplot as plt
import numpy as np
from uav_sim.envs.uav_sim import UavSim


def experiment(config={}):
    tf = 20.0
    N = 1.0
    env = UavSim(
        config
        # {
        #     "target_v": 0.0,
        #     "num_uavs": 4,
        #     "use_safe_action": True,
        #     "num_obstacles": 30,
        #     "max_time": 30.0,
        #     "seed": 0,
        # }
    )

    obs, done = env.reset(), False

    actions = {}
    time_step_list = [[] for idx in range(env.num_uavs)]
    uav_collision_list = [[] for idx in range(env.num_uavs)]
    obstacle_collision_list = [[] for idx in range(env.num_uavs)]
    uav_done_list = [[] for idx in range(env.num_uavs)]
    rel_pad_dist = [[] for idx in range(env.num_uavs)]
    rel_pad_vel = [[] for idx in range(env.num_uavs)]

    while True:
        actions = {}
        pos_er = np.zeros((env.num_uavs, 12))
        t = env.time_elapsed
        for idx in range(env.num_uavs):
            des_pos = np.zeros(12)
            des_pos[0:6] = env.uavs[idx].pad.state[0:6]
            pos_er[idx, :] = des_pos[0:12] - env.uavs[idx].state

            # actions[idx] = np.dot(k_gain, pos_er[0:6])
            t = env.time_elapsed
            r = np.linalg.norm(pos_er[idx, 0:3])
            v = np.linalg.norm(pos_er[idx, 3:6])
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
            time_step_list[k].append(v["time_step"])
            uav_collision_list[k].append(v["uav_collision"])
            obstacle_collision_list[k].append(v["obstacle_collision"])
            uav_done_list[k].append(v["uav_landed"])
            rel_dist = np.linalg.norm(pos_er[k, 0:3])
            rel_vel = np.linalg.norm(pos_er[k, 3:6])
            rel_pad_dist[k].append(rel_dist)
            rel_pad_vel[k].append(rel_vel)
        # env.render()

        if done["__all__"]:
            break

    time_step_list = np.array(time_step_list)
    uav_collision_list = np.array(uav_collision_list)
    obstacle_collision_list = np.array(obstacle_collision_list)
    uav_done_list = np.array(uav_done_list)
    rel_pad_dist = np.array(rel_pad_dist)
    rel_pad_vel = np.array(rel_pad_vel)

    all_axes = []
    fig = plt.figure(figsize=(10, 6))
    all_axes.append(fig.add_subplot(121))
    all_axes.append(fig.add_subplot(122))

    fig = plt.figure(figsize=(10, 6))
    all_axes.append(fig.add_subplot(111))

    fig = plt.figure(figsize=(10, 6))
    all_axes.append(fig.add_subplot(121))
    all_axes.append(fig.add_subplot(122))

    for idx in range(env.num_uavs):
        all_axes[0].plot(
            time_step_list[idx],
            uav_collision_list[idx],
            label=f"uav_{env.uavs[idx].id}",
        )
        all_axes[1].plot(
            time_step_list[idx],
            obstacle_collision_list[idx],
            label=f"uav_{env.uavs[idx].id}",
        )
        all_axes[2].plot(
            time_step_list[idx],
            uav_done_list[idx],
            label=f"uav_{env.uavs[idx].id}",
        )
        all_axes[3].plot(
            time_step_list[idx],
            rel_pad_dist[idx],
            label=f"uav_{env.uavs[idx].id}",
        )
        all_axes[4].plot(
            time_step_list[idx],
            rel_pad_vel[idx],
            label=f"uav_{env.uavs[idx].id}",
        )

    all_axes[0].set_ylabel("UAV collisions")
    all_axes[1].set_ylabel("UAV collisions")
    all_axes[2].set_ylabel("UAV landed")
    all_axes[3].set_ylabel("$\parallel \Delta \mathbf{r} \parallel$")
    all_axes[4].set_ylabel("$\parallel \Delta \mathbf{v} \parallel$")

    for ax_ in all_axes:
        ax_.set_xlabel("t (s)")

    plt.show()

    labels = [*all_axes[0].get_legend_handles_labels()]
    return labels


def main():
    target_v = [0.0, 1.0]
    use_safe_action = [False, True]
    num_obstacles = [20, 30]

    target_v = [0.0]
    use_safe_action = [False]
    num_obstacles = [1]

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
                labels = experiment(config)

    figsize = (10, 3)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    ax_leg.legend(*labels, loc="center", ncol=len(labels[1]))
    # hide the axes frame and the x/y labels
    ax_leg.axis("off")
    # fig_leg.savefig(os.path.join(image_output_folder, 'magnet_test_legend.png'))

    plt.show()


if __name__ == "__main__":
    main()
