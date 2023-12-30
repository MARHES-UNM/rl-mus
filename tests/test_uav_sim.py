from tkinter import W
from matplotlib import pyplot as plt
import numpy as np
import unittest
from uav_sim.envs.uav_sim import UavSim
from uav_sim.utils.trajectory_generator import TrajectoryGenerator
from uav_sim.utils.trajectory_generator import (
    calculate_acceleration,
    calculate_position,
    calculate_velocity,
)


class TestUavSim(unittest.TestCase):
    def setUp(self):
        self.env = UavSim()

    def test_observation_space(self):
        env = UavSim({"num_uavs": 1, "num_obstacles": 0})
        obs_space = env.observation_space
        self.assertEqual(len(obs_space.spaces), 1)
        self.assertEqual(obs_space[0]["obstacles"].shape[0], 0)
        self.assertEqual(obs_space[0]["other_uav_obs"].shape[0], 0)

        env = UavSim({"num_uavs": 1, "num_obstacles": 1})
        obs_space = env.observation_space
        self.assertEqual(len(obs_space.spaces), 1)
        self.assertEqual(obs_space[0]["obstacles"].shape[0], 1)
        self.assertEqual(obs_space[0]["other_uav_obs"].shape[0], 0)

        env = UavSim({"num_uavs": 2, "num_obstacles": 0})
        obs_space = env.observation_space
        self.assertEqual(len(obs_space.spaces), 2)
        self.assertEqual(obs_space[0]["obstacles"].shape[0], 0)
        self.assertEqual(obs_space[0]["other_uav_obs"].shape[0], 1)

        env = UavSim({"num_uavs": 2, "num_obstacles": 1})
        obs_space = env.observation_space
        self.assertEqual(len(obs_space.spaces), 2)
        self.assertEqual(obs_space[0]["obstacles"].shape[0], 1)
        self.assertEqual(obs_space[0]["other_uav_obs"].shape[0], 1)

    def test_random_action_sample(self):
        obs = self.env.reset()

        for i in range(100):
            actions = self.env.action_space.sample()

            obs, reward, done, info = self.env.step(actions)
            self.env.render()

    def test_time_coordinated_control_mat(self):
        tf = 30.0
        tf = 20.0
        N = 1.0
        self.env = UavSim(
            {
                "target_v": 0.0,
                "num_uavs": 4,
                "use_safe_action": True,
                "num_obstacles": 30,
                "max_time": 30.0,
                "seed": 0,
            }
        )

        # des_pos = np.zeros(15)
        des_pos = self.env.uavs[0].pad.state[0:6] - self.env.uavs[0].state[0:6]
        # g_mat = self.env.uavs[0].get_g_mat(des_pos, tf, N)
        # g = self.env.uavs[0].get_g(des_pos, tf, N)
        # p = self.env.uavs[0].get_p_mat(tf, N, 0.0)

        # print(f"\ng: {g[-1]}")
        # print(f"\ng_mat: {g_mat[-1]}")
        # print(f"\np_mat{p}")

        obs, done = self.env.reset(), False
        actions = {}
        time_step_list = [[] for idx in range(self.env.num_uavs)]
        uav_collision_list = [[] for idx in range(self.env.num_uavs)]
        obstacle_collision_list = [[] for idx in range(self.env.num_uavs)]
        uav_done_list = [[] for idx in range(self.env.num_uavs)]
        rel_pad_dist = [[] for idx in range(self.env.num_uavs)]
        rel_pad_vel = [[] for idx in range(self.env.num_uavs)]
        t_go_est = [[] for idx in range(self.env.num_uavs)]

        # self.env.uavs[0].state[0:6] = np.array(
        #     [1500, 800, 350, 0, -40, 0], dtype=np.float64
        # )
        # des_pos[0:6] = np.array([200, 0, 300, 0, -40, 0], dtype=np.float64)
        # k_gain = self.env.uavs[0].get_k(tf, N)
        # g_s = []
        # for idx in range(self.env.num_uavs):
        #     des_pos = np.zeros(15)
        #     # des_pos[0:6] = np.array([200, 0, 300, 0, -40, 0], dtype=np.float64)

        #     des_pos[0:6] = -self.env.uavs[idx].pad.state[0:6]
        #     # des_pos[0:6] = -self.env.target.state[0:6]
        #     # des_pos[0:6] = self.env.uavs[0].pad.state[0:6] - self.env.uavs[0].state[0:6]
        #     # print(np.linalg.norm(des_pos[0:3]))
        #     des_pos[0:6] = np.array([0, 0, 0, 0, 0, 0])
        #     # des_pos[0:6] = self.env.target.state[0:6]

        #     temp_g = self.env.uavs[idx].get_g_mat(des_pos, tf, N)
        #     g = np.zeros(9)
        #     g[0] = temp_g[-1, 0]
        #     g[1] = temp_g[-1, 1]
        #     g[2:] = temp_g[-1, 3:]
        #     # g = self.env.uavs[idx].get_g(des_pos, tf, N)
        #     # g = g[-1, :]
        #     g_s.append(g)

        t = 0
        # for _step in range(500):
        p_idx = 0
        while True:
            actions = {}
            # g_s = []
            # for idx in range(self.env.num_uavs):
            #     des_pos = np.zeros(15)
            #     des_pos[0:6] = self.env.uavs[idx].pad.state[0:6]
            #     des_pos[0:6] = np.array([0, 0, 0, 0, 0, 0])
            #     # des_pos[0:6] = -des_pos[0:6]
            #     temp_g = self.env.uavs[idx].get_g_mat(des_pos, tf, N)
            #     g = np.zeros(9)
            #     g[0] = temp_g[-1, 0]
            #     g[1] = temp_g[-1, 1]
            #     g[2:] = temp_g[-1, 3:]
            #     # g = self.env.uavs[idx].get_g(des_pos, tf, N)
            #     # g = g[-1, :]
            #     g_s.append(g)
            # #     g = self.env.uavs[idx].get_g(des_pos, tf, N)
            # #     g_s.append(g)
            pos_er = np.zeros((self.env.num_uavs, 12))
            for idx in range(self.env.num_uavs):
                des_pos = np.zeros(15)
                des_pos[0:6] = self.env.uavs[idx].pad.state[0:6]
                # # des_pos[0:6] = self.env.target.state[0:6]
                # pos_er[idx, :] = des_pos[0:12] - self.env.uavs[idx].state

                # # actions[idx] = np.dot(k_gain, pos_er[0:6])
                # t = self.env.time_elapsed
                # r = np.linalg.norm(pos_er[idx, 0:3])
                # v = np.linalg.norm(pos_er[idx, 3:6])
                # # t_go = r / (v + 0.00000001)
                # # t_go = (tf) ** N
                # # t_go = 40
                # # t_go = max(0, t_go)
                # # R = 1 / t_go
                # # t0 = max(t, 0.6)
                # # t0 = t
                # t0 = min(t, tf - 0.1)
                # t_go = (tf - t0) ** N
                # p = self.env.uavs[idx].get_p_mat(tf, N, t0)
                # B = np.zeros((2, 1))
                # B[1, 0] = 1.0
                # actions[idx] = t_go * np.array(
                #     [
                #         B.T @ p[-1].reshape((2, 2)) @ pos_er[idx, [0, 3]],
                #         B.T @ p[-1].reshape((2, 2)) @ pos_er[idx, [1, 4]],
                #         B.T @ p[-1].reshape((2, 2)) @ pos_er[idx, [2, 5]],
                #     ]
                # )

                # # actions[idx] = t_go * np.array(
                # #     [
                # #         g2x + p2 * pos_er[0] + p3 * pos_er[3],
                # #         g2y + p2 * pos_er[1] + p3 * pos_er[4],
                # #         g2z + p2 * pos_er[2] + p3 * pos_er[5],
                # #     ]
                # # )
                # # actions[idx] = self.env.uavs[idx].get_time_coordinated_action(
                # #     des_pos, tf, t, N, g_s[idx]
                # # )

                actions[idx] = self.env.get_time_coord_action(self.env.uavs[idx])
            obs, rew, done, info = self.env.step(actions)
            for k, v in info.items():
                time_step_list[k].append(v["time_step"])
                uav_collision_list[k].append(v["uav_collision"])
                obstacle_collision_list[k].append(v["obstacle_collision"])
                uav_done_list[k].append(v["uav_landed"])
                # rel_dist = np.linalg.norm(obs[k]["rel_pad"][0:3])
                rel_dist = np.linalg.norm(v["uav_rel_dist"])
                rel_vel = np.linalg.norm(v["uav_rel_vel"])
                rel_pad_dist[k].append(rel_dist)
                rel_pad_vel[k].append(rel_vel)
                t_go_est[k].append(v["uav_t_go_est"])
            # self.env.render()
            t += self.env.dt
            p_idx += 1
            p_idx = int(min(p_idx, tf / self.env.dt - 2))

            if done["__all__"]:
                break
        time_step_list = np.array(time_step_list)
        uav_collision_list = np.array(uav_collision_list)
        obstacle_collision_list = np.array(obstacle_collision_list)
        uav_done_list = np.array(uav_done_list)
        rel_pad_dist = np.array(rel_pad_dist)
        rel_pad_vel = np.array(rel_pad_vel)
        t_go_est = np.array(t_go_est)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)

        fig = plt.figure(figsize=(10, 6))
        ax2 = fig.add_subplot(111)

        fig = plt.figure(figsize=(10, 6))
        ax3 = fig.add_subplot(121)
        ax4 = fig.add_subplot(122)
        fig = plt.figure(figsize=(10, 6))
        ax5 = fig.add_subplot(111)
        all_axes = [ax, ax1, ax2, ax3, ax4, ax5]
        for idx in range(self.env.num_uavs):
            ax.plot(
                time_step_list[idx],
                uav_collision_list[idx],
                label=f"uav_{self.env.uavs[idx].id}",
            )
            ax1.plot(
                time_step_list[idx],
                obstacle_collision_list[idx],
                label=f"uav_{self.env.uavs[idx].id}",
            )
            ax2.plot(
                time_step_list[idx],
                uav_done_list[idx],
                label=f"uav_{self.env.uavs[idx].id}",
            )
            ax3.plot(
                time_step_list[idx],
                rel_pad_dist[idx],
                label=f"uav_{self.env.uavs[idx].id}",
            )
            ax4.plot(
                time_step_list[idx],
                rel_pad_vel[idx],
                label=f"uav_{self.env.uavs[idx].id}",
            )
            ax5.plot(
                time_step_list[idx],
                t_go_est[idx],
                label=f"uav_{self.env.uavs[idx].id}",
            )

        ax.set_ylabel("# UAV collisions")
        ax1.set_ylabel("# UAV collisions")
        ax2.set_ylabel("UAV landed")
        ax3.set_ylabel(r"$\parallel \Delta \mathbf{r} \parallel$")
        ax4.set_ylabel(r"$\parallel \Delta \mathbf{v} \parallel$")
        for ax_ in all_axes:
            ax_.set_xlabel("t (s)")
            # ax_.legend_.remove()

        figsize = (10, 3)
        fig_leg = plt.figure(figsize=figsize)
        ax_leg = fig_leg.add_subplot(111)
        # add the legend from the previous axes
        ax_leg.legend(
            *ax4.get_legend_handles_labels(), loc="center", ncol=self.env.num_uavs
        )
        # hide the axes frame and the x/y labels
        ax_leg.axis("off")
        # fig_leg.savefig(os.path.join(image_output_folder, 'magnet_test_legend.png'))

        plt.show()
        print()

    def test_time_coordinated_control(self):
        tf = 30.0
        tf = 15.0
        N = 0
        self.env = UavSim(
            {
                "target_v": 2.0,
                "num_uavs": 4,
                "use_safe_action": True,
                "num_obstacles": 5,
                # "seed": 0,
            }
        )

        obs, done = self.env.reset(), False
        actions = {}
        uav_collision_list = [[] for idx in range(self.env.num_uavs)]
        obstacle_collision_list = [[] for idx in range(self.env.num_uavs)]
        uav_done_list = [[] for idx in range(self.env.num_uavs)]
        rel_pad_dist = [[] for idx in range(self.env.num_uavs)]
        rel_pad_vel = [[] for idx in range(self.env.num_uavs)]

        # self.env.uavs[0].state[0:6] = np.array(
        #     [1500, 800, 350, 0, -40, 0], dtype=np.float64
        # )
        # des_pos[0:6] = np.array([200, 0, 300, 0, -40, 0], dtype=np.float64)
        k_gain = self.env.uavs[0].get_k(tf, N)
        g_s = []
        for idx in range(self.env.num_uavs):
            des_pos = np.zeros(15)
            # des_pos[0:6] = np.array([200, 0, 300, 0, -40, 0], dtype=np.float64)

            des_pos[0:6] = self.env.uavs[idx].pad.state[0:6]
            print(np.linalg.norm(des_pos[0:3]))
            des_pos[0:6] = np.array([0, 0, 0, 0, 0, 0])
            # des_pos[0:6] = self.env.target.state[0:6]

            # des_pos[0:6] = self.env.uavs[0].pad.state[0:6] - self.env.uavs[0].state[0:6]
            g = self.env.uavs[idx].get_g(des_pos, tf, N)
            g = g[-1, :]

            g_s.append(g)

        t = 0
        for _step in range(500):
            actions = {}
            g_s = []
            for idx in range(self.env.num_uavs):
                des_pos = np.zeros(15)
                des_pos[0:6] = self.env.uavs[idx].pad.state[0:6]
                des_pos[0:6] = np.array([0, 0, 0, 0, 0, 0])
                # des_pos[0:6] = -des_pos[0:6]

                g = self.env.uavs[idx].get_g(des_pos, tf, N, t)
                g = g[-1, :]
                g_s.append(g)
            for idx in range(self.env.num_uavs):
                # des_pos = np.zeros(15)
                des_pos[0:6] = self.env.uavs[idx].pad.state[0:6]
                pos_er = des_pos[0:12] - self.env.uavs[idx].state

                # actions[idx] = np.dot(k_gain, pos_er[0:6])
                # # t = self.env.time_elapsed
                # t_go = (tf - t) ** N

                # actions[idx] = t_go * np.array(
                #     [
                #         g2x + p2 * pos_er[0] + p3 * pos_er[3],
                #         g2y + p2 * pos_er[1] + p3 * pos_er[4],
                #         g2z + p2 * pos_er[2] + p3 * pos_er[5],
                #     ]
                # )
                actions[idx] = self.env.uavs[idx].get_time_coordinated_action(
                    des_pos, tf, t, N, g_s[idx]
                )

            obs, rew, done, info = self.env.step(actions)
            for k, v in info.items():
                uav_collision_list[k].append(v["uav_collision"])
                obstacle_collision_list[k].append(v["obstacle_collision"])
                uav_done_list[k].append(v["uav_landed"])
                rel_dist = np.linalg.norm(obs[k]["rel_pad"][0:3])
                rel_vel = np.linalg.norm(obs[k]["rel_pad"][3:6])
                # rel_dist = np.linalg.norm(pos_er[0:3])
                # rel_vel = np.linalg.norm(pos_er[k, 3:6])
                rel_pad_dist[k].append(rel_dist)
                rel_pad_vel[k].append(rel_vel)
            # self.env.render()
            t += self.env.dt

            if done["__all__"]:
                break
        uav_collision_list = np.array(uav_collision_list)
        obstacle_collision_list = np.array(obstacle_collision_list)
        uav_done_list = np.array(uav_done_list)
        rel_pad_dist = np.array(rel_pad_dist)
        rel_pad_vel = np.array(rel_pad_vel)
        self.plot_uav_states(
            uav_collision_list,
            obstacle_collision_list,
            uav_done_list,
            rel_pad_dist,
            rel_pad_vel,
        )

    def plot_uav_states(
        self,
        uav_collision_list,
        obstacle_collision_list,
        uav_done_list,
        rel_pad_dist,
        rel_pad_vel,
    ):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(311)
        ax1 = fig.add_subplot(312)
        ax2 = fig.add_subplot(313)
        fig = plt.figure(figsize=(10, 6))
        ax3 = fig.add_subplot(211)
        ax4 = fig.add_subplot(212)
        for idx in range(self.env.num_uavs):
            ax.plot(uav_collision_list[idx], label=f"id:{self.env.uavs[idx].id}")
            ax1.plot(obstacle_collision_list[idx], label=f"id:{self.env.uavs[idx].id}")
            ax2.plot(uav_done_list[idx], label=f"id:{self.env.uavs[idx].id}")
            ax3.plot(rel_pad_dist[idx], label=f"id:{self.env.uavs[idx].id}")
            ax4.plot(rel_pad_vel[idx], label=f"id:{self.env.uavs[idx].id}")
        plt.legend()
        plt.show()
        print()

    def test_setting_pred_targets(self):
        self.env = UavSim(
            {"target_v": 1, "use_safe_action": False, "num_obstacles": 4, "seed": 0}
        )

        obs, done = self.env.reset(), False
        actions = {}
        uav_collision_list = [[] for idx in range(self.env.num_uavs)]
        obstacle_collision_list = [[] for idx in range(self.env.num_uavs)]
        uav_done_list = [[] for idx in range(self.env.num_uavs)]
        rel_pad_dist = [[] for idx in range(self.env.num_uavs)]
        rel_pad_v = [[] for idx in range(self.env.num_uavs)]
        t_go_est_list = [[] for idx in range(self.env.num_uavs)]

        for _step in range(120):
            for idx in range(self.env.num_uavs):
                des_pos = np.zeros(15)
                des_pos[0:6] = self.env.uavs[idx].pad.state[0:6]
                actions[idx] = self.env.uavs[idx].calc_des_action(des_pos)

            obs, rew, done, info = self.env.step(actions)
            for k, v in info.items():
                uav_collision_list[k].append(v["uav_collision"])
                obstacle_collision_list[k].append(v["obstacle_collision"])
                uav_done_list[k].append(v["uav_landed"])
                rel_dist = np.linalg.norm(obs[k]["rel_pad"][0:3])
                rel_pad_dist[k].append(rel_dist)
                rel_v = np.linalg.norm(obs[k]["rel_pad"][3:])
                rel_pad_v[k].append(rel_v)
                t_go_est = rel_dist / (1e-6 + rel_v)
                t_go_est_list[k].append(t_go_est)

            # self.env.render()

            if done["__all__"]:
                break
        uav_collision_list = np.array(uav_collision_list)
        obstacle_collision_list = np.array(obstacle_collision_list)
        uav_done_list = np.array(uav_done_list)
        rel_pad_dist = np.array(rel_pad_dist)
        t_go_est_list = np.array(t_go_est_list)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(311)
        ax1 = fig.add_subplot(312)
        ax2 = fig.add_subplot(313)
        fig = plt.figure(figsize=(10, 6))
        ax3 = fig.add_subplot(111)

        fig = plt.figure(figsize=(10, 6))
        ax4 = fig.add_subplot(111)

        fig = plt.figure(figsize=(10, 6))
        ax5 = fig.add_subplot(111)
        for idx in range(self.env.num_uavs):
            ax.plot(uav_collision_list[idx], label=f"id:{self.env.uavs[idx].id}")
            ax1.plot(obstacle_collision_list[idx], label=f"id:{self.env.uavs[idx].id}")
            ax2.plot(uav_done_list[idx], label=f"id:{self.env.uavs[idx].id}")
            ax3.plot(rel_pad_dist[idx], label=f"id:{self.env.uavs[idx].id}")
            ax4.plot(rel_pad_v[idx], label=f"id:{self.env.uavs[idx].id}")
            ax5.plot(t_go_est_list[idx], label=f"id:{self.env.uavs[idx].id}")

        plt.legend()
        plt.show()
        print()

    def test_lqr_landing_cbf(self):
        self.env = UavSim(
            {"target_v": 0, "use_safe_action": True, "num_obstacles": 4, "seed": 0}
        )
        obs, done = self.env.reset(), False

        des_pos = np.zeros((self.env.num_uavs, 15))
        pads = self.env.target.pads
        # get pad positions
        for idx in range(self.env.num_uavs):
            des_pos[idx, 0:2] = np.array([pads[idx].x, pads[idx].y])
            # set uav starting positions
            self.env.uavs[idx].state[0:3] = np.array([pads[idx].x, pads[idx].y, 3])
            # set obstacle positions
            self.env.obstacles[idx].state[0:3] = np.array(
                [pads[idx].x, pads[idx].y + 0.1, 2]
            )

        actions = {}

        uav_collision_list = [[] for idx in range(self.env.num_uavs)]
        obstacle_collision_list = [[] for idx in range(self.env.num_uavs)]
        uav_done_list = [[] for idx in range(self.env.num_uavs)]
        for _step in range(100):
            for idx in range(self.env.num_uavs):
                actions[idx] = self.env.uavs[idx].calc_torque(des_pos[idx])

            obs, rew, done, info = self.env.step(actions)
            for k, v in info.items():
                uav_collision_list[k].append(v["uav_collision"])
                obstacle_collision_list[k].append(v["obstacle_collision"])
                uav_done_list[k].append(v["uav_landed"])
            self.env.render()

            if done["__all__"]:
                break
        uav_collision_list = np.array(uav_collision_list)
        obstacle_collision_list = np.array(obstacle_collision_list)
        uav_done_list = np.array(uav_done_list)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(311)
        ax1 = fig.add_subplot(312)
        ax2 = fig.add_subplot(313)
        for idx in range(self.env.num_uavs):
            ax.plot(uav_collision_list[idx], label=f"id:{self.env.uavs[idx].id}")
            ax1.plot(obstacle_collision_list[idx], label=f"id:{self.env.uavs[idx].id}")
            ax2.plot(uav_done_list[idx], label=f"id:{self.env.uavs[idx].id}")

        plt.legend()
        plt.show()
        print()

    def test_cbf_multi(self):
        self.env = UavSim({"target_v": 0, "num_obstacles": 4, "use_safe_action": True})
        # self.env.gamma = 4
        obs, done = self.env.reset(), False

        actions = {}

        for _step in range(200):
            pads = self.env.target.pads
            positions = np.zeros((self.env.num_uavs, 15))

            for idx, pos in enumerate(positions):
                positions[idx][0:2] = np.array([pads[idx].x, pads[idx].y])
                actions[idx] = self.env.uavs[idx].calc_torque(pos)

            obs, rew, done, info = self.env.step(actions)
            self.env.render()

            if done["__all__"]:
                break

    def test_landing_minimum_traj(self):
        self.env = UavSim({"target_v": 0, "use_safe_action": False})

        obs, done = self.env.reset(), False

        t_final = 9
        start_pos = np.array([uav.state[0:3] for uav in self.env.uavs])
        pads = self.env.target.pads
        positions = [[pad.x, pad.y, 0, 0] for pad in pads]

        uav_coeffs = np.zeros((self.env.num_uavs, 3, 6, 1), dtype=np.float64)
        for idx in range(self.env.num_uavs):
            traj = TrajectoryGenerator(start_pos[idx], positions[idx], t_final)
            traj.solve()
            uav_coeffs[idx, 0] = traj.x_c
            uav_coeffs[idx, 1] = traj.y_c
            uav_coeffs[idx, 2] = traj.z_c

        actions = {}

        t = 0
        uav_collision_list = [[] for idx in range(self.env.num_uavs)]
        obstacle_collision_list = [[] for idx in range(self.env.num_uavs)]
        uav_done_list = [[] for idx in range(self.env.num_uavs)]
        rel_pad_dist = [[] for idx in range(self.env.num_uavs)]

        for _step in range(150):
            des_pos = np.zeros((self.env.num_uavs, 15), dtype=np.float64)
            for idx in range(self.env.num_uavs):
                # acceleration
                des_pos[idx, 12] = calculate_acceleration(uav_coeffs[idx, 0], t)
                des_pos[idx, 13] = calculate_acceleration(uav_coeffs[idx, 1], t)
                des_pos[idx, 14] = calculate_acceleration(uav_coeffs[idx, 2], t)

                actions[idx] = des_pos[idx, 12:15]
            # obs, rew, done, info = self.env.step(actions)
            obs, rew, done, info = self.env.step(actions)
            for k, v in info.items():
                uav_collision_list[k].append(v["uav_collision"])
                obstacle_collision_list[k].append(v["obstacle_collision"])
                uav_done_list[k].append(v["uav_landed"])
                rel_dist = np.linalg.norm(obs[k]["rel_pad"][0:3])
                rel_pad_dist[k].append(rel_dist)
            self.env.render()
            t += self.env.dt

            if done["__all__"]:
                break
        uav_collision_list = np.array(uav_collision_list)
        obstacle_collision_list = np.array(obstacle_collision_list)
        uav_done_list = np.array(uav_done_list)
        rel_pad_dist = np.array(rel_pad_dist)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(311)
        ax1 = fig.add_subplot(312)
        ax2 = fig.add_subplot(313)
        fig = plt.figure(figsize=(10, 6))
        ax3 = fig.add_subplot(111)
        for idx in range(self.env.num_uavs):
            ax.plot(uav_collision_list[idx], label=f"id:{self.env.uavs[idx].id}")
            ax1.plot(obstacle_collision_list[idx], label=f"id:{self.env.uavs[idx].id}")
            ax2.plot(uav_done_list[idx], label=f"id:{self.env.uavs[idx].id}")
            ax3.plot(rel_pad_dist[idx], label=f"id:{self.env.uavs[idx].id}")

        plt.legend()
        plt.show()
        print()

    def test_barrier_function_single(self):
        env = UavSim({"num_uavs": 1, "num_obstacles": 1, "use_safe_action": True})
        env.gamma = 6

        obs, done = env.reset(), False

        # uav position
        env.uavs[0]._state[0:3] = np.array([3, 3, 1])

        # target
        env.target.x = 3
        env.target.y = 2
        env.target.step([0, 0])

        # obstacle position
        env.obstacles[0]._state[0:3] = np.array([3.1, 1.5, 1])

        des_pos = np.zeros(15)
        des_pos[0:3] = np.array([3, 0, 1])

        actions = {}

        uav_collisions = 0
        obstacle_collision = 0
        for _step in range(100):
            for idx in range(env.num_uavs):
                actions[idx] = env.uavs[idx].calc_des_action(des_pos)
            obs, rew, done, info = env.step(actions)
            uav_collisions += sum([v["uav_collision"] for v in info.values()])
            obstacle_collision += sum(
                [v["obstacle_collision"] for k, v in info.items()]
            )
            env.render()

            if done["__all__"]:
                break

        print(f"uav_collision: {uav_collisions}")
        print(f"obstacle_collision: {obstacle_collision}")

    # @unittest.skip
    def test_lqr_landing(self):
        self.env = UavSim({"target_v": 0, "use_safe_action": False})
        obs, done = self.env.reset(), False

        actions = {}

        uav_collision_list = [[] for idx in range(self.env.num_uavs)]
        obstacle_collision_list = [[] for idx in range(self.env.num_uavs)]
        uav_done_list = [[] for idx in range(self.env.num_uavs)]
        for _step in range(100):
            pads = self.env.target.pads
            positions = np.zeros((self.env.num_uavs, 15))

            for idx, pos in enumerate(positions):
                positions[idx][0:2] = np.array([pads[idx].x, pads[idx].y])
                actions[idx] = self.env.uavs[idx].calc_torque(pos)

            obs, rew, done, info = self.env.step(actions)
            for k, v in info.items():
                uav_collision_list[k].append(v["uav_collision"])
                obstacle_collision_list[k].append(v["obstacle_collision"])
                uav_done_list[k].append(v["uav_landed"])
            self.env.render()

            if done["__all__"]:
                break
        uav_collision_list = np.array(uav_collision_list)
        obstacle_collision_list = np.array(obstacle_collision_list)
        uav_done_list = np.array(uav_done_list)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(311)
        ax1 = fig.add_subplot(312)
        ax2 = fig.add_subplot(313)
        for idx in range(self.env.num_uavs):
            ax.plot(uav_collision_list[idx], label=f"id:{self.env.uavs[idx].id}")
            ax1.plot(obstacle_collision_list[idx], label=f"id:{self.env.uavs[idx].id}")
            ax2.plot(uav_done_list[idx], label=f"id:{self.env.uavs[idx].id}")

        plt.legend()
        plt.show()
        print()

    @unittest.skip
    def test_lqr_waypoints(self):
        T = 20
        t = 0
        start_pos = np.zeros((4, 3))
        for i in range(4):
            start_pos[i, :] = self.env.uavs[i].state[0:3]

        waypoints = [[0.5, 0.5, 2], [0.5, 2, 1.5], [2, 0.5, 2.5], [2, 2, 1], [0, 0, 0]]
        num_waypoints = len(waypoints)

        uav_coeffs = np.zeros((self.env.num_uavs, num_waypoints, 3, 6, 1))

        for i in range(self.env.num_uavs):
            for way_point_num in range(num_waypoints):
                traj = TrajectoryGenerator(
                    waypoints[way_point_num],
                    waypoints[(way_point_num + 1) % num_waypoints],
                    T,
                )
                traj.solve()
                uav_coeffs[i, way_point_num, 0] = traj.x_c
                uav_coeffs[i, way_point_num, 1] = traj.y_c
                uav_coeffs[i, way_point_num, 2] = traj.z_c

        way_point_num = 0
        while True:
            while t <= T:
                des_pos = np.zeros((4, 12), dtype=np.float64)
                actions = {}
                for idx in range(self.env.num_uavs):
                    uav_waypoint_num = (way_point_num + idx) % num_waypoints
                    des_pos[idx, 0] = calculate_position(
                        uav_coeffs[idx, uav_waypoint_num, 0], t
                    )
                    des_pos[idx, 1] = calculate_position(
                        uav_coeffs[idx, uav_waypoint_num, 1], t
                    )
                    des_pos[idx, 2] = calculate_position(
                        uav_coeffs[idx, uav_waypoint_num, 2], t
                    )
                    des_pos[idx, 3] = 1.0
                    actions[idx] = self.env.uavs[idx].calc_torque(des_pos[idx])

                self.env.step(actions)
                self.env.render()

                t += self.env.dt
            t = 0
            way_point_num = (way_point_num + 1) % num_waypoints

    @unittest.skip
    def test_lqr_controller(self):
        positions = np.array(
            [[0.5, 0.5, 1, np.pi], [0.5, 2, 2, 0], [2, 0.5, 2, 1.2], [2, 2, 1, -1.2]]
        )

        actions = {}
        for i in range(100):
            for idx, pos in enumerate(positions):
                actions[idx] = self.env.uavs[idx].calc_torque(pos)
            self.env.step(actions)
            self.env.render()

    def test_setting_uav_pos(self):
        self.env = UavSim(env_config={"dt": 0.1})
        uav_pos = np.array([[0.5, 0.5, 1], [0.5, 2, 2], [2, 0.5, 2], [2, 2, 1]])
        for idx, pos in enumerate(uav_pos):
            self.env.uavs[idx]._state[0] = pos[0]
            self.env.uavs[idx]._state[1] = pos[1]
            self.env.uavs[idx]._state[2] = pos[2]

        for i in range(20):
            u_in = np.zeros(3)
            actions = {_id: np.zeros(3) for _id in range(self.env.num_uavs)}
            self.env.step(actions)

        for idx in range(self.env.num_uavs):
            np.testing.assert_array_almost_equal(
                uav_pos[idx, 0:3], self.env.uavs[idx].state[0:3]
            )

    def test_constraints(self):
        config = {"max_num_obstacles": 2, "num_obstacles": 1, "num_uavs": 2}
        env = UavSim(env_config=config)

        env.uavs[0]._state[:3] = np.array([1, 1, 4])
        env.uavs[1]._state[:3] = np.array([4, 4, 4])

        env.obstacles[0]._state[:3] = np.array([2, 2, 4])

        actions = {i: np.zeros(3) for i in range(env.num_uavs)}

        obs, _, done, info = env.step(actions)

        all_constraints = np.array([ob["constraint"] for ob in obs.values()])
        self.assertTrue((all_constraints > 0).all())
        uav_collisions = np.array(
            [col_info["uav_collision"] for col_info in info.values()]
        )
        obstacle_collisions = np.array(
            [col_info["obstacle_collision"] for col_info in info.values()]
        )

        self.assertTrue((uav_collisions == 0).all())
        self.assertTrue((obstacle_collisions == 0).all())
        env.uavs[0]._state[:3] = np.array([1.5, 1.5, 4])
        obs, _, done, info = env.step(actions)
        all_constraints = np.array([ob["constraint"] for ob in obs.values()])
        self.assertFalse((all_constraints > 0).all())
        uav_collisions = np.array(
            [col_info["uav_collision"] for col_info in info.values()]
        )
        obstacle_collisions = np.array(
            [col_info["obstacle_collision"] for col_info in info.values()]
        )

        self.assertTrue((uav_collisions == 0).all())
        self.assertTrue((obstacle_collisions[0] > 0))

        env.uavs[0]._state[:3] = np.array([3.9, 3.9, 4])
        obs, _, done, info = env.step(actions)
        all_constraints = np.array([ob["constraint"] for ob in obs.values()])
        self.assertFalse((all_constraints > 0).all())
        uav_collisions = np.array(
            [col_info["uav_collision"] for col_info in info.values()]
        )
        obstacle_collisions = np.array(
            [col_info["obstacle_collision"] for col_info in info.values()]
        )

        self.assertTrue((uav_collisions == 1).all())
        self.assertTrue((obstacle_collisions == 0).all())


if __name__ == "__main__":
    unittest.main()
