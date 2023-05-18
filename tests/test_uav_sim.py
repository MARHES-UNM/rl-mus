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

    def test_setting_pred_targets(self):
        self.env = UavSim(
            {"target_v": 1, "use_safe_action": True, "num_obstacles": 4, "seed": 0}
        )

        obs, done = self.env.reset(), False
        actions = {}
        for _step in range(100):
            for idx in range(self.env.num_uavs):
                des_pos = np.zeros(15)
                des_pos[0:6] = self.env.uavs[idx].pad.state[0:6]
                actions[idx] = self.env.uavs[idx].calc_torque(des_pos)

            obs, rew, done, info = self.env.step(actions)
            self.env.render()

            if done["__all__"]:
                break

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
        self.env = UavSim({"target_v": 0})

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

        for _step in range(150):
            des_pos = np.zeros((self.env.num_uavs, 15), dtype=np.float64)
            for idx in range(self.env.num_uavs):
                # acceleration
                des_pos[idx, 12] = calculate_acceleration(uav_coeffs[idx, 0], t)
                des_pos[idx, 13] = calculate_acceleration(uav_coeffs[idx, 1], t)
                des_pos[idx, 14] = calculate_acceleration(uav_coeffs[idx, 2], t)

                actions[idx] = des_pos[idx, 12:15]
            obs, rew, done, info = self.env.step(actions)
            for k, v in info.items():
                uav_collision_list[k].append(v["uav_collision"])
                obstacle_collision_list[k].append(v["obstacle_collision"])
                uav_done_list[k].append(v["uav_landed"])
            self.env.render()
            t += self.env.dt

            if done["__all__"]:
                break

        uav_collision_list = np.array(uav_collision_list)
        obstacle_collision_list = np.array(obstacle_collision_list)
        uav_done_list = np.array(uav_done_list)

        fig = plt.figure(figsize=(10, 6))

        ax = fig.add_subplot(311)
        for idx in range(self.env.num_uavs):
            ax.plot(uav_done_list[idx], label=f"id:{self.env.uavs[idx].id}")

        ax = fig.add_subplot(311)
        for idx in range(self.env.num_uavs):
            ax.plot(uav_done_list[idx], label=f"id:{self.env.uavs[idx].id}")

        # ax.plot(t_final / self.env.dt, 1, label="finale time")
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
                actions[idx] = env.uavs[idx].calc_torque(des_pos)
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


if __name__ == "__main__":
    unittest.main()
