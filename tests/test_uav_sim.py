import imp
import numpy as np
import unittest

# from tests import context
# import context
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

    # @unittest.skip
    def test_lqr_waypoints(self):
        T = 1
        t = 0
        start_pos = np.zeros((4, 3))
        for i in range(4):
            start_pos[i, :] = self.env.uavs[i].state[0:3]

        waypoints = [[0.5, 0.5, 2], [0.5, 2, 1.5], [2, 0.5, 2.5], [2, 2, 1]]
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

        Ks = self.env.uavs[0].calc_gain()
        way_point_num = 0
        m = self.env.uavs[0].m
        g = self.env.uavs[0].g
        while True:
            while t <= T:
                des_pos = np.zeros((4, 12), dtype=np.float64)
                actions = {}
                for idx in range(self.env.num_uavs):
                    des_pos[idx, 0] = calculate_position(
                        uav_coeffs[idx, way_point_num, 0], t
                    )
                    des_pos[idx, 1] = calculate_position(
                        uav_coeffs[idx, way_point_num, 1], t
                    )
                    des_pos[idx, 2] = calculate_position(
                        uav_coeffs[idx, way_point_num, 2], t
                    )
                    des_pos[idx, 8] = 0

                    # # Velocity
                    # des_pos[idx, 3] = calculate_velocity(
                    #     uav_coeffs[idx, way_point_num, 0], t
                    # )
                    # des_pos[idx, 4] = calculate_velocity(
                    #     uav_coeffs[idx, way_point_num, 1], t
                    # )
                    # des_pos[idx, 5] = calculate_velocity(
                    #     uav_coeffs[idx, way_point_num, 2], t
                    # )

                    # # Acceleration
                    # acc = np.zeros(3)
                    # acc[0] = calculate_acceleration(
                    #     uav_coeffs[idx, way_point_num, 0], t
                    # )
                    # acc[1] = calculate_acceleration(
                    #     uav_coeffs[idx, way_point_num, 1], t
                    # )
                    # acc[2] = calculate_acceleration(
                    #     uav_coeffs[idx, way_point_num, 2], t
                    # )

                    # Acceleration
                    # Ks[1][0:2] *= -1
                    pos_er = des_pos[idx] - self.env.uavs[idx].state
                    # pos_er = - des_pos[idx] + self.env.uavs[idx].state
                    # pos_er[8] = self.env.uavs[idx].state[8]
                    # pos_er[11] = self.env.uavs[idx].state[11]
                    Ux = np.dot(Ks[0], pos_er[[0, 3, 7, 10]])[0]
                    Uy = np.dot(Ks[1], pos_er[[1, 4, 6, 9]])[0]
                    Uz = np.dot(Ks[2], pos_er[[2, 5]])[0]
                    Uyaw = np.dot(Ks[3], pos_er[[8, 11]])[0]
                    # Uy = 1
                    # Ux = 0
                    # Ux = 1
                    Uy = max(min(.0005, Uy), -.0005)
                    Ux = max(min(.0005, Ux), -.0005)
                    # Uy = self.env.uavs[0].ixx * Uy * .0001
                    # Ux = self.env.uavs[0].iyy * Ux * .0001
                    actions[idx] = np.array([Uz + m * g, Uy, Ux, Uyaw])

                self.env.step(actions)
                self.env.render()

                t += self.env.dt
            t = 0
            way_point_num = (way_point_num + 1) % num_waypoints

    @unittest.skip
    def test_trajectory_generator(self):
        T = 3

        start_pos = np.zeros((4, 3))
        des_pos = np.zeros((self.env.num_uavs, 3))
        for i in range(4):
            start_pos[i, :] = self.env.uavs[i].state[0:3]
            des_pos[i, 0] = 1
            des_pos[i, 1] = 1
            des_pos[i, 2] = 1

        uav_coeffs = np.zeros((self.env.num_uavs, 3, 6, 1))

        for i in range(self.env.num_uavs):
            traj = TrajectoryGenerator(start_pos[i], des_pos[i], T)
            # traj = TrajectoryGenerator(waypoints[i], waypoints[(i + 1) % 4], T)
            traj.solve()
            uav_coeffs[i, 0] = traj.x_c
            uav_coeffs[i, 1] = traj.y_c
            uav_coeffs[i, 2] = traj.z_c

        actions = {}
        for i in range(500):
            for idx in range(4):
                actions[idx] = self.env.uavs[idx].get_controller_with_coeffs(
                    uav_coeffs[idx], self.env.time_elapsed
                )
            self.env.step(actions)
            self.env.render()

    @unittest.skip
    def test_controller(self):
        des_pos = np.zeros((4, 12), dtype=np.float64)
        des_pos[:, 2] = 3
        des_pos[:, 1] = 0
        des_pos[:, 0] = 2
        # des_pos[:, 8] = np.pi
        # des_pos[:, 1] = 1
        # des_pos[:, 0:3] = np.array([[0.5, 0.5, 1], [0.5, 2, 1], [2, 0.5, 1], [2, 2, 1]])

        actions = {}
        for i in range(200):
            for idx in range(4):
                actions[idx] = self.env.uavs[idx].get_controller(des_pos[idx])
            self.env.step(actions)
            self.env.render()

    # @unittest.skip
    def test_lqr_controller(self):
        positions = np.array([[0.5, 0.5, 1], [0.5, 2, 2], [2, 0.5, 2], [2, 2, 1]])
        des_pos = np.zeros((4, 12), dtype=np.float64)
        for idx in range(4):
            # des_pos[idx, 0:3] = positions[idx, :]
            des_pos[idx, 2] = 1
            # des_pos[idx, 8] = np.pi

            # self.env.uavs[idx]._state[0:2] = positions[idx, 0:2]

        Ks = self.env.uavs[0].calc_k()
        m = self.env.uavs[0].m
        g = self.env.uavs[0].g
        inv_inertia = self.env.uavs[0].inv_inertia

        actions = {}
        for i in range(300):
            for idx, pos in enumerate(des_pos):
                # pos_er = self.env.uavs[idx].state - self.env.uavs[idx].state
                pos_er = np.zeros(12)
                pos_er = pos_er - self.env.uavs[idx].state
                pos_er[2] = 2 - self.env.uavs[idx].state[2]
                pos_er[5] = 0 - self.env.uavs[idx].state[5]
                # pos_er[8] = np.pi/2 - self.env.uavs[idx].state[8]

                Ux = np.dot(Ks[0], pos_er[[0, 3, 7, 10]])[0]
                Uy = np.dot(Ks[1], pos_er[[1, 4, 6, 9]])[0]
                Uz = np.dot(Ks[2], pos_er[[2, 5]])[0]
                Uyaw = np.dot(Ks[3], pos_er[[8, 11]])[0]
                inputs = np.array([Uz + m * g, Uy, Ux, Uyaw])

                actions[idx] = inputs
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
            actions = {_id: np.zeros(4) for _id in range(self.env.num_uavs)}
            self.env.step(actions)

        for idx in range(self.env.num_uavs):
            np.testing.assert_array_almost_equal(
                uav_pos[idx, 0:2], self.env.uavs[idx].state[0:2]
            )
            np.testing.assert_almost_equal(0.0, self.env.uavs[idx].state[2])

    # def test_render(self):
    #     tf = 100
    #     t = 0
    #     actions = {}
    #     while t < tf:
    #         for i in range(self.env.num_uavs):
    #             actions[i] = np.ones(4) * self.env.uavs[i].m * self.env.uavs[i].g / 4
    #         self.env.step(actions)
    #         self.env.render()
    #         t += 1


if __name__ == "__main__":
    unittest.main()
