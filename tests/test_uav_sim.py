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

    # @unittest.skip
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
                    # des_pos[idx, 0:3] = 0.0
                    des_pos[idx, 3] = 1.0

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
            actions = {_id: np.zeros(4) for _id in range(self.env.num_uavs)}
            self.env.step(actions)

        for idx in range(self.env.num_uavs):
            np.testing.assert_array_almost_equal(
                uav_pos[idx, 0:2], self.env.uavs[idx].state[0:2]
            )
            np.testing.assert_almost_equal(0.0, self.env.uavs[idx].state[2])


if __name__ == "__main__":
    unittest.main()