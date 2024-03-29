from matplotlib import pyplot as plt
import torch

from uav_sim.agents.uav import Obstacle, Pad, Target, UavBase
import unittest
import numpy as np

from uav_sim.utils.trajectory_generator import (
    TrajectoryGenerator,
    calculate_acceleration,
    calculate_position,
)


class TestUavBase(unittest.TestCase):
    def setUp(self):
        self.uav = UavBase(0, 2, 2, 2)

    def test_uav_hover(self):
        """Test that the uav can hover with the specified input"""
        for i in range(10):
            x = np.random.rand() * 3
            y = np.random.rand() * 3
            z = np.random.rand() * 3
            uav = UavBase(0, x, y, z)

            for _ in range(10):
                # no need to provide input step takes calculating gravity term
                # action = np.array([0.0, 0.0, uav.m * uav.g])
                action = np.array([0.0, 0.0, 0.0])
                uav.step(action)
                expected_traj = np.array([x, y, z])
                np.testing.assert_array_almost_equal(expected_traj, uav.state[0:3])

    def test_uav_model_gravity(self):
        """Test that the UAV fall to the ground when 0 force is applied"""
        uav = UavBase(0, 5, 5, 1, dt=0.1)

        for _step in range(11):
            action = np.zeros(3)
            action[2] = -uav.m * uav.g
            uav.step(action)
        np.testing.assert_almost_equal(uav.state[2], 0.0)

    def test_get_landed(self):
        pad = Pad(0, 1, 1)

        (is_reached, dist, vel) = self.uav.check_dest_reached()
        self.assertFalse(is_reached)

        # des_pos = np.zeros(15)
        # for i in range(100):
        #     des_pos[0:3] = pad.state[0:3]
        #     action = self.uav.calc_des_action(des_pos)

        #     self.uav.step(action)

        # (is_reached, dist, vel) = self.uav.check_dest_reached()
        # self.assertTrue(is_reached)

    def test_uav_collision(self):
        obs = Obstacle(0, 1, 1, 1)
        uav = UavBase(0, 1, 0, 0.9)

        self.assertFalse(uav.in_collision(obs))

        uav.state[0:3] = np.array([1, 1, 0.9])
        self.assertTrue(uav.in_collision(obs))


if __name__ == "__main__":
    unittest.main()
