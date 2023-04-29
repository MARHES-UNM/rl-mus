# from tests import context

# import context

from uav_sim.agents.uav import Quadrotor
import unittest
import numpy as np


class TestUav(unittest.TestCase):
    def setUp(self):
        self.uav = Quadrotor(0, 3, 4, 5)

    def test_rotation_matrix(self):
        for i in range(10):
            self.uav._state[6:9] = np.random.rand(3) * (np.pi + np.pi) - np.pi
            rot_mat = self.uav.rotation_matrix()
            np.testing.assert_almost_equal(np.linalg.det(rot_mat), 1.0)

    def test_uav_model_rot(self):
        for i in range(10):
            x = np.random.rand() * 3
            y = np.random.rand() * 3
            z = np.random.rand() * 3
            uav = Quadrotor(0, x, y, z)

            for _step in range(10):
                action = np.ones(4) * uav.m * uav.g / 4
                uav.step(action)
                expected_traj = np.array([x, y, z])
                np.testing.assert_array_almost_equal(expected_traj, uav.state[0:3])

    def test_uav_model_gravity(self):
        uav = Quadrotor(0, 5, 5, 1, dt=0.1)

        for _step in range(10):
            action = np.zeros(4)
            uav.step(action)
        np.testing.assert_almost_equal(0.0, uav.state[2])


if __name__ == "__main__":
    unittest.main()
