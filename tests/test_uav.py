# from tests import context

# import context

from uav_sim.agents.uav import Quadrotor
import unittest
import numpy as np


class TestUav(unittest.TestCase):
    def setUp(self):
        self.uav = Quadrotor(3, 4, 5)

    def test_rotation_matrix(self):
        for i in range(10):
            self.uav._state[6:9] = np.random.rand(3) * (np.pi + np.pi) - np.pi
            rot_mat = self.uav.rotation_matrix()
            np.testing.assert_almost_equal(np.linalg.det(rot_mat), 1.0)


if __name__ == "__main__":
    unittest.main()
