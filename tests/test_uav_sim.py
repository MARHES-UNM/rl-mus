import numpy as np
import unittest

import context

from uav_sim.envs.uav_sim import UavSim, Quadrotor

class TestUavSim(unittest.TestCase):
    def setUp(self):
        self.env = UavSim()

    def test_render(self):
        tf = 10000
        t = 0
        actions = {}
        while t < tf:
            actions = np.zeros(4)
            self.env.step(actions)
            self.env.render()
            t += 1


class TestUav(unittest.TestCase):
    def setUp(self):
        self.uav = Quadrotor(3, 4, 5)

    def test_rotation_matrix(self):
        for i in range(9):
            self.uav._state[6:9] = np.random.rand(3) * (np.pi + np.pi) - np.pi
            rot_mat = self.uav.rotation_matrix()
            np.testing.assert_almost_equal(np.linalg.det(rot_mat), 1.0)


if __name__ == "__main__":
    unittest.main()
