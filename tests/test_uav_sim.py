import numpy as np
import unittest

from tests import context
import unittest
from uav_sim.envs.uav_sim import UavSim


class TestUavSim(unittest.TestCase):
    def setUp(self):
        self.env = UavSim()

    def test_render(self):
        tf = 100
        t = 0
        actions = {}
        while t < tf:
            actions = np.zeros(4)
            self.env.step(actions)
            self.env.render()
            t += 1


if __name__ == "__main__":
    unittest.main()
