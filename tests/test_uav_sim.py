import numpy as np
import unittest

from tests import context
import unittest
from uav_sim.envs.uav_sim import UavSim


class TestUavSim(unittest.TestCase):
    def setUp(self):
        self.env = UavSim()

    def test_setting_uav_pos(self):
        uav_pos = [[.5, .5, 1], [.5, 2, 2], [2, 0.5, 2], [2, 2, 1]]
        for idx, pos in enumerate(uav_pos):
            self.env.uavs[idx]._state[0] = pos[0]
            self.env.uavs[idx]._state[1] = pos[1]
            self.env.uavs[idx]._state[2] = pos[2]

        for i in range(20):
            actions = {_id: np.zeros(4) for _id in range(self.env.num_uavs)}
            self.env.step(actions)
            self.env.render()

    def test_render(self):
        tf = 100
        t = 0
        actions = {}
        while t < tf:
            for i in range(self.env.num_uavs):
                actions[i] = np.ones(4) * self.env.uavs[i].m * self.env.uavs[i].g / 4
            self.env.step(actions)
            self.env.render()
            t += 1


if __name__ == "__main__":
    unittest.main()
