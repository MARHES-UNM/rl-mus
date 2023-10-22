from matplotlib import pyplot as plt

from uav_sim.agents.uav import Target
import unittest
import numpy as np
from uav_sim.utils.gui import Gui


class TestUav(unittest.TestCase):
    def setUp(self):
        self.target = Target(_id=0, x=1.5, y=1.5, v=0, psi=np.pi / 2)
        self.gui = Gui(target=self.target)

    def test_target_move(self):
        t = 0
        for i in range(100):
            t = i * self.target.dt
            action = self.target.get_target_action(t, 20)
            # action = np.array([1, 0])
            self.target.step(action)
            self.gui.update(t)


if __name__ == "__main__":
    unittest.main()
