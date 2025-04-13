import unittest
import numpy as np
from uav_sim.agents.uav import Obstacle, Target, Uav
from uav_sim.utils.gui import Gui


class TestGui(unittest.TestCase):
    def setUp(self):
        target = Target(_id=0, x=2, y=2, num_landing_pads=4)
        uavs = {
            0: Uav(0, 1, 1, 1, pad=target.pads[0]),
            1: Uav(1, 1, 2, 2, pad=target.pads[1]),
            2: Uav(2, 3, 1, 1, pad=target.pads[2]),
            3: Uav(4, 1, 1, pad=target.pads[3]),
        }
        self.obstacles = [Obstacle(0, 2, 2, 4)]
        self.gui = Gui(
            uavs,
            target=target,
            obstacles=self.obstacles,
            max_x=5,
            max_y=5,
            max_z=5,
        )

    def test_show_gui(self):
        self.gui.update(0.1)
        self.obstacles[0]._state[0:3] = np.array([3, 4, 5])
        self.gui.update(0.2)

        self.gui.close()


if __name__ == "__main__":
    unittest.main()
