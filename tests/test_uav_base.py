from uav_sim.agents.uav import Obstacle, Pad, UavBase
import unittest
import numpy as np


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
                action = np.array([0.0, 0.0, 0.0])
                uav.step(action)
                expected_traj = np.array([x, y, z])
                np.testing.assert_array_almost_equal(expected_traj, uav.state[0:3])

    def test_get_landed(self):
        pad = Pad(0, 1, 1)

        uav = UavBase(0, 2, 2, 0, pad=pad)

        (is_reached, dist, vel) = uav.check_dest_reached()
        self.assertFalse(is_reached)

        uav._state = np.array([1, 1, 0, 0, 0, 0])
        (is_reached, dist, vel) = uav.check_dest_reached()
        self.assertTrue(is_reached)

    def test_uav_collision(self):
        obs = Obstacle(0, 1, 1, 1)
        uav = UavBase(0, 1, 0, 0.9)

        self.assertFalse(uav.in_collision(obs))

        uav.state[0:3] = np.array([1, 1, 0.9])
        self.assertTrue(uav.in_collision(obs))
    
    # TODO: update
    def test_uav_t_go_est(self):
        t_go_est = self.uav.get_t_go_est()
        print(t_go_est)

        self.uav._state = np.array([1, 1, 1, 0, 0, 1])
        t_go_est = self.uav.get_t_go_est()
        print(t_go_est)


if __name__ == "__main__":
    unittest.main()
