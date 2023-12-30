from matplotlib import pyplot as plt
from _archives.full_uav import Quadrotor

from uav_sim.agents.uav import Obstacle, Pad, Uav
from uav_sim.utils.utils import plot_traj
import unittest
import numpy as np


class TestUav(unittest.TestCase):
    def setUp(self):
        self.uav = Uav(0, 2, 2, 2)

    def test_rotation_matrix(self):
        """Asserts that rotation matrix is orthogonal."""
        for i in range(10):
            self.uav._state[6:9] = np.random.rand(3) * (np.pi + np.pi) - np.pi
            rot_mat = self.uav.rotation_matrix()
            np.testing.assert_almost_equal(np.linalg.det(rot_mat), 1.0)

    def test_uav_hover(self):
        """Test that the uav can hover with the specified input"""
        for i in range(10):
            x = np.random.rand() * 3
            y = np.random.rand() * 3
            z = np.random.rand() * 3
            uav = Uav(0, x, y, z)

            for _ in range(10):
                # expected input is ax, ay, az
                action = np.array([0.0, 0.0, 0.0])
                uav.step(action)
                expected_traj = np.array([x, y, z])
                np.testing.assert_array_almost_equal(expected_traj, uav.state[0:3])

    def test_uav_model_gravity(self):
        """Test that the UAV fall to the ground when 0 force is applied"""
        uav = Uav(0, 5, 5, 1, dt=0.1)

        for _step in range(10):
            action = np.zeros(4)
            uav.step(action)
        np.testing.assert_almost_equal(0.0, uav.state[2])

    def test_controller_independent_control(self):
        """Test that we can independently control UAV x, y, z and psi"""
        uav = Uav(0, 0, 0, 2)

        des_pos = np.zeros(15)
        uav_des_traj = []
        uav_trajectory = []
        for i in range(200):
            # hover
            if i < 20:
                des_pos[0:4] = np.array([0, 0, 1, 0])
            # x direction
            elif i > 30 and i < 50:
                des_pos[0:4] = np.array([1, 0, 1, 0])
            # y direction
            elif i > 60 and i < 80:
                des_pos[0:4] = np.array([1, 1, 1, 0])
            elif i > 90:
                des_pos[0:4] = np.array([1, 1, 1, 0])
                des_pos[8] = 0.3

            action = uav.calc_des_action(des_pos)
            uav.step(action)
            uav_des_traj.append(des_pos.copy())
            uav_trajectory.append(uav.state)

        uav_des_traj = np.array(uav_des_traj)
        uav_trajectory = np.array(uav_trajectory)

        plot_traj(uav_des_traj, uav_trajectory, title="Independent Control Test")

    def test_traj_line(self):
        """Test uav can follow a line trajectory"""
        uav = Uav(0, 1, 0, 1)
        uav = Uav(0, 0, 0, 0)
        uav_des_traj = []
        uav_trajectory = []
        t = 0
        t_max = 4
        while t < 20:  # 20 s
            des_pos = np.zeros(15)
            t_func = max(0, min(t, t_max))
            t_func = t_func / t_max

            # posistion
            des_pos[0:3] = 10 * t_func**3 - 15 * t_func**4 + 6 * t_func**5
            des_pos[8] = des_pos[0]

            # velocity
            des_pos[3:6] = (
                (30 / t_max) * t_func**2
                - (60 / t_max) * t_func**3
                + (30 / t_max) * t_func**4
            )
            des_pos[11] = des_pos[3]

            # acceleration
            des_pos[12:] = (
                (60 / t_max**2) * t_func
                - (180 / t_max**2) * t_func**2
                + (120 / t_max**2) * t_func**3
            )

            action = uav.calc_des_action(des_pos)

            uav.step(action)
            uav_des_traj.append(des_pos.copy())
            uav_trajectory.append(uav.state)

            t += uav.dt
        uav_des_traj = np.array(uav_des_traj)
        uav_trajectory = np.array(uav_trajectory)

        plot_traj(uav_des_traj, uav_trajectory, title="Trajectory Line Controller")

    def test_controller_des_controller(self):
        """Test uav can reach a desired position."""
        uav = Uav(0, 1, 0, 1)
        des_pos = np.zeros(15)
        uav_des_traj = []
        uav_trajectory = []
        for i in range(220):
            if i < 20:
                des_pos[0:3] = uav.state[0:3].copy()
                des_pos[8] = uav.state[8].copy()
            elif i > 30 and i < 60:
                des_pos[0:3] = np.array([2, 2, 1])
                # des_pos[8] = np.pi / 2
            elif i > 90 and i < 140:
                des_pos[0:3] = np.array([1, 0, 0])
                # des_pos[8] = 0.2
            elif i > 150 and i < 180:
                des_pos[0:3] = np.array([3, 1, 0.5])
                # des_pos[8] = 0.2
            elif i > 190:
                des_pos[0:3] = np.array([2, 2, 2])
                # des_pos[8] = np.pi
            action = uav.calc_des_action(des_pos)

            uav.step(action)
            uav_des_traj.append(des_pos.copy())
            uav_trajectory.append(uav.state)

        uav_des_traj = np.array(uav_des_traj)
        uav_trajectory = np.array(uav_trajectory)

        plot_traj(uav_des_traj, uav_trajectory, title="Test Desired Controller")

    def test_get_landed(self):
        pad = Pad(0, 1, 1)

        # uav should not be on pad
        self.assertFalse(self.uav.get_landed(pad))

        des_pos = np.zeros(15)

        uav_des_traj = []
        uav_trajectory = []
        for i in range(100):
            des_pos[0:3] = pad.state[0:3]
            action = self.uav.calc_des_action(des_pos)

            self.uav.step(action)

            uav_trajectory.append(self.uav.state)
            uav_des_traj.append(des_pos.copy())

        uav_des_traj = np.array(uav_des_traj)
        uav_trajectory = np.array(uav_trajectory)

        plot_traj(uav_des_traj, uav_trajectory, title="Test Get Landed")

        self.assertTrue(self.uav.get_landed(pad))

    def test_uav_collision(self):
        obs = Obstacle(0, 1, 1, 1)
        uav = Uav(0, 1, 0, 0.9)

        self.assertFalse(uav.in_collision(obs))

        uav.state[0:3] = np.array([1, 1, 0.9])
        self.assertTrue(uav.in_collision(obs))


if __name__ == "__main__":
    unittest.main()
