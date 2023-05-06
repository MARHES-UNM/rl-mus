from matplotlib import pyplot as plt

from uav_sim.agents.uav import Quadrotor
import unittest
import numpy as np
from uav_sim.utils.gui import Gui


class TestUav(unittest.TestCase):
    def setUp(self):
        self.uav = Quadrotor(0, 2, 2, 2)

    def test_rotation_matrix(self):
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
            uav = Quadrotor(0, x, y, z)

            for _ in range(10):
                # TODO: expected input is T, tau_x, tau_y, tau_z
                action = np.array([uav.m * uav.g, 0, 0, 0])
                uav.step(action)
                expected_traj = np.array([x, y, z])
                np.testing.assert_array_almost_equal(expected_traj, uav.state[0:3])

    def test_uav_model_gravity(self):
        """Test that the UAV fall to the ground when 0 force is applied"""
        uav = Quadrotor(0, 5, 5, 1, dt=0.1)

        for _step in range(10):
            action = np.zeros(4)
            uav.step(action)
        np.testing.assert_almost_equal(0.0, uav.state[2])

    # @unittest.skip
    def test_controller_hover(self):
        uav = Quadrotor(0, 0, 0, 2)

        des_pos = np.array([0, 0, 1, 0])
        uav_des_traj = []
        uav_trajectory = []
        for i in range(100):
            action = uav.calc_torque(des_pos)
            uav.step(action)
            uav_des_traj.append(des_pos)
            uav_trajectory.append(uav.state)

        uav_des_traj = np.array(uav_des_traj)
        uav_trajectory = np.array(uav_trajectory)

        self.plot_traj(uav_des_traj, uav_trajectory)

    def test_controller_x(self):
        uav = Quadrotor(0, 1, 0, 1)
        des_pos = np.zeros(4)
        uav_des_traj = []
        uav_trajectory = []
        for i in range(100):
            if i < 20:
                des_pos[0:3] = uav.state[0:3].copy()
                des_pos[3] = uav.state[8].copy()
            else:
                des_pos = np.array([2, 1, 2, np.pi])

            action = uav.calc_torque(des_pos)

            uav.step(action)
            uav_des_traj.append(des_pos)
            uav_trajectory.append(uav.state)

        uav_des_traj = np.array(uav_des_traj)
        uav_trajectory = np.array(uav_trajectory)

        self.plot_traj(uav_des_traj, uav_trajectory)

    def plot_traj(self, uav_des_traj, uav_trajectory):
        fig = plt.figure(figsize=(10, 6))

        ax = fig.add_subplot(411)
        ax.plot(uav_des_traj[:, 0])
        ax.plot(uav_trajectory[:, 0])
        ax.set_xlabel("t(s)")
        ax.set_ylabel("x (m)")

        ax1 = fig.add_subplot(412)
        ax1.plot(uav_des_traj[:, 1])
        ax1.plot(uav_trajectory[:, 1])
        ax1.set_ylabel("y (m)")

        ax2 = fig.add_subplot(413)
        ax2.plot(uav_des_traj[:, 2])
        ax2.plot(uav_trajectory[:, 2])
        ax2.set_ylabel("z (m)")

        ax3 = fig.add_subplot(414)
        ax3.plot(uav_des_traj[:, 3])
        ax3.plot(uav_trajectory[:, 8])
        ax3.set_ylabel("psi (rad)")

        plt.show()
        print()


if __name__ == "__main__":
    unittest.main()