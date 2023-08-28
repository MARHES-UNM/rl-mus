from matplotlib import pyplot as plt
import torch

from uav_sim.agents.uav import Obstacle, Pad, Quadrotor, Quad2DInt, Target
import unittest
import numpy as np

from uav_sim.utils.trajectory_generator import (
    TrajectoryGenerator,
    calculate_acceleration,
    calculate_position,
)


class TestQuad2D(unittest.TestCase):
    def setUp(self):
        self.uav = Quad2DInt(0, 2, 2, 2)

    def test_f_dot_torch(self):
        state = torch.from_numpy(self.uav.state.astype(np.float32))
        action = torch.tensor([[1.0, 1.0, 1.0]])

        dxdt = self.uav.f_dot_torch(state, action)
        dxdt_np = self.uav.f_dot(0, self.uav.state, [1.0, 1.0, 1.0])

        print(dxdt)

    def test_uav_hover(self):
        """Test that the uav can hover with the specified input"""
        for i in range(10):
            x = np.random.rand() * 3
            y = np.random.rand() * 3
            z = np.random.rand() * 3
            uav = Quad2DInt(0, x, y, z)

            for _ in range(10):
                # no need to provide input step takes calculating gravity term
                # action = np.array([0.0, 0.0, uav.m * uav.g])
                action = np.array([0.0, 0.0, 0])
                uav.step(action)
                expected_traj = np.array([x, y, z])
                np.testing.assert_array_almost_equal(expected_traj, uav.state[0:3])

    def test_time_coord_controller(self):
        tf = 10
        N = 1
        p = self.uav.get_p(tf, N)
        print(p)

        g = self.uav.get_g(2, 0, p, tf, N)

        print(g)

    def test_uav_model_gravity(self):
        """Test that the UAV fall to the ground when 0 force is applied"""
        uav = Quad2DInt(0, 5, 5, 1, dt=0.1)

        for _step in range(11):
            action = np.zeros(3)
            action[2] = -uav.m * uav.g
            uav.step(action)
        np.testing.assert_almost_equal(uav.state[2], 0.0)

    def test_landing_minimum_traj(self):
        uav = Quad2DInt(0, 1, 0, 3)
        target = Target(0, 2, 3)

        Tf = 10
        start_pos = uav.state[0:3]
        end_pos = np.array([target.pads[0].x, target.pads[0].y, 0])

        uav_coeff = np.zeros((3, 6, 1), dtype=np.float64)
        traj = TrajectoryGenerator(start_pos, end_pos, Tf)
        traj.solve()
        uav_coeff[0] = traj.x_c
        uav_coeff[1] = traj.y_c
        uav_coeff[2] = traj.z_c

        t = 0
        uav_des_traj = []
        uav_trajectory = []
        for _step in range(150):
            des_pos = np.zeros(15)
            des_pos[0] = calculate_position(uav_coeff[0], t)
            des_pos[1] = calculate_position(uav_coeff[1], t)
            des_pos[2] = calculate_position(uav_coeff[2], t)

            # acceleration
            des_pos[12] = calculate_acceleration(uav_coeff[0], t)
            des_pos[13] = calculate_acceleration(uav_coeff[1], t)
            des_pos[14] = calculate_acceleration(uav_coeff[2], t)
            # des_pos[14] = uav.m * (uav.g + des_pos[14])

            uav.step(des_pos[12:])

            uav_des_traj.append(des_pos.copy())
            uav_trajectory.append(uav.state)
            t += uav.dt

        uav_des_traj = np.array(uav_des_traj)
        uav_trajectory = np.array(uav_trajectory)

        self.plot_traj(uav_des_traj, uav_trajectory)

    def test_get_landed(self):
        pad = Pad(0, 1, 1)

        self.assertFalse(self.uav.get_landed(pad))

        des_pos = np.zeros(15)
        for i in range(100):
            des_pos[0:3] = pad.state[0:3]
            action = self.uav.calc_des_action(des_pos)

            self.uav.step(action)

        self.assertTrue(self.uav.get_landed(pad))

    def test_uav_collision(self):
        obs = Obstacle(0, 1, 1, 1)
        uav = Quad2DInt(0, 1, 0, 0.9)

        self.assertFalse(uav.in_collision(obs))

        uav.state[0:3] = np.array([1, 1, 0.9])
        self.assertTrue(uav.in_collision(obs))

    def test_controller_des_pos(self):
        """Test uav can reach a desired position."""
        uav = Quad2DInt(0, 1, 0, 1)
        des_pos = np.zeros(15)
        uav_des_traj = []
        uav_trajectory = []
        for i in range(290):
            if i < 20:
                des_pos[0:3] = uav.state[0:3].copy()
            elif i > 30 and i < 60:
                des_pos[0:3] = np.array([1, 0, 1])
            elif i > 70 and i < 100:
                des_pos[0:3] = np.array([0, 3, 0])
            elif i > 110 and i < 130:
                des_pos[0:3] = np.array([2, 1, 0.5])
            elif i > 140 and i < 170:
                des_pos[0:3] = np.array([2, 1, 2])
            elif i < 180 and i < 210:
                des_pos[0:3] = np.array([1, 0, 3])
            elif i > 220 and i < 250:
                des_pos[0:3] = np.array([3, 0, 3])
            elif i > 260:
                des_pos[0:3] = np.array([3, 2, 3])

            action = uav.calc_des_action(des_pos)

            uav.step(action)
            uav_des_traj.append(des_pos.copy())
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

        plt.show()
        print()


if __name__ == "__main__":
    unittest.main()
