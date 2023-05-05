from math import cos, sin
import numpy as np
from uav_sim.utils.utils import lqr
import scipy.integrate
import scipy
from enum import IntEnum

from uav_sim.utils.trajectory_generator import (
    calculate_acceleration,
    calculate_position,
    calculate_velocity,
)


class AgentType(IntEnum):
    U = 0  # uav
    O = 1  # obstacle
    C = 2  # moving car as target


class ObsType(IntEnum):
    S = 0  # Static
    M = 1  # Moving


class Entity:
    def __init__(self, _id, _type=AgentType.O):
        self.id = _id
        self.type = _type


class Quadrotor(Entity):
    def __init__(
        self,
        _id,
        x=0,
        y=0,
        z=0,
        phi=0,
        theta=0,
        psi=0,
        dt=1 / 10,
        m=0.18,
        l=0.086,
        use_ode=False,
    ):
        super().__init__(_id=_id, _type=AgentType.U)

        self.use_ode = use_ode

        if self.use_ode:
            self.ode = scipy.integrate.ode(self.f_dot).set_integrator(
                "vode", nsteps=500, method="bdf"
            )

        # timestep
        self.dt = dt  # s

        # gravity constant
        self.g = 9.81  # m/s^2

        # mass
        self.m = m  # kg

        # lenght of arms
        self.l = l  # m

        self.inertia = np.array(
            [[0.00025, 0, 2.55e-6], [0, 0.000232, 0], [2.55e-6, 0, 0.0003738]]
        )

        self.ixx = self.inertia[0, 0]
        self.iyy = self.inertia[1, 1]
        self.izz = self.inertia[2, 2]

        self.inv_inertia = np.linalg.pinv(self.inertia)

        self.min_f = 0.0  # Neutons kg * m / s^2
        self.max_f = 2.0 * self.m * self.g  # Neutons

        # gamma = k_M / k_F
        self.gamma = 1.5e-9 / 6.11e-8  # k_F = N / rpm^2, k_M = N*m / rpm^2

        self._state = np.zeros(12)
        self._state[0] = x
        self._state[1] = y
        self._state[2] = z
        self._state[6] = phi
        self._state[7] = theta
        self._state[8] = psi

    @property
    def state(self):
        return self._state

    def calc_gain(self):
        # The control can be done in a decentralized style
        # The linearized system is divided into four decoupled subsystems

        # X-subsystem
        # The state variables are x, dot_x, pitch, dot_pitch
        Ax = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, self.g, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        Bx = np.array([[0.0], [0.0], [0.0], [1 / self.ixx]])

        Qx = np.diag([0.5, 100, 1, 100])
        Rx = np.diag([1.0])

        # Y-subsystem
        # The state variables are y, dot_y, roll, dot_roll
        Ay = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -self.g, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        By = np.array([[0.0], [0.0], [0.0], [1 / self.iyy]])

        Qy = np.diag([0.5, 100, 1, 100])
        Ry = np.diag([1.0])

        # Z-subsystem
        # The state variables are z, dot_z
        Az = np.array([[0.0, 1.0], [0.0, 0.0]])
        Bz = np.array([[0.0], [1 / self.m]])

        Qz = np.diag(
            [
                10,
                1,
            ]
        )
        Rz = np.diag([1.0])

        # Yaw-subsystem
        # The state variables are yaw, dot_yaw
        Ayaw = np.array([[0.0, 1.0], [0.0, 0.0]])
        Byaw = np.array([[0.0], [1 / self.izz]])
        Qyaw = np.diag([1, 1])
        Ryaw = np.diag([1.0])

        ####################### solve LQR #######################
        Ks = []  # feedback gain matrices K for each subsystem
        for A, B, Q, R in (
            (Ax, Bx, Qx, Rx),
            (Ay, By, Qy, Ry),
            (Az, Bz, Qz, Rz),
            (Ayaw, Byaw, Qyaw, Ryaw),
        ):
            K, _, _ = lqr(A, B, Q, R)
            Ks.append(K)

        return Ks

    def get_controller(self, des_pos_w):
        state_error = des_pos_w - self.state

        kx = 10
        ky = 10
        kz = 1
        k_x_dot = 100
        k_y_dot = 100
        k_z_dot = 10
        k_phi = 25
        k_theta = 25
        k_psi = 25
        k_phi_dot = 25
        k_theta_dot = 25
        k_psi_dot = 25

        K = self.calc_k()
        # print(K)

        # [array([[10.        ,  8.30334392, 33.81728717,  8.22408502]]), array([[-10.        ,  -8.68764941,  37.02012078,   8.60472205]]), array([[0.1       , 0.19235384]]), array([[0.1       , 0.44833024]])]
        kx = 10
        k_x_dot = 8.3
        ky = -10
        k_y_dot = -8.68
        kz = 0.1
        k_z_dot = 1
        k_phi = 33
        k_theta = 37
        k_psi = 0.1
        k_psi_dot = 0.448333024
        # Ky = K[0].squeeze()
        # Kx = K[1].squeeze()
        # Kz = K[2].squeeze()
        # K_psi = K[3].squeeze()

        # kx = Kx[0]
        # k_x_dot = Kx[1]
        # k_theta = Kx[2]
        # k_theta_dot = Kx[3]

        # ky = Ky[0]
        # k_y_dot = Ky[1]
        # k_phi = Ky[2]
        # k_phi_dot = Ky[3]

        # kz = Kz[0]
        # k_z_dot = Kz[1]

        # k_psi = K_psi[0]
        # k_psi_dot = K_psi[1]
        des_x_acc = 0
        des_y_acc = 0
        des_z_acc = 0
        # https://upcommons.upc.edu/bitstream/handle/2117/112404/Thesis-Jesus_Valle.pdf?sequence=1&isAllowed=y
        r_ddot_des_x = des_x_acc + kx * state_error[0] + k_x_dot * state_error[3]
        r_ddot_des_y = des_y_acc + ky * state_error[1] + k_y_dot * state_error[4]
        r_ddot_des_x = des_x_acc
        r_ddot_des_y = des_y_acc
        r_ddot_des_z = des_z_acc + kz * state_error[2] + k_z_dot * state_error[5]

        # des_psi = self.state[8]
        des_psi = des_pos_w[8]

        u1 = self.m * self.g + self.m * (r_ddot_des_z)
        # roll
        u2_phi = (
            k_phi
            * (
                ((r_ddot_des_x * sin(des_psi) - r_ddot_des_y * cos(des_psi)) / self.g)
                - self.state[6]
            )
            # + k_phi_dot * state_error[9]
        )

        # pitch
        u2_theta = (
            k_theta
            * (
                ((r_ddot_des_x * cos(des_psi) + r_ddot_des_y * sin(des_psi)) / self.g)
                - self.state[7]
            )
            # + k_theta_dot * self.state[10]
        )

        # yaw
        u2_psi = k_psi * state_error[8] + k_psi_dot * state_error[11]

        action = np.array([u1, u2_phi, u2_theta, u2_psi])
        return action

    def get_controller_with_coeffs(self, coeffs, t):
        des_pos_w = np.zeros(12)
        des_pos_w[2] = calculate_position(coeffs[2], t)
        des_pos_w[5] = calculate_velocity(coeffs[2], t)
        des_pos_w[8] = np.pi

        des_x_acc = calculate_acceleration(coeffs[0], t)
        des_y_acc = calculate_acceleration(coeffs[1], t)
        des_z_acc = calculate_acceleration(coeffs[2], t)

        # des_pos_w[0:3] = 2.5

        state_error = des_pos_w - self.state

        kx = 1
        ky = 1
        kz = 1
        k_x_dot = 10
        k_y_dot = 10
        k_z_dot = 1
        k_phi = 25
        k_theta = 25
        k_psi = 25
        k_phi_dot = 25
        k_theta_dot = 25
        k_phi_dot = 25

        # https://upcommons.upc.edu/bitstream/handle/2117/112404/Thesis-Jesus_Valle.pdf?sequence=1&isAllowed=y
        # r_ddot_des_x = des_x_acc + kx * state_error[0] + k_x_dot * state_error[3]
        # r_ddot_des_y = des_y_acc + ky * state_error[1] + k_y_dot * state_error[4]
        r_ddot_des_x = des_x_acc
        r_ddot_des_y = des_y_acc
        r_ddot_des_z = des_z_acc + kz * state_error[2] + k_z_dot * state_error[5]

        # des_psi = self.state[8]
        des_psi = des_pos_w[8]

        u1 = self.m * self.g + self.m * (r_ddot_des_z)
        # roll
        u2_phi = k_phi * (
            ((r_ddot_des_x * sin(des_psi) - r_ddot_des_y * cos(des_psi)) / self.g)
            - self.state[6]
        )

        # pitch
        u2_theta = (
            k_theta
            * (
                ((r_ddot_des_x * cos(des_psi) - r_ddot_des_y * sin(des_psi)) / self.g)
                - self.state[7]
            )
            # + k_theta_dot * self.state[10]
        )

        # yaw
        u2_psi = k_psi * state_error[8]  # + k_psi_dot * state_error[11]

        u2_phi = 0
        u2_theta = 0
        action = np.array([u1, u2_phi, u2_theta, u2_psi])
        return action

    # def get_controller(self, des_pos_w):
    #     state_error = des_pos_w - self.state

    #     # TODO: don't calculate every time
    # K = self.calc_k()
    # Kx = K[0].squeeze()
    # Ky = K[1].squeeze()
    # Kz = K[2].squeeze()
    # K_psi = K[3].squeeze()

    # kx = Kx[0]
    # k_x_dot = Kx[1]
    # k_theta = Kx[2]
    # k_theta_dot = Kx[3]

    # ky = Ky[0]
    # k_y_dot = Ky[1]
    # k_phi = Ky[2]
    # k_phi_dot = Ky[3]

    # kz = Kz[0]
    # k_z_dot = Kz[1]

    # k_psi = K_psi[0]
    # k_psi_dot = K_psi[1]

    #     kx = k_x_dot = 1
    #     ky = k_y_dot = 1
    #     kz = k_z_dot = 1
    #     k_phi = k_phi_dot = 25
    #     k_theta = k_theta_dot = 25
    #     k_psi = k_psi_dot = 1

    #     # https://upcommons.upc.edu/bitstream/handle/2117/112404/Thesis-Jesus_Valle.pdf?sequence=1&isAllowed=y
    #     r_ddot_des_x = kx * state_error[0] + k_x_dot * state_error[3]
    #     r_ddot_des_y = ky * state_error[1] + k_y_dot * state_error[4]
    #     r_ddot_des_z = kz * state_error[2] + k_z_dot * state_error[5]

    #     # des_psi = self.state[8]
    #     des_psi = des_pos_w[8]

    #     u1 = self.m * self.g + self.m * (r_ddot_des_z)
    #     phi_des = (
    #         1 / self.g * (r_ddot_des_x * sin(des_psi) - r_ddot_des_y * cos(des_psi))
    #     )
    #     # phi_des = ky * state_error[1] + k_y_dot * state_error[4]
    #     theta_des = (
    #         1 / self.g * (r_ddot_des_x * cos(des_psi) - r_ddot_des_y * sin(des_psi))
    #     )
    #     # theta_des = kx * state_error[0] + k_x_dot * state_error[3]
    #     # theta_des = np.clip(theta_des, -0.17, 0.17)
    #     # phi_des = np.clip(phi_des, -0.17, 0.17)

    #     u2_theta = k_theta * (theta_des - self.state[7]) - k_theta_dot * self.state[10]
    #     # u2_x = k_theta * (state_error[7]) - k_theta_dot * self.state[10]
    #     u2_phi = k_phi * (phi_des - self.state[6]) - k_phi_dot * self.state[9]
    #     # u2_y = k_phi * (state_error[6]) - k_phi_dot * self.state[10]

    #     # roll
    #     u2_phi = (
    #         k_phi
    #         * (
    #             ((r_ddot_des_x * sin(des_psi) - r_ddot_des_y * cos(des_psi)) / self.g)
    #             - self.state[6]
    #         )
    #         # + k_phi_dot * state_error[9]
    #         # + k_phi_dot * self.state[9]
    #     )

    #     # pitch
    #     u2_theta = (
    #         k_theta
    #         * (
    #             ((r_ddot_des_x * cos(des_psi) - r_ddot_des_y * sin(des_psi)) / self.g)
    #             - self.state[7]
    #         )
    #         # + k_theta_dot * state_error[10]q
    #         # + k_theta_dot * self.state[10]
    #     )

    #     # yaw
    #     u2_psi = k_psi * state_error[8] + k_psi_dot * state_error[11]

    #     # u2_psi = 0
    #     u2_theta = 0
    #     u2_phi = 0
    #     # action = np.dot(
    #     #     np.linalg.inv(self.torque_to_inputs()),
    #     #     np.array([u1, u2_phi, u2_theta, u2_psi]),
    #     # )
    #     return np.array([u1, u2_phi, u2_theta, u2_psi])
    #     # action = np.dot(
    #     #     np.linalg.pinv(
    #     #         np.array(
    #     #             [
    #     #                 [0, self.l, 0, -self.l],
    #     #                 [-self.l, 0, self.l, 0],
    #     #                 [self.gamma, -self.gamma, self.gamma, -self.gamma],
    #     #             ]
    #     #         )
    #     #     ),
    #     #     np.array([u2_phi, u2_theta, u2_psi]),
    #     # )
    #     # theta_r = kx * state_error[0] + k_x_dot * state_error[3]
    #     # tau_x = k_theta * (theta_r - self.state[7]) - k_theta_dot * self.state[10]
    #     # tau_x = k_theta * (state_error[7]) - k_theta_dot * self.state[10]

    #     # phi_r = ky * state_error[1] + k_y_dot * state_error[4]
    #     # tau_y = k_phi * (phi_r - self.state[6]) - k_phi_dot * self.state[9]
    #     # tau_y = k_phi * (state_error[6]) - k_phi_dot * self.state[9]

    #     # tau_z = k_psi * state_error[8] + k_psi_dot * state_error[11]

    #     # des_actions = np.array([u1, tau_x, tau_y, tau_z])

    #     # action = np.dot(np.linalg.inv(self.torque_to_inputs()), des_actions)

    #     return action

    def torque_to_inputs(self):
        # t_array = np.ones((1,4))
        # l = np.array(
        #     [
        #         [0, self.l, 0, -self.l],
        #         [-self.l, 0, self.l, 0],
        #         [self.gamma, -self.gamma, self.gamma, -self.gamma],
        #     ]
        # )
        # l = np.dot(self.inv_inertia, l)

        # l = np.concatenate([t_array, l])
        l = np.array(
            [
                [1, 1, 1, 1],
                [0, self.l, 0, -self.l],
                [-self.l, 0, self.l, 0],
                [self.gamma, -self.gamma, self.gamma, -self.gamma],
            ]
        )
        return l

    def rotation_matrix(
        self,
    ):
        cp = cos(self._state[6])
        ct = cos(self._state[7])
        cg = cos(self._state[8])
        sp = sin(self._state[6])
        st = sin(self._state[7])
        sg = sin(self._state[8])
        R_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        R_y = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        R_z = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
        # R = np.dot(R_z, np.dot(R_y, R_x))
        R = np.dot(R_y, np.dot(R_x, R_z))
        return R

    def f(self, action):
        ft, tau_x, tau_y, tau_z = action.reshape(-1).tolist()

        omega = self._state[9:12]
        tau = np.array([tau_x, tau_y, tau_z])
        # tau = np.zeros(3)

        omega_dot = np.dot(
            self.inv_inertia, (tau - np.cross(omega, np.dot(self.inertia, omega)))
        )

        # omega_dot = self.wrap_angle(omega_dot)

        self._state[9:12] += omega_dot * self.dt
        # self._state[9] += tau_x * self.dt / self.ixx
        # self._state[10] += tau_y * self.dt / self.iyy
        # self._state[11] += tau_z * self.dt / self.izz
        self._state[9:12] = self.wrap_angle(self._state[9:12])

        self._state[6] += self._state[9] * self.dt
        self._state[7] += self._state[10] * self.dt
        self._state[8] += self._state[11] * self.dt
        self._state[6:9] = self.wrap_angle(self._state[6:9])

        # R = rotation_matrix(self._state[6], self._state[7], self._state[8])
        R = self.rotation_matrix()
        acc = (
            np.dot(R, np.array([0, 0, ft], dtype=np.float64).T)
            - np.array([0, 0, self.m * self.g], dtype=np.float64).T
        ) / self.m

        # acc = np.zeros(3)
        m = self.m
        g = self.g
        phi1 = self._state[6]
        theta1 = self._state[7]
        psi1 = self._state[8]
        # acc[0] = (
        #     ft
        #     / m
        #     * (
        #         np.sin(phi1) * np.sin(psi1)
        #         + np.cos(phi1) * np.cos(psi1) * np.sin(theta1)
        #     )
        # )
        # acc[1] = (
        #     ft
        #     / m
        #     * (
        #         np.cos(phi1) * np.sin(psi1) * np.sin(theta1)
        #         - np.cos(psi1) * np.sin(phi1)
        #     )
        # )
        # acc[2] = (-g + ft / m * np.cos(phi1) * np.cos(theta1))
        self._state[3] += acc[0] * self.dt
        self._state[4] += acc[1] * self.dt
        self._state[5] += acc[2] * self.dt

        self._state[0] += self._state[3] * self.dt
        self._state[1] += self._state[4] * self.dt
        self._state[2] += self._state[5] * self.dt

        self._state[2] = max(0, self._state[2])

        s_phi = sin(self._state[6])
        c_phi = cos(self._state[6])
        s_theta = sin(self._state[7])
        c_theta = cos(self._state[7])

        # phi_rot = np.array(
        #     [
        #         [c_theta, 0, -c_phi * s_theta],
        #         [0, 1, s_phi],
        #         [s_theta, 0, c_phi * c_theta],
        #     ]
        # )

        # # p, q, r
        # omega = np.dot(phi_rot, self._state[9:12])

        # self._state[9:12] += omega_dot * self.dt
        # self._state[9:12] = np.dot(np.linalg.inv(phi_rot), omega)

    def f_dot(self, time, state, action):
        ft, tau_x, tau_y, tau_z = action.reshape(-1).tolist()

        omega = state[9:12]
        tau = np.array([tau_x, tau_y, tau_z])

        omega_dot = np.dot(
            self.inv_inertia, (tau - np.cross(omega, np.dot(self.inertia, omega)))
        )

        R = self.rotation_matrix()
        acc = (
            np.dot(R, np.array([0, 0, ft], dtype=np.float64).T)
            - np.array([0, 0, self.m * self.g], dtype=np.float64).T
        ) / self.m

        dot_x = np.array(
            [
                state[3],
                state[4],
                state[5],
                acc[0],
                acc[1],
                acc[2],
                state[9],
                state[10],
                state[11],
                omega_dot[0],
                omega_dot[1],
                omega_dot[2],
            ]
        )

        return dot_x

    def step(self, action=np.zeros(4)):
        """Action is propeller forces in body frame

        Args:
            action (_type_, optional): _description_. Defaults to np.zeros(4).
            state:
            x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot
        """

        if self.use_ode:
            self.ode.set_initial_value(self._state, 0).set_f_params(action)
            self._state = self.ode.integrate(self.ode.t + self.dt)
            assert self.ode.successful()

            self._state[9:12] = self.wrap_angle(self._state[9:12])

            self._state[6:9] = self.wrap_angle(self._state[6:9])
            self._state[2] = max(0, self._state[2])
            # self.f(action)

            # s_phi = sin(self._state[6])
            # c_phi = cos(self._state[6])
            # s_theta = sin(self._state[7])
            # c_theta = cos(self._state[7])
            # psi = self._state[8]

            # phi_rot = np.array(
            #     [
            #         [c_theta, 0, -c_phi * s_theta],
            #         [0, 1, s_phi],
            #         [s_theta, 0, c_phi * c_theta],
            #     ]
            # )

            # # p, q, r
            # # omega = np.dot(np.linalg.inv(phi_rot), self._state[9:12])
            # self._state[9:12] = self.wrap_angle(self._state[9:12])

            # self._state[6:9] = self.wrap_angle(self._state[6:9])

            # # self.f(action)
            # # for i in range(6, 8):
            # # self._state[i] = max(90 * np.pi / 180, self._state[i])
            # # self._state[i] = min(-np.pi * 90 / 180, self._state[i])
            # self._state[2] = max(0, self._state[2])
        else:
            action = np.clip(action, self.min_f, self.max_f)
            x_ddot = (
                np.array([0, 0, -self.m * self.g])
                + np.dot(self.rotation_matrix(), np.array([0, 0, action.sum()]))
            ) / self.m

            s_phi = sin(self._state[6])
            c_phi = cos(self._state[6])
            s_theta = sin(self._state[7])
            c_theta = cos(self._state[7])

            phi_rot = np.array(
                [
                    [c_theta, 0, -c_phi * s_theta],
                    [0, 1, s_phi],
                    [s_theta, 0, c_phi * c_theta],
                ]
            )

            # p, q, r
            omega = np.dot(phi_rot, self._state[9:12])

            tau = np.dot(
                np.array(
                    [
                        [0, self.l, 0, -self.l],
                        [-self.l, 0, self.l, 0],
                        [self.gamma, -self.gamma, self.gamma, -self.gamma],
                    ]
                ),
                action,
            )

            omega_dot = np.dot(
                self.inv_inertia, (tau - np.cross(omega, np.dot(self.inertia, omega)))
            )

            self._state[3:6] += x_ddot * self.dt
            self._state[0:3] += self._state[3:6] * self.dt

            # make sure uav don't fall below z=0
            self._state[2] = max(0, self._state[2])

            omega += omega_dot * self.dt
            self._state[9:12] += np.dot(np.linalg.inv(phi_rot), omega)
            self._state[6:9] += self._state[9:12] * self.dt

            # wrap angles
            self._state[6:12] = self.wrap_angle(self._state[6:12])

    def wrap_angle(self, val):
        return (val + np.pi) % (2 * np.pi) - np.pi
