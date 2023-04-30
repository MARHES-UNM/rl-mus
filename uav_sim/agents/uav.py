from math import cos, sin
import stat
import numpy as np
import scipy.integrate
import scipy
from zmq import IntEnum


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


def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u

    """
    # http://www.mwm.im/lqr-controllers-with-python/
    # https://github.com/ssloy/tutorials/blob/master/tutorials/pendulum/lqr.py
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # compute the LQR gain
    K = np.dot(np.linalg.inv(R), np.dot(B.T, X))

    eig_vals, eig_vecs = np.linalg.eig(A - np.dot(B, K))

    return K, X, eig_vals


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
            self.ode = scipy.integrate.ode(self.f).set_integrator(
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

        # self.inertia = np.eye(3) * 0.00025

        # self.m = 1.0
        # self.inertia = np.eye(3)
        # self.inertia[0, 0] = 8.1 * 1e-3
        # self.inertia[1, 1] = 8.1 * 1e-3
        # self.inertia[2, 2] = 14.2 * 1e-3

        self.inv_inertia = np.linalg.inv(self.inertia)

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

    def calc_k(self):
        Ix = self.inertia[0, 0]
        Iy = self.inertia[1, 1]
        Iz = self.inertia[2, 2]
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
        Bx = np.array([[0.0], [0.0], [0.0], [1 / Ix]])

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
        By = np.array([[0.0], [0.0], [0.0], [1 / Iy]])

        # Z-subsystem
        # The state variables are z, dot_z
        Az = np.array([[0.0, 1.0], [0.0, 0.0]])
        Bz = np.array([[0.0], [1 / self.m]])

        # Yaw-subsystem
        # The state variables are yaw, dot_yaw
        Ayaw = np.array([[0.0, 1.0], [0.0, 0.0]])
        Byaw = np.array([[0.0], [1 / Iz]])

        ####################### solve LQR #######################
        Ks = []  # feedback gain matrices K for each subsystem
        for A, B in ((Ax, Bx), (Ay, By), (Az, Bz), (Ayaw, Byaw)):
            n = A.shape[0]
            m = B.shape[1]
            Q = np.eye(n)
            Q[0, 0] = 25  # The first state variable is the one we care about.
            R = np.diag(
                [
                    1.0,
                ]
            )
            K, _, _ = lqr(A, B, Q, R)
            Ks.append(K)

        return Ks

        # return Ks
        A = np.zeros((12, 12), dtype=np.float64)
        A[0:3, 3:6] = np.eye(3)
        # A[5, 5] = -self.g
        A[3, 6] = self.g
        A[4, 7] = -self.g
        A[6:9, 9:12] = np.eye(3)

        ix = self.inertia[0, 0]
        iy = self.inertia[1, 1]
        iz = self.inertia[2, 2]
        B = np.zeros((12, 4))
        B[5, :] = 1 / self.m
        # B[5, 0] = 1 / self.m
        # B[9, 1] = 1 / ix
        # B[10, 2] = 1 / iy
        # B[11, 3] = 1 / iz
        # # B[9:12, 1:] = np.eye(3)
        B[9:12, :] = np.dot(
            self.inv_inertia,
            np.array(
                [
                    [0, self.l, 0, -self.l],
                    [-self.l, 0, self.l, 0],
                    [self.gamma, -self.gamma, self.gamma, -self.gamma],
                ]
            ),
        )

        # A = np.zeros((12, 12), dtype=np.float64)
        # A[0, 1] = 1.0
        # A[1, 8] = self.g
        # A[2, 3] = 1.0
        # A[3, 6] = -self.g
        # A[4, 5] = 1.0
        # A[6, 7] = 1.0
        # A[8, 9] = 1.0
        # A[10, 11] = 1.0

        # B = np.zeros((12, 4))
        # ix = self.inertia[0, 0]
        # iy = self.inertia[1, 1]
        # iz = self.inertia[2, 2]
        # B[5, 0] = 1 / self.m
        # B[7, 1] = 1 / ix
        # B[9, 2] = 1 / iy
        # B[11, 3] = 1 / iz

        Q = np.eye(12) * 1000
        # Q[3, 3] = 1000000000

        # Q[:3,:3] = np.eye(3) * 110
        R = np.eye(4)
        # R = np.diag(
        #     [
        #         1.0,
        #     ]
        # )
        # R[0, 0] = 1.0
        # R[1, 1] = 10
        # R[2, 2] = 100
        # R[3, 3] = 10

        K, _, _ = lqr(A, B, Q, R)
        # positions = np.array([[0.5, 0.5, 1], [0.5, 2, 2], [2, 0.5, 2], [2, 2, 1]])
        # # positions = np.array([[4, 4, 1], [4, 2, 2], [4, 4, 2], [4, 4, 1]])
        # des_pos = np.zeros((4, 12), dtype=np.float64)
        # for idx in range(4):
        #     des_pos[idx, 0:3] = positions[idx, :]

        #     # self._state[0:3] = positions[idx, :]

        # error = des_pos[0, :] - self.state

        # u = dlqr_dyn(error, Q, R, A, B, dt=0.01)

        # return u
        return K

    def get_controller(self, des_pos_w):
        state_error = self.state - des_pos_w

        # TODO: don't calculate every time
        K = self.calc_k()
        Kx = K[0].squeeze()
        Ky = K[1].squeeze()
        Kz = K[2].squeeze()
        K_psi = K[3].squeeze()

        kx = Kx[0]
        k_x_dot = Kx[1]
        k_theta = Kx[2]
        k_theta_dot = Kx[3]

        ky = Ky[0]
        k_y_dot = Ky[1]
        k_phi = Ky[2]
        k_phi_dot = Ky[3]

        kz = Kz[0]
        k_z_dot = Kz[1]

        k_psi = K_psi[0]
        k_psi_dot = K_psi[1]

        # https://upcommons.upc.edu/bitstream/handle/2117/112404/Thesis-Jesus_Valle.pdf?sequence=1&isAllowed=y
        r_ddot_des_x = kx * state_error[0] + k_x_dot * state_error[3]
        r_ddot_des_y = ky * state_error[1] + k_y_dot * state_error[4]
        r_ddot_des_z = kz * state_error[2] + k_z_dot * state_error[5]

        cur_psi = self.state[8]
        u1 = self.m * self.g - self.m * (r_ddot_des_z)
        phi_des = (
            1 / self.g * (r_ddot_des_x * sin(cur_psi) - r_ddot_des_y * cos(cur_psi))
        )
        # phi_des = ky * state_error[1] + k_y_dot * state_error[4]
        theta_des = (
            1 / self.g * (r_ddot_des_x * cos(cur_psi) + r_ddot_des_y * sin(cur_psi))
        )
        # theta_des = kx * state_error[0] + k_x_dot * state_error[3]
        # theta_des = np.clip(theta_des, -0.17, 0.17)
        # phi_des = np.clip(phi_des, -0.17, 0.17)

        u2_theta = k_theta * (theta_des - self.state[7]) - k_theta_dot * self.state[10]
        # u2_x = k_theta * (state_error[7]) - k_theta_dot * self.state[10]
        u2_phi = k_phi * (phi_des - self.state[6]) - k_phi_dot * self.state[9]
        # u2_y = k_phi * (state_error[6]) - k_phi_dot * self.state[10]
        u2_psi = k_psi * state_error[8] + k_psi_dot * state_error[11]

        # u2_psi = 0
        u2_theta = 0
        u2_phi = 0
        action = np.dot(
            np.linalg.inv(self.torque_to_inputs()),
            np.array([u1, u2_phi, u2_theta, u2_psi]),
        )
        return np.array([u1, u2_phi, u2_theta, u2_psi])
        # action = np.dot(
        #     np.linalg.pinv(
        #         np.array(
        #             [
        #                 [0, self.l, 0, -self.l],
        #                 [-self.l, 0, self.l, 0],
        #                 [self.gamma, -self.gamma, self.gamma, -self.gamma],
        #             ]
        #         )
        #     ),
        #     np.array([u2_phi, u2_theta, u2_psi]),
        # )
        # theta_r = kx * state_error[0] + k_x_dot * state_error[3]
        # tau_x = k_theta * (theta_r - self.state[7]) - k_theta_dot * self.state[10]
        # tau_x = k_theta * (state_error[7]) - k_theta_dot * self.state[10]

        # phi_r = ky * state_error[1] + k_y_dot * state_error[4]
        # tau_y = k_phi * (phi_r - self.state[6]) - k_phi_dot * self.state[9]
        # tau_y = k_phi * (state_error[6]) - k_phi_dot * self.state[9]

        # tau_z = k_psi * state_error[8] + k_psi_dot * state_error[11]

        # des_actions = np.array([u1, tau_x, tau_y, tau_z])

        # action = np.dot(np.linalg.inv(self.torque_to_inputs()), des_actions)

        return action

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
        ct = cos(self._state[7])
        cp = cos(self._state[6])
        cg = cos(self._state[8])
        st = sin(self._state[7])
        sp = sin(self._state[6])
        sg = sin(self._state[8])
        R_x = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_z = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def state_dot(self, time, state, action=np.zeros(4)):
        action = np.clip(action, self.min_f, self.max_f)
        state_dot = np.zeros(12)

        # The velocities(t+1 x_dots equal the t x_dots)
        state_dot[0] = self._state[3]
        state_dot[1] = self._state[4]
        state_dot[2] = self._state[5]

        # The acceleration
        x_ddot = (
            np.array([0, 0, -self.m * self.g])
            + np.dot(self.rotation_matrix(), np.array([0, 0, action.sum()]))
        ) / self.m
        state_dot[3] = x_ddot[0]
        state_dot[4] = x_ddot[1]
        state_dot[5] = x_ddot[2]

        # the angular rates
        state_dot[6] = self._state[9]
        state_dot[7] = self._state[10]
        state_dot[8] = self._state[11]

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

        omega = self._state[9:12]

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

        state_dot[9] = omega_dot[0]
        state_dot[10] = omega_dot[1]
        state_dot[11] = omega_dot[2]

        return state_dot

    def f(self, time, state, action):
        m = self.m
        g = self.g
        Ix = self.inertia[0, 0]
        Iy = self.inertia[1, 1]
        Iz = self.inertia[2, 2]
        action = np.clip(action, self.min_f, self.max_f)
        x1 = self._state[0]
        y1 = self._state[1]
        z1 = self._state[2]
        x2 = self._state[3]
        y2 = self._state[4]
        z2 = self._state[5]
        phi1 = self._state[6]
        theta1 = self._state[7]
        psi1 = self._state[8]
        phi2 = self._state[9]
        theta2 = self._state[10]
        psi2 = self._state[11]

        # x1, x2, y1, y2, z1, z2, phi1, phi2, theta1, theta2, psi1, psi2 = x.reshape(-1).tolist()
        # ft, tau_x, tau_y, tau_z = u.reshape(-1).tolist()
        ft, tau_x, tau_y, tau_z = action.reshape(-1).tolist()
        dot_x = np.array(
            [
                x2,
                y2,
                z2,
                ft
                / m
                * (
                    np.sin(phi1) * np.sin(psi1)
                    + np.cos(phi1) * np.cos(psi1) * np.sin(theta1)
                ),
                ft
                / m
                * (
                    np.cos(phi1) * np.sin(psi1) * np.sin(theta1)
                    - np.cos(psi1) * np.sin(phi1)
                ),
                -g + ft / m * np.cos(phi1) * np.cos(theta1),
                phi2,
                theta2,
                psi2,
                (Iy - Iz) / Ix * theta2 * psi2 + tau_x / Ix,
                (Iz - Ix) / Iy * phi2 * psi2 + tau_y / Iy,
                (Ix - Iy) / Iz * phi2 * theta2 + tau_z / Iz,
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

            # self._state[6:12] = self.wrap_angle(self._state[6:12])
            self._state[2] = max(0, self._state[2])

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
