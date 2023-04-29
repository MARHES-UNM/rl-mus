from math import cos, sin
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
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return np.asarray(K), np.asarray(X), np.asarray(eigVals)


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
            self.ode = scipy.integrate.ode(self.state_dot).set_integrator(
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
            Q[0, 0] = 1  # The first state variable is the one we care about.
            R = np.diag(
                [
                    1.0,
                ]
            )
            K, _, _ = lqr(A, B, Q, R)
            Ks.append(K)

        return Ks

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

        # Q = np.ones((12, 12)) * 0.25
        # R = np.ones((4, 4))
        # R[0, 0] = 1.0
        # R[1, 1] = 10
        # R[2, 2] = 100
        # R[3, 3] = 10

        # K, _, _ = lqr(A, B, Q, R)

        # return K

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
                    [self.gamma, self.gamma, self.gamma, self.gamma],
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

            self._state[6:9] = self.wrap_angle(self._state[6:9])
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
            # self._state[9:12] += omega
            self._state[6:9] += self._state[9:12] * self.dt

            # wrap angles
            self._state[6:9] = self.wrap_angle(self._state[6:9])

    def wrap_angle(self, val):
        return (val + np.pi) % (2 * np.pi) - np.pi
