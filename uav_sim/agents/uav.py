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
        use_ode=True,
        k=None,
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

        # self.inertia = np.eye(3)
        self.ixx = self.inertia[0, 0]
        self.iyy = self.inertia[1, 1]
        self.izz = self.inertia[2, 2]

        self.inv_inertia = np.linalg.pinv(self.inertia)

        self.min_f = 0.0  # Neutons kg * m / s^2
        self.max_f = 2.0 * self.m * self.g  # Neutons

        # gamma = k_M / k_F
        self.gamma = 1.5e-9 / 6.11e-8  # k_F = N / rpm^2, k_M = N*m / rpm^2

        if k is None:
            self.k = self.calc_gain()
        else:
            self.k = k

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
        Rx = np.diag([10000000])

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
        Ry = np.diag([10000000])

        # Z-subsystem
        # The state variables are z, dot_z
        Az = np.array([[0.0, 1.0], [0.0, 0.0]])
        Bz = np.array([[0.0], [1 / self.m]])

        Qz = np.diag([100, 10])
        Rz = np.diag([10000000])

        # Yaw-subsystem
        # The state variables are yaw, dot_yaw
        Ayaw = np.array([[0.0, 1.0], [0.0, 0.0]])
        Byaw = np.array([[0.0], [1 / self.izz]])
        Qyaw = np.diag([1, 1])
        Ryaw = np.diag([10000000])

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

    def f_dot(self, time, state, action):
        ft, tau_x, tau_y, tau_z = action.reshape(-1).tolist()

        omega = state[9:12].copy()
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

    def calc_torque(self, des_pos=np.array([0, 0, 0, 0])):
        """
        Inputs are the desired position x, y, z, psi
        Outputs are T, tau_x, tau_y, tau_z

        Args:
            des_pos (_type_, optional): _description_. Defaults to np.arary([0, 0, 0, 0]).
        """
        pos_er = np.zeros(12) - self._state
        pos_er[0:3] = des_pos[0:3] - self._state[0:3]
        pos_er[8] = des_pos[3] - self._state[8]

        T = np.dot(self.k[2], pos_er[[2, 5]]).squeeze()
        u_x = np.dot(self.k[0], pos_er[[0, 3, 7, 10]]).squeeze()

        u_y = np.dot(self.k[1], pos_er[[1, 4, 6, 9]]).squeeze()

        u_z = np.dot(self.k[3], pos_er[[8, 11]]).squeeze()

        return np.array([T + self.m * self.g, u_y, u_x, u_z])

    def step(self, action=np.zeros(4)):
        """Action is propeller forces in body frame

        Args:
            action (_type_, optional): _description_. Defaults to np.zeros(4).
            state:
            x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot
        """

        self.ode.set_initial_value(self._state, 0).set_f_params(action)
        self._state = self.ode.integrate(self.ode.t + self.dt)
        assert self.ode.successful()

        self._state[9:12] = self.wrap_angle(self._state[9:12])

        self._state[6:9] = self.wrap_angle(self._state[6:9])
        self._state[2] = max(0, self._state[2])

    def wrap_angle(self, val):
        return (val + np.pi) % (2 * np.pi) - np.pi
