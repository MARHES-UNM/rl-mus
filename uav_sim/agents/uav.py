from math import cos, sin
import numpy as np
from uav_sim.utils.utils import lqr
from uav_sim.utils.utils import distance, angle
import scipy.integrate
import scipy
from enum import IntEnum


class AgentType(IntEnum):
    U = 0  # uav
    O = 1  # obstacle
    T = 2  # moving car as target
    P = 3  # landing pad


class ObsType(IntEnum):
    S = 0  # Static
    M = 1  # Moving


class Entity:
    def __init__(self, _id, x=0, y=0, z=0, _type=AgentType.O):
        self.id = _id
        self.type = _type
        self.x = x
        self.y = y
        self.z = z

        # x, y, z, x_dot, y_dot, z_dot
        self._state = np.array([self.x, self.y, self.z, 0, 0, 0])

    @property
    def state(self):
        self._state = np.array([self.x, self.y, self.z, 0, 0, 0])
        return self._state

    def wrap_angle(self, val):
        return (val + np.pi) % (2 * np.pi) - np.pi

    def rel_distance(self, entity):
        dist = distance((self.x, self.y), (entity.x, entity.y))

        return dist

    def rel_bearing_error(self, entity):
        """[summary]

        Args:
            entity ([type]): [description]

        Returns:
            [type]: [description]
        """
        bearing = angle((self.x, self.y), (entity.x, entity.y)) - self.theta
        # TODO: verify this from Deep RL for Swarms
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        return bearing


class Pad(Entity):
    def __init__(self, _id, x, y, _type=AgentType.P):
        self.x = x
        self.y = y
        self.id = _id
        super().__init__(_id=_id, x=x, y=y, z=0, _type=_type)

    @property
    def state(self):
        return self._state


class Target(Entity):
    def __init__(
        self,
        _id,
        x=0,
        y=0,
        psi=0,
        v=0,
        w=0,
        dt=0.1,
        num_landing_pads=1,
        pad_offset=0.5,
        r=1,
    ):
        super().__init__(_id=_id, x=x, y=y, z=0, _type=AgentType.T)
        self.id = _id
        self.dt = dt
        self.psi = psi
        self.num_landing_pads = num_landing_pads
        self._state = np.array([x, y, psi, v, w])
        self.r = r  # radius, m
        self.pad_offset = pad_offset  # m

        self.pads = [
            Pad(_id, pad_loc[0], pad_loc[1])
            for _id, pad_loc in enumerate(self.get_pad_offsets())
        ]

    @property
    def state(self):
        return self._state

    def get_pad_offsets(self):
        x = self._state[0]
        y = self._state[1]
        return [
            (x - self.pad_offset, y),
            (x + self.pad_offset, y),
            (x, y - self.pad_offset),
            (x, y + self.pad_offset),
        ]

    def update_pads_state(self):
        pad_offsets = self.get_pad_offsets()
        for pad, offset in zip(self.pads, pad_offsets):
            pad.x, pad.y = offset

            pad._state = np.array(
                [
                    pad.x,
                    pad.y,
                    0,
                    self.vx,
                    self.vy,
                    0,
                ]
            )

    def step(self, action):
        self.v = action[0]
        self.w = action[1]

        self.psi = self._state[2]
        self.vx = self.v * cos(self.psi)
        self.vy = self.v * sin(self.psi)
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.psi += self.w * self.dt

        self.psi = self.wrap_angle(self.psi)

        self._state = np.array([self.x, self.y, 0, self.vx, self.vy, 0, self.psi])

        self.update_pads_state()


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
        super().__init__(_id=_id, x=x, y=y, z=z, _type=AgentType.U)

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
        self.m = 1

        # lenght of arms
        self.l = l  # m

        self.inertia = np.array(
            [[0.00025, 0, 2.55e-6], [0, 0.000232, 0], [2.55e-6, 0, 0.0003738]]
        )

        self.inertia = np.eye(3)
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
        self.done = False

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

        Qx = np.diag([1, 1, 1, 1])
        Rx = np.diag([10])

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

        Qy = np.diag([1, 1, 1, 1])
        Ry = np.diag([10])

        # Z-subsystem
        # The state variables are z, dot_z
        Az = np.array([[0.0, 1.0], [0.0, 0.0]])
        Bz = np.array([[0.0], [1 / self.m]])

        Qz = np.diag([1, 1])
        Rz = np.diag([10])

        # Yaw-subsystem
        # The state variables are yaw, dot_yaw
        Ayaw = np.array([[0.0, 1.0], [0.0, 0.0]])
        Byaw = np.array([[0.0], [1 / self.izz]])
        Qyaw = np.diag([1, 1])
        Ryaw = np.diag([1])

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

    def get_landed(self, pad):
        dist = np.linalg.norm(self._state[0:3] - pad._state[0:3])

        return dist <= 0.01
