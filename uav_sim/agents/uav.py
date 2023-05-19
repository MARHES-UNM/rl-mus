from math import cos, sin
import math
from xml.dom.expatbuilder import theDOMImplementation
import numpy as np
from uav_sim.utils.utils import distance, angle, lqr
import scipy.integrate
import scipy
from scipy.integrate import odeint
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
    def __init__(self, _id, x=0, y=0, z=0, r=0.1, _type=AgentType.O):
        self.id = _id
        self.type = _type
        self.x = x
        self.y = y
        self.z = z
        self.r = r

        # x, y, z, x_dot, y_dot, z_dot
        self._state = np.array([self.x, self.y, self.z, 0, 0, 0])

    @property
    def state(self):
        return self._state

    @property
    def pos(self):
        return self._state[0:3]

    @property
    def vel(self):
        return self._state[3:6]

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


class Obstacle(Entity):
    def __init__(self, _id, x=0, y=0, z=0, r=0.1, dt=0.1, _type=ObsType.S):
        super().__init__(_id, x, y, z, r, _type=_type)
        self.dt = dt

    def step(self):
        if self.type == ObsType.M:
            vx = np.random.random() * 0.2
            vy = np.random.random() * 0.2

            self._state[3] = vx
            self._state[4] = vy

            self._state[0] += vx * self.dt
            self._state[1] += vy * self.dt


class Pad(Entity):
    def __init__(self, _id, x, y):
        self.x = x
        self.y = y
        self.id = _id
        super().__init__(_id=_id, x=x, y=y, z=0, _type=AgentType.P)


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
        r=1,
        num_landing_pads=1,
        pad_offset=0.5,
    ):
        super().__init__(_id=_id, x=x, y=y, z=0, r=r, _type=AgentType.T)
        self.id = _id
        self.dt = dt
        self.psi = psi
        self.num_landing_pads = num_landing_pads
        self.r = r  # radius, m
        self.pad_offset = pad_offset  # m

        # x, y, z, x_dot, y_dot, z_dot, psi, psi_dot
        self.v = v
        self.w = w
        self.vx = self.v * cos(self.psi)
        self.vy = self.v * sin(self.psi)

        # verifies psi
        if not v == 0:
            np.testing.assert_almost_equal(self.psi, np.arctan2(self.vy, self.vx))
        self._state = np.array([x, y, 0, 0, 0, 0, psi, self.w])
        self.pads = [
            Pad(_id, pad_loc[0], pad_loc[1])
            for _id, pad_loc in enumerate(self.get_pad_offsets())
        ]

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

        self._state = np.array(
            [self.x, self.y, 0, self.vx, self.vy, 0, self.psi, self.w]
        )

        self.update_pads_state()


class Quad2DInt(Entity):
    def __init__(self, _id, x=0, y=0, z=0, r=0.1, dt=1 / 10, m=0.18, l=0.086, pad=None):
        super().__init__(_id, x, y, z, r, _type=AgentType.U)

        self.ode = scipy.integrate.ode(self.f_dot).set_integrator(
            "vode", nsteps=500, method="bdf"
        )

        # timestep
        self.dt = dt  # s

        # gravity constant
        self.g = 9.81  # m/s^2

        self.m = m

        # lenght of arms
        self.l = l  # m

        self._state = np.zeros(12)
        self._state[0] = x
        self._state[1] = y
        self._state[2] = z
        self.done = False
        self.landed = False
        self.pad = pad

    def f_dot(self, time, state, action):
        action_z = 1 / self.m * action[2] - self.g
        return np.array(
            [
                state[3],
                state[4],
                state[5],
                action[0],
                action[1],
                action_z,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )

    def rotation_matrix(self):
        return np.eye(3)

    def calc_torque(self, des_pos):
        return self.calc_des_action(des_pos)

    def calc_des_action(self, des_pos):
        kx = ky = kz = 2
        k_x_dot = k_y_dot = k_z_dot = 3

        pos_er = des_pos[0:12] - self._state
        r_ddot_1 = des_pos[12]
        r_ddot_2 = des_pos[13]
        r_ddot_3 = des_pos[14]

        # https://upcommons.upc.edu/bitstream/handle/2117/112404/Thesis-Jesus_Valle.pdf?sequence=1&isAllowed=y
        r_ddot_des_x = kx * pos_er[0] + k_x_dot * pos_er[3] + r_ddot_1
        r_ddot_des_y = ky * pos_er[1] + k_y_dot * pos_er[4] + r_ddot_2
        r_ddot_des_z = kz * pos_er[2] + k_z_dot * pos_er[5] + r_ddot_3

        # takes care of hovering
        # r_ddot_des_z = self.m * (self.g + r_ddot_des_z)

        action = np.array([r_ddot_des_x, r_ddot_des_y, r_ddot_des_z])
        return action

    def step(self, action=np.zeros(3)):
        """Action is propeller forces in body frame

        Args:
            action (_type_, optional): _description_. Defaults to np.zeros(3).
            state:
            x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot
        """
        # keeps uav hovering
        action[2] = self.m * (self.g + action[2])

        self.ode.set_initial_value(self._state, 0).set_f_params(action)
        self._state = self.ode.integrate(self.ode.t + self.dt)
        assert self.ode.successful()

        self._state[2] = max(0, self._state[2])

    def in_collision(self, entity):
        dist = np.linalg.norm(self._state[0:3] - entity._state[0:3])
        return dist <= (self.r + entity.r)

    def get_landed(self, pad):
        dist = np.linalg.norm(self._state[0:3] - pad._state[0:3])
        return dist <= 0.01

    def check_dest_reached(self):
        dist = np.linalg.norm(self._state[0:3] - self.pad._state[0:3])

        return dist <= 0.01, dist

    def get_p(self, tf, N=1):
        """_summary_https://danielmuellerkomorowska.com/2021/02/16/differential-equations-with-scipy-odeint-or-solve_ivp/

        Args:
            tf (_type_): _description_
            N (int, optional): _description_. Defaults to 1.
        """

        def dp_dt(time, state, tf, N=1):
            # print(time)
            t_go = (tf - time) ** N
            p1 = state[0]
            p2 = state[1]
            p3 = state[2]
            return np.array(
                [t_go * p2**2, -p1 + t_go * p2 * p3, -2 * p2 + t_go * p3**2]
            )

        f1 = 2
        f2 = 1
        p0 = np.array([f1, 0, f2])
        t = np.arange(tf, 0.0, -0.1)
        params = (tf, N)
        p = odeint(dp_dt, p0, t, params, tfirst=True)
        return p

    def get_g(self, x, vx, p, tf, N=1):
        """_summary_https://danielmuellerkomorowska.com/2021/02/16/differential-equations-with-scipy-odeint-or-solve_ivp/

        Args:
            tf (_type_): _description_
            N (int, optional): _description_. Defaults to 1.
        """
        f1 = 2
        f2 = 1
        g0 = np.array([f1, 0, f2, f1 * x, f2 * vx])
        t = np.arange(tf, 0.0, -0.1)
        params = (tf, N)

        def ret_p(p, _t):
            dis = _t - t
            dis[dis < 0] = np.inf
            idx = dis.argmin()
            print(idx)
            return p[idx]

        # def dg_dt(time, state, tf, N, p):
        def dg_dt(time, state, ret_p, p, tf, N):
            print(f"g_time:{time}")
            # tf = 10
            # N = 1
            t_go = (tf - time) ** N
            g1 = state[3]
            g2 = state[4]
            # _p = ret_p(p, time)
            # p1 = _p[0]
            # p2 = _p[1]
            # p3 = _p[2]
            p1 = state[0]
            p2 = state[1]
            p3 = state[2]
            return np.array(
                [
                    t_go * p2**2,
                    -p1 + t_go * p2 * p3,
                    -2.0 * p2 + t_go * p3**2,
                    t_go * g2 * p2,
                    -g1 + t_go * g2 * p3,
                ]
            )
            # return np.array([t_go * p2, -g1 + t_go * g2 * p3])

        g = odeint(dg_dt, g0, t, args=(ret_p, p, tf, N), tfirst=True)
        return g


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

        Qx = np.diag([1, 1, 10, 1])
        Rx = np.diag([100])

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
        Ry = np.diag([100])

        # Z-subsystem
        # The state variables are z, dot_z
        Az = np.array([[0.0, 1.0], [0.0, 0.0]])
        Bz = np.array([[0.0], [1 / self.m]])

        Qz = np.diag([100, 1])
        Rz = np.diag([10])

        # Yaw-subsystem
        # The state variables are yaw, dot_yaw
        Ayaw = np.array([[0.0, 1.0], [0.0, 0.0]])
        Byaw = np.array([[0.0], [1 / self.izz]])
        Qyaw = np.diag([100, 10])
        Ryaw = np.diag([10])

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

    def r_dot_matrix(self):
        """ """
        cp = cos(self._state[6])
        sp = sin(self._state[6])
        ct = cos(self._state[7])
        st = sin(self._state[7])
        cg = cos(self._state[8])
        sg = sin(self._state[8])

        return np.array([[ct, 0, -cp * st], [0, 1, sp], [st, 0, cp * ct]])

    def rotation_matrix(
        self,
    ):
        """Calculates the Z-Y-X rotation matrix.
           Based on Different Linearization Control Techniques for a Quadrotor System

        Returns: R - 3 x 3 rotation matrix
        """
        cp = cos(self._state[6])
        sp = sin(self._state[6])
        ct = cos(self._state[7])
        st = sin(self._state[7])
        cg = cos(self._state[8])
        sg = sin(self._state[8])
        R_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        R_y = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        R_z = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])

        # Z Y X
        R = np.dot(R_x, np.dot(R_y, R_z))
        # # R = np.dot(np.dot(R_z, R_x), R_y)
        # R = np.array(
        #     [
        #         [ct * cg, sp * st * cg - cp * sg, cp * st * cg + sp * sg],
        #         [ct * sg, sp * st * sg + cp * cg, cp * st * sg - sp * cg],
        #         [-st, sp * ct, cp * ct],
        #     ]
        # )
        # R = np.array(
        #     [
        #         [cg * ct - sp * sg * st, -cp * sg, cg * st + ct * sp * sg],
        #         [ct * sg + cg * sp * st, cp * cg, sg * st - cg * ct * sp],
        #         [-cp * st, sp, cp * ct],
        #     ]
        # )
        # # R = np.dot(np.dot(R_z, R_y), R_x)
        R = np.dot(np.dot(R_z, R_x), R_y)
        return R

    def f_dot(self, time, state, action):
        ft, tau_x, tau_y, tau_z = action.reshape(-1).tolist()

        # omega = np.dot(self.r_dot_matrix(), state[9:12].copy())
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

    def calc_des_action(self, des_pos):
        kx = 0.12
        ky = 0.12
        kz = 1
        k_x_dot = 0.6
        k_y_dot = 0.6
        k_z_dot = 2
        k_phi = 1.2
        k_theta = 0.9
        k_psi = 1
        k_phi_dot = 5
        k_theta_dot = 2
        k_psi_dot = 2

        K = self.k
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

        pos_er = des_pos[0:12] - self._state
        # pos_er[6:7] = max(min(pos_er[6:7], 0.1), -.1)
        r_ddot_1 = des_pos[12]
        r_ddot_2 = des_pos[13]
        r_ddot_3 = des_pos[14]

        # https://upcommons.upc.edu/bitstream/handle/2117/112404/Thesis-Jesus_Valle.pdf?sequence=1&isAllowed=y
        # r_ddot_des_x = des_x_acc + kx * state_error[0] + k_x_dot * state_error[3]
        # r_ddot_des_y = des_y_acc + ky * state_error[1] + k_y_dot * state_error[4]
        r_ddot_des_x = kx * pos_er[0] + k_x_dot * pos_er[3] + r_ddot_1
        r_ddot_des_y = ky * pos_er[1] + k_y_dot * pos_er[4] + r_ddot_2
        r_ddot_des_z = kz * pos_er[2] + k_z_dot * pos_er[5] + r_ddot_3

        des_psi = des_pos[8]
        # des_psi = 0

        u1 = self.m * (self.g + r_ddot_des_z)

        # roll
        u2_phi = k_phi * (
            ((r_ddot_des_x * sin(des_psi) - r_ddot_des_y * cos(des_psi)) / self.g)
            + k_phi_dot * (pos_er[6])
        )

        # pitch
        u2_theta = k_theta * (
            ((r_ddot_des_x * cos(des_psi) + r_ddot_des_y * sin(des_psi)) / self.g)
            + k_theta_dot * (pos_er[7])
        )

        # yaw
        u2_psi = k_psi * pos_er[8] + k_psi_dot * pos_er[11]
        # u2_psi = self.k[3][0, 0] * pos_er[8] + self.k[3][0,1] * pos_er[11]

        # u2_phi = 0
        # u2_theta = 0
        action = np.array([u1, u2_phi, u2_theta, u2_psi])
        return action

    def calc_torque(self, des_pos=np.zeros(15)):
        """
        Inputs are the desired states: x, y, z, x_dot, y_dot,
        Inputs are the desired position x, y, z, psi
        Outputs are T, tau_x, tau_y, tau_z

        Args:
            des_pos (_type_, optional): _description_. Defaults to np.arary([0, 0, 0, 0]).
        """
        pos_er = des_pos[0:12] - self._state
        # pos_er[6:7] = np.max(np.min(pos_er[6:7], 0.1), -0.1)
        r_ddot_1 = des_pos[12]
        r_ddot_2 = des_pos[13]
        r_ddot_3 = des_pos[14]

        # T = np.dot(self.k[2], pos_er[[2, 5]]).squeeze() + r_ddot_3
        # u_theta = np.dot(self.k[0], pos_er[[0, 3, 7, 10]]).squeeze() + r_ddot_2
        # u_phi = np.dot(self.k[1], pos_er[[1, 4, 6, 9]]).squeeze() + r_ddot_1
        # u_psi = np.dot(self.k[3], pos_er[[8, 11]]).squeeze()

        T = np.dot(self.k[2], pos_er[[2, 5]]).squeeze()
        u_theta = np.dot(self.k[0], pos_er[[0, 3, 7, 10]]).squeeze()
        u_phi = np.dot(self.k[1], pos_er[[1, 4, 6, 9]]).squeeze()
        u_psi = np.dot(self.k[3], pos_er[[8, 11]]).squeeze()

        return np.array([T + self.m * self.g, u_phi, u_theta, u_psi])

    def step(self, action=np.zeros(4)):
        """Action is propeller forces in body frame

        Args:
            action (_type_, optional): _description_. Defaults to np.zeros(4).
            state:
            x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot
        """

        state = self._state.copy()
        state[9:12] = np.dot(self.r_dot_matrix(), state[9:12])
        self.ode.set_initial_value(state, 0).set_f_params(action)
        self._state = self.ode.integrate(self.ode.t + self.dt)
        assert self.ode.successful()

        self._state[9:12] = np.dot(
            np.linalg.inv(self.r_dot_matrix()), self._state[9:12]
        )
        self._state[9:12] = self.wrap_angle(self._state[9:12])

        self._state[6:9] = self.wrap_angle(self._state[6:9])
        self._state[2] = max(0, self._state[2])

    def in_collision(self, entity):
        dist = np.linalg.norm(self._state[0:3] - entity._state[0:3])

        return dist <= (self.r + entity.r)

    def get_landed(self, pad):
        dist = np.linalg.norm(self._state[0:3] - pad._state[0:3])

        return dist <= 0.01
