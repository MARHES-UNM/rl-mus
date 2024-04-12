from math import cos, sin
import numpy as np
import torch
from uav_sim.utils.utils import cir_traj, distance, angle, lqr
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
        self._state = np.array([self.x, self.y, self.z, 0.0, 0.0, 0.0])

    @property
    def state(self):
        return self._state

    @property
    def pos(self):
        return self._state[0:3]

    @property
    def vel(self):
        return self._state[3:6]

    def in_collision(self, entity):
        dist = np.linalg.norm(self._state[0:3] - entity.state[0:3])
        return dist <= (self.r + entity.r)

    def wrap_angle(self, val):
        return (val + np.pi) % (2 * np.pi) - np.pi

    def rel_distance(self, entity):
        dist = np.linalg.norm(self._state[0:3] - entity.state[0:3])
        return dist

    def rel_vel(self, entity):
        vel = np.linalg.norm(self._state[3:6] - entity.state[3:6])
        return vel

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

    def step(self, action=np.zeros(2)):
        if self.type == ObsType.M:
            vx = action[0] + np.random.random() * 0.2
            vy = action[1] + np.random.random() * 0.2

            self._state[3] = vx
            self._state[4] = vy

            self._state[0] += vx * self.dt
            self._state[1] += vy * self.dt


class Pad(Entity):
    def __init__(self, _id, x, y, r=0.1):
        self.x = x
        self.y = y
        self.id = _id
        super().__init__(_id=_id, x=x, y=y, z=0, r=r, _type=AgentType.P)


class Target(Entity):
    def __init__(
        self,
        _id,
        x=0,
        y=0,
        z=0,
        psi=0,
        v=0,
        w=0,
        dt=0.1,
        r=0.5,
        num_landing_pads=1,
        pad_offset=0.25,
        pad_r=0.1,
    ):
        super().__init__(_id=_id, x=x, y=y, z=z, r=r, _type=AgentType.T)
        self.id = _id
        self.dt = dt
        self.psi = psi
        self.num_landing_pads = num_landing_pads
        self.r = r  # radius, m
        self.pad_r = pad_r
        self.pad_offset = pad_offset  # m

        self.v = v
        self.w = w
        self.vx = self.v * cos(self.psi)
        self.vy = self.v * sin(self.psi)

        # verifies psi
        if not v == 0:
            np.testing.assert_almost_equal(self.psi, np.arctan2(self.vy, self.vx))
        # x, y, z, x_dot, y_dot, z_dot, psi, psi_dot
        self._state = np.array([x, y, z, self.vx, self.vy, 0, psi, self.w])

        self.pads = [
            Pad(_id, pad_loc[0], pad_loc[1], r=self.pad_r)
            for _id, pad_loc in enumerate(self.get_pad_offsets())
        ]
        self.update_pads_state()

    def get_random_pos(self):
        """generate random points around a sphere
        https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/

        """
        r = self.r * np.sqrt(np.random.random())
        t = np.random.random() * 2 * np.pi

        x = self._state[0] + r * cos(t)
        y = self._state[1] + r * sin(t)
        z = self._state

    def get_pad_offsets(self):
        x = self._state[0]
        y = self._state[1]
        z = self._state[2]
        return [
            (x - self.pad_offset, y, z),
            (x + self.pad_offset, y, z),
            (x, y - self.pad_offset, z),
            (x, y + self.pad_offset, z),
        ]

    def update_pads_state(self):
        pad_offsets = self.get_pad_offsets()
        for pad, offset in zip(self.pads, pad_offsets):
            pad.x, pad.y, pad.z = offset

            pad._state = np.array(
                [
                    pad.x,
                    pad.y,
                    pad.z,
                    self.vx,
                    self.vy,
                    0,
                ]
            )

    # TODO: fix target generating controller
    def get_target_action(self, t, tf):
        """Generate control to get target to follow a trajectory
            https://web2.qatar.cmu.edu/~gdicaro/16311-Fall17/slides/control-theory-for-robotics.pdf
        Args:
            t (_type_): _description_
            tf (_type_): _description_

        Returns:
            _type_: _description_
        """
        e = 2.0 * np.pi / tf
        des_cir_traj = cir_traj(t, e=e, r=1, x_c=2, y_c=2)

        u = np.zeros((2))
        agent_state = np.zeros(3)
        agent_state[:2] = self._state[:2]
        agent_state[2] = self._state[6]
        des_xy = des_cir_traj[:3]
        vd = des_cir_traj[3]
        wd = des_cir_traj[4]
        zeta = 0.7
        b = ((4 / (zeta * 0.8)) ** 2 - wd**2) / (vd**2)
        k1 = k3 = 2.0 * zeta * np.sqrt(wd**2 + b * vd**2)
        k2 = b * np.abs(vd)

        u[0] = vd * np.cos(des_xy[2] - agent_state[2]) + k1 * (
            np.cos(agent_state[2]) * (des_xy[0] - agent_state[0])
            + np.sin(agent_state[2]) * (des_xy[1] - agent_state[1])
        )
        u[1] = (
            wd
            + k2
            * np.sign(vd)
            * (
                np.cos(agent_state[2]) * (des_xy[0] - agent_state[0])
                - np.sin(agent_state[2]) * (des_xy[1] - agent_state[1])
            )
            + k3 * (des_xy[2] - agent_state[2])
        )
        # u[0] = vd
        # u[1] = wd
        # print(f"vd: {vd}, wd: {wd}, b: {b}")

        # u[0] = 0.198 * np.linalg.norm(agent_state[:2] - des_xy[:2])
        # u[1] = 1.2 * (des_xy[2] - agent_state[2])
        return u

    def step(self, action=np.zeros(2)):
        self.v = action[0]
        self.w = action[1]

        self.psi = self._state[6]
        self.vx = self.v * cos(self.psi)
        self.vy = self.v * sin(self.psi)
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.psi += self.w * self.dt

        self.psi = self.wrap_angle(self.psi)

        self._state = np.array(
            [self.x, self.y, self.z, self.vx, self.vy, 0, self.psi, self.w]
        )

        self.update_pads_state()


class UavBase(Entity):
    def __init__(
        self,
        _id,
        x=0,
        y=0,
        z=0,
        r=0.1,
        dt=1 / 10,
        m=0.18,
        l=0.086,
        pad=None,
        d_thresh=0.01,
    ):
        super().__init__(_id, x, y, z, r, _type=AgentType.U)

        # timestep
        self.dt = dt  # s

        # gravity constant
        self.g = 9.81  # m/s^2

        self.m = m

        # lenght of arms
        self.l = l  # m

        if pad is None:
            self.pad = Pad(0, 0, 0)
        else:
            self.pad = pad

        self._state = np.zeros(12)
        self._state[0] = x
        self._state[1] = y
        self._state[2] = z
        self.done = False
        self.landed = False
        self.crashed = False
        self.dt_go = 0
        self.done_dt = 0
        self.done_time = 0
        self.d_thresh = d_thresh
        self.last_rel_dist = 0
        self.max_v = 0.2
        self.min_v = -self.max_v

    def rk4(self, state, action):
        """Based on: https://github.com/mahaitongdae/Safety_Index_Synthesis/blob/master/envs_and_models/collision_avoidance_env.py#L194
        https://www.geeksforgeeks.org/runge-kutta-4th-order-method-solve-differential-equation/

        Args:
            state (_type_): _description_
            action (_type_): _description_
            dt (_type_): _description_

        Returns:
            _type_: _description_
        """
        dot_s1 = self.f_dot(state, action)
        dot_s2 = self.f_dot(state + 0.5 * self.dt * dot_s1, action)
        dot_s3 = self.f_dot(state + 0.5 * self.dt * dot_s2, action)
        dot_s4 = self.f_dot(state + self.dt * dot_s3, action)
        dot_s = (dot_s1 + 2 * dot_s2 + 2 * dot_s3 + dot_s4) / 6.0
        return dot_s

    def f_dot(self, state, action):

        A = np.zeros((12, 12), dtype=np.float32)
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0

        B = np.zeros((12, 3), dtype=np.float32)
        B[3, 0] = 1.0
        B[4, 1] = 1.0
        B[5, 2] = 1.0

        dxdt = A.dot(state) + B.dot(action)

        return dxdt

    def rotation_matrix(self):
        return np.eye(3)

    def step(self, action=np.zeros(3)):
        """Action is propeller forces in body frame

        Args:
            action (_type_, optional): _description_. Defaults to np.zeros(3).
            state:
            x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot
        """

        dot_state = self.rk4(self._state, action)
        self._state = self._state + dot_state * self.dt

        self._state[3:6] = np.clip(self._state[3:6], self.min_v, self.max_v)
        self._state[2] = max(0, self._state[2])

    def check_dest_reached(self, pad=None):
        if pad is None:
            pad = self.pad

        rel_dist = np.linalg.norm(self._state[0:3] - pad.state[0:3])
        rel_vel = np.linalg.norm(self._state[3:6] - pad.state[3:6])
        return rel_dist <= (self.r), rel_dist, rel_vel

    def get_t_go_est(self, rel_vel=None):

        _, rel_dist, _rel_vel = self.check_dest_reached()

        if rel_vel is None:
            rel_vel = _rel_vel

        return rel_dist / (1e-6 + rel_vel)


class Uav(UavBase):
    def __init__(
        self,
        _id,
        x=0,
        y=0,
        z=0,
        phi=0,
        theta=0,
        psi=0,
        r=0.1,
        dt=1 / 10,
        m=0.18,
        l=0.086,
        k=None,
        use_ode=False,
        pad=None,
        d_thresh=0.01,
    ):
        super().__init__(
            _id=_id, x=x, y=y, z=z, r=r, dt=dt, m=m, l=l, pad=pad, d_thresh=d_thresh
        )

        self.use_ode = use_ode
        if self.use_ode:
            self.f_dot_ode = lambda time, state, action: self.f_dot(state, action)
            self.ode = scipy.integrate.ode(self.f_dot_ode).set_integrator(
                "vode", nsteps=500, method="bdf"
            )

        self.inertia = np.array(
            [[0.00025, 0, 2.55e-6], [0, 0.000232, 0], [2.55e-6, 0, 0.0003738]],
            dtype=np.float64,
        )

        ## parameters from: https://upcommons.upc.edu/bitstream/handle/2117/187223/final-thesis.pdf?sequence=1&isAllowed=y
        # self.inertia = np.eye(3)
        # self.inertia[0, 0] = 0.0034  # kg*m^2
        # self.inertia[1, 1] = 0.0034  # kg*m^2
        # self.inertia[2, 2] = 0.006  # kg*m^2
        # self.m = 0.698
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
        self.done = False
        self.landed = False
        self.pad = pad
        self.done_time = None

        # set up gain matrix
        self.kx = self.ky = 1
        self.kz = 1
        self.k_x_dot = self.k_y_dot = 2
        self.k_z_dot = 2
        self.k_phi = 40
        self.k_theta = 30
        self.k_psi = 19
        self.k_phi_dot = self.k_theta_dot = 5
        self.k_psi_dot = 2

        # set up gain matrix
        self.kx = self.ky = 3.5
        self.kz = 7
        self.k_x_dot = self.k_y_dot = 3
        self.k_z_dot = 4.5
        self.k_phi = self.k_theta = 100
        self.k_psi = 50
        self.k_phi_dot = self.k_theta_dot = 15
        self.k_psi_dot = 10

    def get_r_matrix(self, phi, theta, psi):
        """Calculates the Z-Y-X rotation matrix.
           Based on Different Linearization Control Techniques for a Quadrotor System

        Returns: R - 3 x 3 rotation matrix
        """
        cp = cos(phi)
        sp = sin(phi)
        ct = cos(theta)
        st = sin(theta)
        cg = cos(psi)
        sg = sin(psi)
        # R_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        # R_y = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        # R_z = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])

        # R = np.dot(np.dot(R_x, R_y), R_z)
        # ZXY matrix
        R = np.array(
            [
                [cg * ct - sp * sg * st, -cp * sg, cg * st + ct * sp * sg],
                [ct * sg + cg * sp * st, cp * cg, sg * st - cg * ct * sp],
                [-cp * st, sp, cp * ct],
            ]
        )
        # R = np.array(
        #     [
        #         [cg * ct - sp * sg * st, ct * sg + cg * sp * st, -cp * st],
        #         [-cp * sg, cp * cg, sp],
        #         [cg * st + ct * sp * sg, sg * st - cg * ct * sp, cp * ct],
        #     ]
        # )
        # R = R.transpose()
        # R = np.dot(np.dot(R_z, R_x), R_y)
        # R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def get_r_dot_matrix(self, phi, theta, psi):
        """ """
        cp = cos(phi)
        sp = sin(phi)
        ct = cos(theta)
        st = sin(theta)
        cg = cos(psi)
        sg = sin(psi)

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
        # R = np.dot(R_x, np.dot(R_y, R_z))
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
        R = np.array(
            [
                [cg * ct - sp * sg * st, ct * sg + cg * sp * st, -cp * st],
                [-cp * sg, cp * cg, sp],
                [cg * st + ct * sp * sg, sg * st - cg * ct * sp, cp * ct],
            ]
        )
        R = np.dot(np.dot(R_z, R_x), R_y)
        return R

    def f_dot(self, state, action):
        ft, tau_x, tau_y, tau_z = action.reshape(-1).tolist()

        A = np.array(
            [
                [0.25, 0, -0.5 / self.l],
                [0.25, 0.5 / self.l, 0],
                [0.25, 0, 0.5 / self.l],
                [0.25, -0.5 / self.l, 0],
            ]
        )
        prop_thrusts = np.dot(A, np.array([ft, tau_x, tau_y]))
        prop_thrusts_clamped = np.clip(prop_thrusts, self.min_f / 4.0, self.max_f / 4.0)

        B = np.array(
            [
                [1, 1, 1, 1],
                [0, self.l, 0, -self.l],
                [-self.l, 0, self.l, 0],
            ]
        )

        ft = np.dot(B[0, :], prop_thrusts_clamped)
        M = np.dot(B[1:, :], prop_thrusts_clamped)
        tau = np.array([*M, tau_z])

        # TODO: convert angular velocity to angle rates here:
        # state[6:9] = self.wrap_angle(state[6:9])
        phi = state[6]
        theta = state[7]
        psi = state[8]

        omega = state[9:12].copy()
        # tau = np.array([tau_x, tau_y, tau_z])

        omega_dot = np.dot(
            self.inv_inertia, (tau - np.cross(omega, np.dot(self.inertia, omega)))
        )

        # R = self.rotation_matrix()
        # TODO: need to update the rotation matrix here using information from the ODE
        R = self.get_r_matrix(phi, theta, psi)
        acc = (
            np.dot(R, np.array([0, 0, ft], dtype=np.float64).T)
            - np.array([0, 0, self.m * self.g], dtype=np.float64).T
        ) / self.m

        # TODO: troubleshoot why we get small deviations in psi when doing this conversion
        rot_dot = np.dot(np.linalg.inv(self.get_r_dot_matrix(phi, theta, psi)), omega)
        # rot_dot = np.dot(self.get_r_dot_matrix(phi, theta, psi), omega)
        # rot_dot = omega.copy()

        # TODO: fix the x derivative matrix. This matrix doesn't provide angle rates
        x_dot = np.array(
            [
                state[3],
                state[4],
                state[5],
                acc[0],
                acc[1],
                acc[2],
                # TODO: use angle rates here instead
                rot_dot[0],
                rot_dot[1],
                rot_dot[2],
                omega_dot[0],
                omega_dot[1],
                omega_dot[2],
            ]
        )

        return x_dot

    def calc_des_action(self, des_pos):
        pos_er = des_pos[0:12] - self._state
        r_ddot_1 = des_pos[12]
        r_ddot_2 = des_pos[13]
        r_ddot_3 = des_pos[14]

        action = np.zeros(3, dtype=np.float32)
        # https://upcommons.upc.edu/bitstream/handle/2117/112404/Thesis-Jesus_Valle.pdf?sequence=1&isAllowed=y
        action[0] = self.kx * pos_er[0] + self.k_x_dot * pos_er[3] + r_ddot_1
        action[1] = self.ky * pos_er[1] + self.k_y_dot * pos_er[4] + r_ddot_2
        action[2] = self.kz * pos_er[2] + self.k_z_dot * pos_er[5] + r_ddot_3

        des_psi = des_pos[8]
        des_psi_dot = des_pos[11]

        des_action = self.get_torque_from_acc(action, des_psi, des_psi_dot)

        return des_action

    def get_torque_from_acc(self, action, des_psi=0, des_psi_dot=0):
        u1 = self.m * (self.g + action[2])

        # desired angles
        phi_des = (action[0] * sin(des_psi) - action[1] * cos(des_psi)) / self.g
        theta_des = (action[0] * cos(des_psi) + action[1] * sin(des_psi)) / self.g

        # desired torques
        u2_phi = self.k_phi * (phi_des - self._state[6]) + self.k_phi_dot * (
            -self._state[9]
        )
        u2_theta = self.k_theta * (theta_des - self._state[7]) + self.k_theta_dot * (
            -self._state[10]
        )

        # yaw
        u2_psi = self.k_psi * (des_psi - self._state[8]) + self.k_psi_dot * (
            des_psi_dot - self._state[11]
        )

        M = np.dot(self.inertia, np.array([u2_phi, u2_theta, u2_psi]))
        action = np.array([u1, *M])
        return action

    def step(self, action=np.zeros(4)):
        """Action is propeller forces in body frame

        Args:
            action (_type_, optional): _description_. Defaults to np.zeros(4).
            state:
            x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot
        """

        if len(action) == 3:
            action = self.get_torque_from_acc(action)

        if self.use_ode:
            state = self._state.copy()
            self.ode.set_initial_value(state, 0).set_f_params(action)
            self._state = self.ode.integrate(self.ode.t + self.dt)
        else:
            dot_state = self.rk4(self._state, action)
            self._state = self._state + dot_state * self.dt

        self._state[9:12] = self.wrap_angle(self._state[9:12])

        self._state[6:9] = self.wrap_angle(self._state[6:9])
        self._state[2] = max(0, self._state[2])
