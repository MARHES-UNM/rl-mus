from math import cos, sin
import numpy as np
import scipy.integrate

class Quadrotor:
    def __init__(
        self,
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

        self.inertia = np.eye(3) * 0.0003738

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
                        [self.gamma, self.gamma, self.gamma, self.gamma],
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