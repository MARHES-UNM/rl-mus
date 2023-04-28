import numpy as np
from math import cos, sin
import matplotlib

# matplotlib.use("Qt5Agg")
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

# R2D = 57.295779513
# D2R = 0.017453293

from gym.utils import seeding


class Quadrotor:
    def __init__(
        self, x=0, y=0, z=0, phi=0, theta=0, psi=0, dt=1 / 50, m=0.18, l=0.086
    ):

        # l = 1
        # timestep
        self.dt = dt  # s

        # gravity constant
        self.g = 9.81  # m/s^2

        # mass
        self.m = m  # kg

        # lenght of arms
        self.l = 1  # m

        self.inertia = np.array(
            [[0.00025, 0, 2.55e-6], [0, 0.000232, 0], [2.55e-6, 0, 0.0003738]]
        )

        self.inertia = np.eye(3) * 1

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
        c_phi = cos(self._state[6])
        s_phi = sin(self._state[6])
        c_theta = cos(self._state[7])
        s_theta = sin(self._state[7])
        c_psi = cos(self._state[8])
        s_psi = sin(self._state[8])

        return np.array(
            [
                [
                    c_psi * c_theta - s_phi * s_psi * s_theta,
                    -c_phi * s_psi,
                    c_psi * s_theta + c_theta * s_phi * s_psi,
                ],
                [
                    c_theta * s_psi + c_psi * s_phi * s_theta,
                    c_phi * c_psi,
                    s_psi * s_theta - c_psi * c_theta * s_phi,
                ],
                [-c_phi * s_theta, s_phi, c_phi * c_theta],
            ]
        )

    def state_dot(self, time, state, action=np.zeros(4)):
        action = np.clip(action, self.min_f, self.max_f)
        state_dot = np.zeros(12)

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

        # phi_rot = np.eye(3)
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

    def step(self, action=np.zeros(4)):
        """Action is propeller forces in body frame

        Args:
            action (_type_, optional): _description_. Defaults to np.zeros(4).
            state:
            x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot
        """

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

        phi_rot = np.eye(3)
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

        self._state[3:6] += x_ddot * self.dt
        self._state[0:3] += self._state[3:6] * self.dt
        if self._state[2] < 0:
            self._state[2] = 0

        omega += omega_dot * self.dt
        # self._state[9:12] += np.dot(np.linalg.inv(phi_rot), omega)
        self._state[9:12] += omega
        self._state[6:9] += self._state[9:12] * self.dt
        self._state[6:9] = self.wrap_angle(self._state[6:9])

    def wrap_angle(self, val):
        return (val + np.pi) % (2 * np.pi) - np.pi

    def transformation_matrix(self):

        x = self._state[0]
        y = self._state[1]
        z = self._state[2]
        roll = self._state[6]
        pitch = self._state[7]
        yaw = self._state[8]
        return np.array(
            [
                [
                    cos(yaw) * cos(pitch),
                    -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll),
                    sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll),
                    x,
                ],
                [
                    sin(yaw) * cos(pitch),
                    cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) * sin(roll),
                    -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll),
                    y,
                ],
                [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw), z],
            ]
        )


class UavSim:
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, env_config={}):
        self.dt = env_config.get("dt", 0.01)
        self._seed = env_config.get("seed", None)
        self.render_mode = env_config.get("render_mode", "human")
        self.num_uavs = env_config.get("num_uavs", 4)
        self.fig = None
        self.clock = None
        self._time_elapsed = 0

        self.reset()

    @property
    def time_elapsed(self):
        return self._time_elapsed

    def step(self, actions):
        action = np.random.rand(4) * (self.uav.max_f - self.uav.min_f) + self.uav.min_f
        action = np.ones(4) * self.uav.m * self.uav.g / 4

        self.uav.step(action)

        self._time_elapsed += self.dt

    def seed(self, seed=None):
        """Random value to seed"""
        np.random.seed(seed)

        seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        self.seed(seed)

        self.uav = Quadrotor(2, 4, 5)

    # TODO: update this to use blit
    # https://matplotlib.org/stable/api/animation_api.html
    # https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
    def render(self, mode="human"):
        if self.render_mode == "human":
            if self.fig is None:
                self.fig = plt.figure()

                # for stopping simulation with the esc key
                self.fig.canvas.mpl_connect(
                    "key_release_event",
                    lambda event: [exit(0) if event.key == "escape" else None],
                )

                self.ax = self.fig.add_subplot(111, projection="3d")

                x_axis = np.arange(-2, 3)
                y_axis = np.arange(-2, 3)
                z_axis = np.arange(-2, 3)

                self.ax.plot([0, 0], [0, 0], [0, 0], "k+")
                self.ax.plot(x_axis, np.zeros(5), np.zeros(5), "r--", linewidth=0.5)
                self.ax.plot(np.zeros(5), y_axis, np.zeros(5), "g--", linewidth=0.5)
                self.ax.plot(np.zeros(5), np.zeros(5), z_axis, "b--", linewidth=0.5)

                self.ax.set_xlim([-5, 5])
                self.ax.set_ylim([-5, 5])
                self.ax.set_zlim([0, 10])

                self.ax.set_xlabel("X-axis (in meters)")
                self.ax.set_ylabel("Y-axis (in meters)")
                self.ax.set_zlabel("Z-axis (in meters)")

                self.ax.set_title("Multi-UAV Simulation")

                # (0, 0) is bottom left, (1, 1) is top right
                # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.
                self.time_display = self.ax.text2D(
                    0.75, 0.95, "red", color="red", transform=self.ax.transAxes
                )

                self.state_display = self.ax.text2D(
                    0, 0.95, "green", color="green", transform=self.ax.transAxes
                )

                (self.l1,) = self.ax.plot(
                    [], [], [], color="blue", linewidth=1, antialiased=False
                )
                (self.l2,) = self.ax.plot(
                    [], [], [], color="red", linewidth=1, antialiased=False
                )

                R = self.uav.rotation_matrix()
                l = self.uav.l

                self.points = np.array([[-l, 0, 0], [l, 0, 0], [0, -l, 0], [0, l, 0]]).T
                self.points = np.dot(R, self.points)

                self.points[0, :] += self.uav._state[0]
                self.points[1, :] += self.uav._state[1]
                self.points[2, :] += self.uav._state[2]

                self.l1.set_data(self.points[0, 0:2], self.points[1, 0:2])
                self.l1.set_3d_properties(self.points[2, 0:2])
                self.l2.set_data(self.points[0, 2:4], self.points[1, 2:4])
                self.l2.set_3d_properties(self.points[2, 2:4])

                # TODO: See if method below can improve plotting speed
                # https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
                # self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

            self.time_display.set_text(f"Simulation time = {self.time_elapsed:.2f} s")
            uav_state = self.uav.state
            self.state_display.set_text(
                f"x:{uav_state[0]:.2f}, y:{uav_state[1]:.2f}, z:{uav_state[2]:.2f}\n"
                f"phi: {uav_state[6]:.2f}, theta: {uav_state[7]:.2f}, psi: {uav_state[8]:.2f}"
            )

            R = self.uav.rotation_matrix()
            l = self.uav.l

            points = np.array([[-l, 0, 0], [l, 0, 0], [0, -l, 0], [0, l, 0]]).T
            points = np.dot(R, points)

            points[0, :] += self.uav._state[0]
            points[1, :] += self.uav._state[1]
            points[2, :] += self.uav._state[2]

            self.l1.set_data(points[0, 0:2], points[1, 0:2])
            self.l1.set_3d_properties(points[2, 0:2])
            self.l2.set_data(points[0, 2:4], points[1, 2:4])
            self.l2.set_3d_properties(points[2, 2:4])
            plt.pause(0.0000000001)
