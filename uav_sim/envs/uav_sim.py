from matplotlib import transforms
import numpy as np
from math import cos, sin
import matplotlib

# matplotlib.use("Qt5Agg")
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

R2D = 57.295779513
D2R = 0.017453293

from gym.utils import seeding


class Quadrotor:
    def __init__(self, x=0, y=0, z=0, phi=0, theta=0, psi=0, dt=1 / 50, m=1.4, l=0.56):

        self.p1 = np.array([l / 2, 0, 0, 1]).T
        self.p2 = np.array([-l / 2, 0, 0, 1]).T
        self.p3 = np.array([0, l / 2, 0, 1]).T
        self.p4 = np.array([0, -l / 2, 0, 1]).T

        self.x = x
        self.y = y
        self.z = z
        self.x_dot = 0
        self.y_dot = 0
        self.z_dot = 0

        # roll
        self.phi = phi
        # pitch
        self.theta = theta
        # yaw
        self.psi = psi

        # timestep
        self.dt = dt
        # gravity constant
        self.g = 9.81

        # mass
        self.m = m

        # lenght of arms
        self.l = l

        self.phi_dot = 0
        self.theta_dot = 0
        self.psi_dot = 0

    @property
    def state(self):
        return np.array(
            [
                self.x,
                self.y,
                self.z,
                self.x_dot,
                self.y_dot,
                self.z_dot,
                self.phi,
                self.theta,
                self.psi,
                self.phi_dot,
                self.theta_dot,
                self.psi_dot,
            ]
        )

    # def rot_matrix(self, ):
    #     ct = cos(theta)
    #     st = sin(theta)
    #     cf = cos(phi)
    #     sf = sin(phi)
    #     cp = cos(psi)
    #     sp = sin(psi)

    #     return np.array([
    #         [cp * ct, sp * ct, -st],
    #         []
    #     ])

    def step(self, action=np.zeros(4)):
        pass

    def transformation_matrix(self):
        # x = self.x
        # y = self.y
        # z = self.z
        # phi = self.phi
        # theta = self.theta
        # psi = self.psi

        # return np.array([cos(psi) * cos(theta), -sin()])

        x = self.x
        y = self.y
        z = self.z
        roll = self.phi
        pitch = self.theta
        yaw = self.psi
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
                (pointCM,) = self.ax.plot([0], [0], [0], "b.")
                (pointBLDC1,) = self.ax.plot([0], [0], [0], "b.")
                (pointBLDC2,) = self.ax.plot([0], [0], [0], "b.")
                (pointBLDC3,) = self.ax.plot([0], [0], [0], "b.")
                (pointBLDC4,) = self.ax.plot([0], [0], [0], "b.")
                (line1,) = self.ax.plot([0, 0], [0, 0], [0, 0], "b.")
                (line2,) = self.ax.plot([0, 0], [0, 0], [0, 0], "b.")

                self.ax.plot([0, 0], [0, 0], [0, 0], "k+")
                self.ax.plot(x_axis, np.zeros(5), np.zeros(5), "r--", linewidth=0.5)
                self.ax.plot(np.zeros(5), y_axis, np.zeros(5), "g--", linewidth=0.5)
                self.ax.plot(np.zeros(5), np.zeros(5), z_axis, "b--", linewidth=0.5)

                self.ax.set_xlim([-5, 5])
                self.ax.set_ylim([-5, 5])
                self.ax.set_zlim([-5, 5])

                self.ax.set_xlabel("X-axis (in meters)")
                self.ax.set_ylabel("Y-axis (in meters)")
                self.ax.set_zlabel("Z-axis (in meters)")

                # (0, 0) is bottom left, (1, 1) is top right
                # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.
                self.time_display = self.ax.text2D(
                    0.75, 0.95, "red", color="red", transform=self.ax.transAxes
                )

                self.state_display = self.ax.text2D(
                    0, 0.95, "green", color="green", transform=self.ax.transAxes
                )


                T = self.uav.transformation_matrix()

                p1_t = np.matmul(T, self.uav.p1)
                p2_t = np.matmul(T, self.uav.p2)
                p3_t = np.matmul(T, self.uav.p3)
                p4_t = np.matmul(T, self.uav.p4)

                plt.cla()

                self.props = self.ax.plot(
                    [p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                    [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                    [p1_t[2], p2_t[2], p3_t[2], p4_t[2]],
                    "k.",
                )

                self.line1 = self.ax.plot(
                    [p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]], [p1_t[2], p2_t[2]], "r-"
                )

                self.line2 = self.ax.plot(
                    [p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]], [p3_t[2], p4_t[2]], "r-"
                )
                
            self.time_display.set_text(f"Simulation time = {self.time_elapsed:.2f} s")

            T = self.uav.transformation_matrix()

            p1_t = np.matmul(T, self.uav.p1)
            p2_t = np.matmul(T, self.uav.p2)
            p3_t = np.matmul(T, self.uav.p3)
            p4_t = np.matmul(T, self.uav.p4)

            # self.props.remove()
            # self.line1.remove()
            # self.line2.remote()
            # plt.cla()

            self.props = self.ax.plot(
                [p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                [p1_t[2], p2_t[2], p3_t[2], p4_t[2]],
                "k.",
            )

            self.line1 = self.ax.plot(
                [p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]], [p1_t[2], p2_t[2]], "r-"
            )

            self.line2 = self.ax.plot(
                [p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]], [p3_t[2], p4_t[2]], "r-"
            )

            self.time_display.set_text(f"Simulation time = {self.time_elapsed:.2f} s")

            self.ax.set_xlim([-5, 5])
            self.ax.set_ylim([-5, 5])
            self.ax.set_zlim([0, 10])

            # self.ax.plot(self.x_data, self.y_data, self.z_data, 'b:')
            plt.pause(0.0001)
