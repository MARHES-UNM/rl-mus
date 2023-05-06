import sys
from turtle import color
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import mpl_toolkits.mplot3d.art3d as art3d


class Sprite:
    def __init__(self, ax, t_lim=30):
        self.t_lim = t_lim
        self.ax = ax

    def update(self, t):
        raise NotImplemented


class TargetSprite:
    """
    Creates circular patch on 3d surface
    https://matplotlib.org/stable/gallery/mplot3d/pathpatch3d.html

    https://nickcharlton.net/posts/drawing-animating-shapes-matplotlib
    https://stackoverflow.com/questions/73892040/moving-circle-animation-3d-plot
    """

    def __init__(self, ax, target=None, num_targets=1, t_lim=30):
        self.ax = ax
        self.target = target
        self.t_lim = t_lim
        self.body = None
        self.pads = [None] * 4
        # self.body = Circle((0, 0), self.target.r)
        # self.body.center = (0, 0)
        # self.ax["ax_3d"].add_patch(self.body)
        # art3d.pathpatch_2d_to_3d(self.body, z=0, zdir="z")

        self.trajectory = {
            "t": [],
            "x": [],
            "y": [],
            "psi": [],
        }

        (self.x_bar,) = self.ax["ax_error_x"].plot(
            [], [], label=f"id: {self.target.id}"
        )
        (self.y_bar,) = self.ax["ax_error_y"].plot(
            [], [], label=f"id: {self.target.id}"
        )
        (self.psi_bar,) = self.ax["ax_error_psi"].plot(
            [], [], label=f"id: {self.target.id}"
        )

    def update(self, t):
        # self.cm.set_data(self.target.state[0], self.target.state[1])
        # self.body.verts = ()
        # self.ax["ax_3d"].patches.pop()

        if self.body:
            self.body.remove()

        self.body = Circle(
            (self.target.state[0], self.target.state[1]),
            self.target.r,
            fill=False,
            color="green",
        )
        self.ax["ax_3d"].add_patch(self.body)
        art3d.pathpatch_2d_to_3d(self.body, z=0, zdir="z")

        for idx, pad in enumerate(self.pads):
            if pad is not None:
                pad.remove()

            pad = Circle(
                (self.target.pads[idx].x, self.target.pads[idx].y),
                0.25,
                fill=False,
                color="red",
            )
            self.ax["ax_3d"].add_patch(pad)
            art3d.pathpatch_2d_to_3d(pad, z=0, zdir="z")

        # self.body.set_3d_properties(
        #     (self.target.state[0], self.target.state[1]), zs=0, zdir="z"
        # )
        # self.body.verts = (self.target.state[0], self.target.state[1])

        # self.cm.set_offsets(se)
        # self.cm.set_3d_properties([0])
        #         target_points = np.array(
        #     [[0.5, 0.5, 0], [0.5, 1.5, 0], [1.5, 0.5, 0], [1.5, 1.5, 0]]
        # ).T
        # for idx, target in enumerate(self.target_sprites.targets):
        #     target.set_data(
        #         target_points[0, idx : idx + 1], target_points[1, idx : idx + 1]
        #     )
        #     target.set_3d_properties(target_points[2, idx : idx + 1])

        self.trajectory["t"].append(t)
        self.trajectory["x"].append(self.target._state[0])
        self.trajectory["y"].append(self.target._state[1])
        self.trajectory["psi"].append(self.target._state[2])

        self.ax["ax_error_x"].set_xlim(
            left=max(0, t - self.t_lim), right=t + self.t_lim
        )
        self.x_bar.set_data(self.trajectory["t"], self.trajectory["x"])
        self.ax["ax_error_y"].set_xlim(
            left=max(0, t - self.t_lim), right=t + self.t_lim
        )
        self.y_bar.set_data(self.trajectory["t"], self.trajectory["y"])
        self.ax["ax_error_psi"].set_xlim(
            left=max(0, t - self.t_lim), right=t + self.t_lim
        )
        self.psi_bar.set_data(self.trajectory["t"], self.trajectory["psi"])


class UavSprite:
    def __init__(self, ax, uav=None):
        self.ax = ax
        self.uav = uav

        (self.l1,) = self.ax["ax_3d"].plot(
            [], [], [], color="blue", linewidth=1, antialiased=False
        )
        (self.l2,) = self.ax["ax_3d"].plot(
            [], [], [], color="blue", linewidth=1, antialiased=False
        )

        (self.cm,) = self.ax["ax_3d"].plot([], [], [], "k.")

        # TODO: plot rotors
        # (self.rotor1,) = self.ax["ax_3d"].plot([], [], [], "k.")
        # (self.rotor2,) = self.ax["ax_3d"].plot([], [], [], "k.")
        # (self.rotor3,) = self.ax["ax_3d"].plot([], [], [], "k.")
        # (self.rotor4,) = self.ax["ax_3d"].plot([], [], [], "k.")

        self.trajectory = {
            "t": [],
            "x": [],
            "y": [],
            "z": [],
            "psi": [],
        }

        (self.x_bar,) = self.ax["ax_error_x"].plot([], [], label=f"id: {self.uav.id}")
        (self.y_bar,) = self.ax["ax_error_y"].plot([], [], label=f"id: {self.uav.id}")
        (self.z_bar,) = self.ax["ax_error_z"].plot([], [], label=f"id: {self.uav.id}")
        (self.psi_bar,) = self.ax["ax_error_psi"].plot(
            [], [], label=f"id: {self.uav.id}"
        )

        l = self.uav.l
        self.points = np.array(
            [
                [-l, 0, 0],
                [l, 0, 0],
                [0, -l, 0],
                [0, l, 0],
                [0, 0, 0],  # cm
                [-l, 0, 0],
                [l, 0, 0],
                [0, -l, 0],
                [0, l, 0],
            ]
        ).T

        self.t_lim = 30

    def update(self, t):
        R = self.uav.rotation_matrix()

        body = np.dot(R, self.points)
        body[0, :] += self.uav._state[0]
        body[1, :] += self.uav._state[1]
        body[2, :] += self.uav._state[2]

        self.l1.set_data(body[0, 0:2], body[1, 0:2])
        self.l1.set_3d_properties(body[2, 0:2])
        self.l2.set_data(body[0, 2:4], body[1, 2:4])
        self.l2.set_3d_properties(body[2, 2:4])
        self.cm.set_data(body[0, 4:5], body[1, 4:5])
        self.cm.set_3d_properties(body[2, 4:5])

        self.trajectory["t"].append(t)
        self.trajectory["x"].append(self.uav._state[0])
        self.trajectory["y"].append(self.uav._state[1])
        self.trajectory["z"].append(self.uav._state[2])
        self.trajectory["psi"].append(self.uav._state[8])

        self.ax["ax_error_x"].set_xlim(
            left=max(0, t - self.t_lim), right=t + self.t_lim
        )
        self.x_bar.set_data(self.trajectory["t"], self.trajectory["x"])
        self.ax["ax_error_y"].set_xlim(
            left=max(0, t - self.t_lim), right=t + self.t_lim
        )
        self.y_bar.set_data(self.trajectory["t"], self.trajectory["y"])
        self.ax["ax_error_z"].set_xlim(
            left=max(0, t - self.t_lim), right=t + self.t_lim
        )
        self.z_bar.set_data(self.trajectory["t"], self.trajectory["z"])
        self.ax["ax_error_psi"].set_xlim(
            left=max(0, t - self.t_lim), right=t + self.t_lim
        )
        self.psi_bar.set_data(self.trajectory["t"], self.trajectory["psi"])


class Gui:
    def __init__(self, uavs=[], target=None, max_x=3, max_y=3, max_z=3, fig=None):
        self.uavs = uavs
        self.fig = fig
        self.target = target

        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 6))
            gs0 = self.fig.add_gridspec(1, 2)
            gs00 = gs0[0].subgridspec(1, 1)
            gs01 = gs0[1].subgridspec(4, 1)
            self.ax = {}
            self.ax["ax_3d"] = self.fig.add_subplot(gs00[0], projection="3d")
            self.ax["ax_error_x"] = self.fig.add_subplot(gs01[0])
            self.ax["ax_error_x"].set_ylim([0, max_x])
            self.ax["ax_error_y"] = self.fig.add_subplot(gs01[1])
            self.ax["ax_error_y"].set_ylim([0, max_y])
            self.ax["ax_error_z"] = self.fig.add_subplot(gs01[2])
            self.ax["ax_error_z"].set_ylim([0, max_z])
            self.ax["ax_error_psi"] = self.fig.add_subplot(gs01[3])
            self.ax["ax_error_psi"].set_ylim([-np.pi, np.pi])

        self.ax["ax_3d"].set_xlim3d([0, max_x])
        self.ax["ax_3d"].set_ylim3d([0, max_y])
        self.ax["ax_3d"].set_zlim3d([0, max_z])

        self.ax["ax_3d"].set_xlabel("X (m)")
        self.ax["ax_3d"].set_ylabel("Y (m)")
        self.ax["ax_3d"].set_zlabel("Z (m)")

        self.ax["ax_3d"].set_title("Multi-UAV Simulation")

        # add axis
        n_points = 5
        x_axis = np.linspace(0, max_x, n_points)
        y_axis = np.linspace(0, max_y, n_points)
        z_axis = np.linspace(0, max_z, n_points)

        self.ax["ax_3d"].plot([0, 0], [0, 0], [0, 0], "k+")
        self.ax["ax_3d"].plot(
            x_axis, np.zeros(n_points), np.zeros(n_points), "r--", linewidth=0.5
        )
        self.ax["ax_3d"].plot(
            np.zeros(n_points), y_axis, np.zeros(n_points), "g--", linewidth=0.5
        )
        self.ax["ax_3d"].plot(
            np.zeros(n_points), np.zeros(n_points), z_axis, "b--", linewidth=0.5
        )

        # (0, 0) is bottom left, (1, 1) is top right
        # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.
        self.time_display = self.ax["ax_3d"].text2D(
            0.0, 0.95, "", color="red", transform=self.ax["ax_3d"].transAxes
        )
        self.init_entities()
        # for stopping simulation with the esc key
        self.fig.canvas.mpl_connect("key_press_event", self.keypress_routine)

        # # plt.show(False)
        # plt.draw()

        # self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def init_entities(self):
        self.sprites = []
        target_sprite = TargetSprite(self.ax, self.target)
        self.sprites.append(target_sprite)

        for uav in self.uavs:
            uav_sprite = UavSprite(self.ax, uav)
            self.sprites.append(uav_sprite)

    # TODO: update this to use blit
    # https://matplotlib.org/stable/api/animation_api.html
    # https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
    def update(self, time_elapsed):
        # self.fig.canvas.restore_region(self.background)
        self.time_display.set_text(f"Sim time = {time_elapsed:.2f} s")

        for sprite in self.sprites:
            sprite.update(time_elapsed)

        # self.ax["ax_3d"].plot(1, 1, "r+")

        #     self.ax.draw_artist(uav_sprite.cm)
        # self.fig.canvas.blit(self.ax.bbox)
        # for key, ax in self.ax.items():
        #     if key == "ax_3d":
        #         continue
        #     ax.legend()

        plt.pause(0.0000000000001)

    def keypress_routine(self, event):
        sys.stdout.flush()
        if event.key == "x":
            y = list(self.ax["ax_3d"].get_ylim3d())
            y[0] += 0.2
            y[1] += 0.2
            self.ax["ax_3d"].set_ylim3d(y)
        elif event.key == "w":
            y = list(self.ax["ax_3d"].get_ylim3d())
            y[0] -= 0.2
            y[1] -= 0.2
            self.ax["ax_3d"].set_ylim3d(y)
        elif event.key == "d":
            x = list(self.ax["ax_3d"].get_xlim3d())
            x[0] += 0.2
            x[1] += 0.2
            self.ax["ax_3d"].set_xlim3d(x)
        elif event.key == "a":
            x = list(self.ax["ax_3d"].get_xlim3d())
            x[0] -= 0.2
            x[1] -= 0.2
            self.ax["ax_3d"].set_xlim3d(x)

        elif event.key == "escape":
            exit(0)

    def __del__(self):
        plt.close(self.fig)

    def close(self):
        plt.close(self.fig)
        self.fig = None
