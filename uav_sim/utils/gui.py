import sys
import matplotlib

#matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import mpl_toolkits.mplot3d.art3d as art3d


class Sprite:
    def __init__(self, ax, t_lim=30):
        self.t_lim = t_lim
        self.ax = ax

    def update(self, t, done=False):
        raise NotImplementedError("update method not implemented")

    def get_sphere(self, center, radius, color="r", alpha=0.1):
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        return self.ax["ax_3d"].plot_wireframe(x, y, z, color=color, alpha=alpha)

    def get_cube(self, vertex, l=1, w=1, h=1, alpha=0.1, color="r"):
        x1, y1, z1 = vertex[0], vertex[1], vertex[2]
        x2, y2, z2 = x1 + l, y1 + w, z1 + h
        ax = self.ax["ax_3d"]
        xs, ys = np.meshgrid([x1, x2], [y1, y2])
        zs = np.ones_like(xs)
        # plot bottom and top surfaces
        body = []
        body.append(ax.plot_wireframe(xs, ys, zs * z1, alpha=alpha, color=color))

        body.append(ax.plot_wireframe(xs, ys, zs * z2, alpha=alpha, color=color))

        # plot left and right side
        # xs, zs = np.meshgrid([x1, x2], z)
        # ys = np.ones_like(xs)
        # ax.plot_wireframe(xs, ys * y1, zs, alpha=alpha, color=color)
        # ax.plot_wireframe(xs, ys * y2, zs, alpha=alpha, color=color)

        ys, zs = np.meshgrid([y1, y2], [z1, z2])
        xs = np.ones_like(ys)
        body.append(ax.plot_wireframe(xs * x1, ys, zs, alpha=alpha, color=color))
        body.append(ax.plot_wireframe(xs * x2, ys, zs, alpha=alpha, color=color))
        return body


class SphereSprite(Sprite):
    def __init__(self, ax, color="r", alpha=0.1, t_lim=30):
        self.color = color
        self.alpha = alpha
        self.body = None

        super().__init__(ax, t_lim)

    def update(self, t, done=False):
        if self.body:
            if isinstance(self.body, list):
                for body in self.body:
                    body.remove()
            else:
                self.body.remove()

        center = self.obstacle._state[0:3]
        radius = self.obstacle.r

        self.body = self.get_sphere(center, radius)

    def get_sphere(self, center, radius, color="r", alpha=0.1):
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        return self.ax["ax_3d"].plot_wireframe(x, y, z, color=color, alpha=alpha)


class ObstacleSprite(Sprite):
    def __init__(self, ax, obstacle, t_lim=30):
        super().__init__(ax, t_lim)
        self.obstacle = obstacle
        self.body = None
        # self.body = self.ax["ax_3d"].scatter(
        #     [], [], [], marker="o", color="r", s=100 * 4 * 0.5**2
        # )

    def update(self, t, done=False):
        # xa = [self.obstacle._state[0]]
        # ya = [self.obstacle._state[1]]
        # z = [self.obstacle._state[2]]
        # self.body._offsets3d = (xa, ya, z)
        if self.body:
            if isinstance(self.body, list):
                for body in self.body:
                    body.remove()
            else:
                self.body.remove()

        center = self.obstacle._state[0:3]
        radius = self.obstacle.r

        self.body = self.get_sphere(center, radius)


class ObstacleSpriteCube(Sprite):
    def __init__(self, ax, obstacle, t_lim=30):
        super().__init__(ax, t_lim)
        self.obstacle = obstacle
        self.body = None

    def update(self, t, done=False):
        if self.body:
            for body in self.body:
                body.remove()

        vertex = self.obstacle._state[0:3]

        self.body = self.get_cube(vertex)


class TargetSprite:
    """
    Creates circular patch on 3d surface
    https://matplotlib.org/stable/gallery/mplot3d/pathpatch3d.html

    https://nickcharlton.net/posts/drawing-animating-shapes-matplotlib
    https://stackoverflow.com/questions/73892040/moving-circle-animation-3d-plot
    """

    def __init__(self, ax, target=None, num_targets=1, t_lim=10, color="g"):
        self.ax = ax
        self.target = target
        self.t_lim = t_lim
        self.body = None
        self.pad_sprites = [None] * 4
        self.color = color

        self.trajectory = {
            "t": [],
            "x": [],
            "y": [],
            "z": [],
            "psi": [],
        }

        (self.x_bar,) = self.ax["ax_error_x"].plot(
            [], [], c=color, label=f"{self.target.type}: {self.target.id}"
        )
        (self.y_bar,) = self.ax["ax_error_y"].plot(
            [], [], c=color, label=f"{self.target.type}: {self.target.id}"
        )
        (self.z_bar,) = self.ax["ax_error_z"].plot(
            [], [], c=color, label=f"{self.target.type}: {self.target.id}"
        )
        (self.psi_bar,) = self.ax["ax_error_psi"].plot(
            [], [], c=color, label=f"{self.target.type}: {self.target.id}"
        )

        self.x_lim = list(self.ax["ax_3d"].get_xlim3d())
        self.y_lim = list(self.ax["ax_3d"].get_ylim3d())

    def update(self, t, done=False):
        if self.body:
            self.body.remove()

        self.body = Circle(
            (self.target.state[0], self.target.state[1]),
            self.target.r,
            fill=False,
            color=self.color,
        )
        self.ax["ax_3d"].add_patch(self.body)
        art3d.pathpatch_2d_to_3d(self.body, z=self.target.state[2], zdir="z")

        self.trajectory["t"].append(t)
        self.trajectory["x"].append(self.target._state[0])
        self.trajectory["y"].append(self.target._state[1])
        self.trajectory["z"].append(self.target._state[2])
        self.trajectory["psi"].append(self.target._state[6])

        # self.update_axis()

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

    def update_axis(self):
        target_x = self.target.state[0]
        target_y = self.target.state[1]

        self.ax["ax_3d"].set_xlim3d(
            [target_x - self.x_lim[1] / 2, target_x + self.x_lim[1] / 2]
        )
        self.ax["ax_3d"].set_ylim3d(
            [target_y - self.y_lim[1] / 2, target_y + self.y_lim[1] / 2]
        )


class UavSprite:
    def __init__(self, ax, uav=None, color=None, t_lim=10):
        self.ax = ax
        self.t_lim = t_lim
        self.uav = uav
        self.pad_sprite = None
        self.color = color

        (self.l1,) = self.ax["ax_3d"].plot(
            [], [], [], color=color, linewidth=1, antialiased=False
        )
        (self.l2,) = self.ax["ax_3d"].plot(
            [], [], [], color=color, linewidth=1, antialiased=False
        )

        (self.cm,) = self.ax["ax_3d"].plot([], [], [], color=color, marker=".")
        (self.traj,) = self.ax["ax_3d"].plot([], [], [], color=color, marker=".")

        self.trajectory = {
            "t": [],
            "x": [],
            "y": [],
            "z": [],
            "psi": [],
        }

        (self.x_bar,) = self.ax["ax_error_x"].plot(
            [], [], color=color, label=f"id: {self.uav.id}"
        )
        (self.y_bar,) = self.ax["ax_error_y"].plot(
            [], [], color=color, label=f"id: {self.uav.id}"
        )
        (self.z_bar,) = self.ax["ax_error_z"].plot(
            [], [], color=color, label=f"id: {self.uav.id}"
        )
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

    def update(self, t, done=False):
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

        if self.pad_sprite:
            self.pad_sprite.remove()

        self.pad_sprite = Circle(
            (self.uav.pad.x, self.uav.pad.y),
            self.uav.pad.r,
            fill=False,
            color=self.color,
        )
        self.ax["ax_3d"].add_patch(self.pad_sprite)
        art3d.pathpatch_2d_to_3d(self.pad_sprite, z=self.uav.pad.z, zdir="z")

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

        if done:
            self.traj.set_data(self.trajectory["x"], self.trajectory["y"])
            self.traj.set_3d_properties(self.trajectory["z"])


class Gui:
    def __init__(
        self, uavs=[], target=None, obstacles=[], max_x=3, max_y=3, max_z=3, fig=None
    ):
        self.uavs = uavs
        self.fig = fig
        self.target = target
        self.obstacles = obstacles
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z
        self.cmap = plt.get_cmap("tab10")

        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            gs0 = self.fig.add_gridspec(1, 2)
            gs00 = gs0[0].subgridspec(8, 1)
            gs01 = gs0[1].subgridspec(4, 1)
            self.ax = {}
            self.ax["legend"] = self.fig.add_subplot(gs00[0, :])
            self.ax["ax_3d"] = self.fig.add_subplot(gs00[1:, :], projection="3d")
            self.ax["ax_error_x"] = self.fig.add_subplot(gs01[0])
            self.ax["ax_error_x"].set_ylim([-max_x, max_x])
            self.ax["ax_error_x"].set_ylabel("x (m)")
            self.ax["ax_error_y"] = self.fig.add_subplot(gs01[1])
            self.ax["ax_error_y"].set_ylim([-max_y, max_y])
            self.ax["ax_error_y"].set_ylabel("y (m)")
            self.ax["ax_error_z"] = self.fig.add_subplot(gs01[2])
            self.ax["ax_error_z"].set_ylim([0, max_z])
            self.ax["ax_error_z"].set_ylabel("z (m)")
            self.ax["ax_error_psi"] = self.fig.add_subplot(gs01[3])
            self.ax["ax_error_psi"].set_ylim([-np.pi, np.pi])
            self.ax["ax_error_psi"].set_ylabel(r"$\psi$ (m)")
            # self.ax["done"] = self.fig.add_subplot(gs01[4])
            # self.ax["done"].set_ylabel("Done")
            # self.ax['done_delta_t'] = self.fig.add_subplot(gs01[5])
            # self.ax['delta_t_go'] = self.fig.add_subplot(gs01[6])
            # self.ax['t_go'] = self.fig.add_subplot(gs01[7])
            # self.ax['t_go'].set_ylabel("$t\_go$ (s)")
            # self.ax['reward'] = self.fig.add_subplot(gs01[8])
            # self.ax['reward'].set_ylabel("Reward")
            # self.ax['uav_col'] = self.fig.add_subplot(gs01[9])
            # self.ax['uav_col'].set_ylabel("UAV_col")
            # self.ax['obs_col'] = self.fig.add_subplot(gs01[10])
            # self.ax['obs_col'].set_ylabel("NCFO_col")
            # self.ax['delta_r'] = self.fig.add_subplot(gs01[11])
            # self.ax['delta_r'].set_ylabel("$\parallel \Delta \mathbf{r} \parallel$")
            # self.ax['delta_v'] = self.fig.add_subplot(gs01[12])
            # self.ax['delta_r'].set_ylabel("$\parallel \Delta \mathbf{v} \parallel$")

        self.ax["ax_3d"].set_xlim3d([-self.max_x, self.max_x])
        self.ax["ax_3d"].set_ylim3d([-self.max_y, self.max_y])
        self.ax["ax_3d"].set_zlim3d([0, self.max_z])

        self.ax["ax_3d"].set_xlabel("X (m)")
        self.ax["ax_3d"].set_ylabel("Y (m)")
        self.ax["ax_3d"].set_zlabel("Z (m)")

        self.ax["ax_3d"].set_title("Multi-UAV Simulation")

        # # add axis
        # n_points = 5
        # x_axis = np.linspace(0, self.max_x, n_points)
        # y_axis = np.linspace(0, self.max_y, n_points)
        # z_axis = np.linspace(0, self.max_z, n_points)

        # self.ax["ax_3d"].plot([0, 0], [0, 0], [0, 0], "k+")
        # self.ax["ax_3d"].plot(
        #     x_axis, np.zeros(n_points), np.zeros(n_points), "r--", linewidth=0.5
        # )
        # self.ax["ax_3d"].plot(
        #     np.zeros(n_points), y_axis, np.zeros(n_points), "g--", linewidth=0.5
        # )
        # self.ax["ax_3d"].plot(
        #     np.zeros(n_points), np.zeros(n_points), z_axis, "b--", linewidth=0.5
        # )

        # (0, 0) is bottom left, (1, 1) is top right
        # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.
        self.time_display = self.ax["ax_3d"].text2D(
            0.0, 0.95, "", color="red", transform=self.ax["ax_3d"].transAxes
        )
        self.init_entities()

        # for stopping simulation with the esc key
        self.fig.canvas.mpl_connect("key_press_event", self.keypress_routine)

        # view is elevation and azimuth angle
        # self.ax["ax_3d"].view_init(25, 45)
        # Overhead Z
        # self.ax["ax_3d"].view_init(90, 270)

        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def init_entities(self):
        c_idx = 0
        self.sprites = []
        target_sprite = TargetSprite(self.ax, self.target, color=self.cmap(c_idx))
        self.sprites.append(target_sprite)

        for uav in self.uavs.values():
            c_idx += 1
            uav_sprite = UavSprite(self.ax, uav, color=self.cmap(c_idx))
            self.sprites.append(uav_sprite)

        for obstacle in self.obstacles:
            obs_sprite = ObstacleSprite(self.ax, obstacle)
            self.sprites.append(obs_sprite)

    # TODO: update this to use blit
    # https://matplotlib.org/stable/api/animation_api.html
    # https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
    def update(
        self,
        time_elapsed,
        done=False,
        obs=None,
        rew=None,
        info=None,
        plot_results=False,
    ):
        self.fig.canvas.restore_region(self.background)
        self.time_display.set_text(f"Sim time = {time_elapsed:.2f} s")

        for sprite in self.sprites:
            sprite.update(time_elapsed, done)

        #     self.ax.draw_artist(uav_sprite.cm)
        self.fig.canvas.blit(self.fig.bbox)

        # # only plot legends if uav or target
        # for key, ax in self.ax.items():
        #     if key == "ax_3d":
        #         continue

        #     handles, labels = ax.get_legend_handles_labels()
        #     if labels:
        #         ax.legend()

        handles, labels = self.ax["ax_error_x"].get_legend_handles_labels()
        self.ax["legend"].legend(handles, labels, loc="center", ncol=len(labels))
        self.ax["legend"].axis("off")
        plt.pause(0.0000000000001)
        # if plot_results:
        # plt.show()
        # TODO: pass figure as rgba image instead
        return self.fig

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
