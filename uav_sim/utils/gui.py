import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import numpy as np


class TargetSprite:
    def __init__(self, ax, num_targets=1):
        self.ax = ax
        self.targets = []
        for i in range(num_targets):
            (t_sprite,) = self.ax.plot([], [], [], "r.", markersize=4)

            self.targets.append(t_sprite)


class UavSprite:
    def __init__(self, ax):
        self.ax = ax
        (self.l1,) = self.ax.plot(
            [], [], [], color="blue", linewidth=1, antialiased=False
        )
        (self.l2,) = self.ax.plot(
            [], [], [], color="blue", linewidth=1, antialiased=False
        )

        (self.cm,) = self.ax.plot([], [], [], "k.")
        (self.rotor1,) = self.ax.plot([], [], [], "k.")
        (self.rotor2,) = self.ax.plot([], [], [], "k.")
        (self.rotor3,) = self.ax.plot([], [], [], "k.")
        (self.rotor4,) = self.ax.plot([], [], [], "k.")


class Gui:
    def __init__(self, uavs, max_x, max_y, max_z, fig=None):
        self.uavs = uavs
        self.fig = fig

        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_xlim3d([0, max_x])
        self.ax.set_ylim3d([0, max_y])
        self.ax.set_zlim3d([0, max_z])

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        self.ax.set_title("Multi-UAV Simulation")

        # add axis
        n_points = 5
        x_axis = np.linspace(0, max_x, n_points)
        y_axis = np.linspace(0, max_y, n_points)
        z_axis = np.linspace(0, max_z, n_points)

        self.ax.plot([0, 0], [0, 0], [0, 0], "k+")
        self.ax.plot(
            x_axis, np.zeros(n_points), np.zeros(n_points), "r--", linewidth=0.5
        )
        self.ax.plot(
            np.zeros(n_points), y_axis, np.zeros(n_points), "g--", linewidth=0.5
        )
        self.ax.plot(
            np.zeros(n_points), np.zeros(n_points), z_axis, "b--", linewidth=0.5
        )

        # (0, 0) is bottom left, (1, 1) is top right
        # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.
        self.time_display = self.ax.text2D(
            0.75, 0.95, "", color="red", transform=self.ax.transAxes
        )

        self.state_display = self.ax.text2D(
            0, 0.95, "", color="green", transform=self.ax.transAxes
        )

        self.init_entities()
        # for stopping simulation with the esc key
        self.fig.canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )

    def init_entities(self):
        self.sprites = []

        self.target_sprites = TargetSprite(self.ax, len(self.uavs))

        for idx in range(len(self.uavs)):
            uav_sprite = UavSprite(self.ax)
            self.sprites.append(uav_sprite)

    # TODO: update this to use blit
    # https://matplotlib.org/stable/api/animation_api.html
    # https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
    def update(self, time_elapsed):
        self.time_display.set_text(f"Sim time = {time_elapsed:.2f} s")

        target_points = np.array(
            [[0.5, 0.5, 0], [0.5, 1.5, 0], [1.5, 0.5, 0], [1.5, 1.5, 0]]
        ).T
        for idx, target in enumerate(self.target_sprites.targets):
            target.set_data(
                target_points[0, idx : idx + 1], target_points[1, idx : idx + 1]
            )
            target.set_3d_properties(target_points[2, idx : idx + 1])

        for uav, uav_sprite in zip(self.uavs, self.sprites):
            uav_state = uav.state
            self.state_display.set_text(
                f"x:{uav_state[0]:.2f}, y:{uav_state[1]:.2f}, z:{uav_state[2]:.2f}\n"
                f"phi: {uav_state[6]:.2f}, theta: {uav_state[7]:.2f}, psi: {uav_state[8]:.2f}"
            )

            R = uav.rotation_matrix()
            l = uav.l

            points = np.array(
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
            points = np.dot(R, points)

            # for idx in range(len(points)):
            #     points[idx, :] += uav._state[idx]
            points[0, :] += uav._state[0]
            points[1, :] += uav._state[1]
            points[2, :] += uav._state[2]

            uav_sprite.l1.set_data(points[0, 0:2], points[1, 0:2])
            uav_sprite.l1.set_3d_properties(points[2, 0:2])
            uav_sprite.l2.set_data(points[0, 2:4], points[1, 2:4])
            uav_sprite.l2.set_3d_properties(points[2, 2:4])
            uav_sprite.cm.set_data(points[0, 4:5], points[1, 4:5])
            uav_sprite.cm.set_3d_properties(points[2, 4:5])
            # uav_sprite.rotor1.set_data(points[0, 5:6], points[1, 5:6])
            # uav_sprite.rotor1.set_3d_properties(points[2, 5:6])
            # uav_sprite.rotor2.set_data(points[0, 6:7], points[1, 6:7])
            # uav_sprite.rotor2.set_3d_properties(points[2, 6:7])
            # uav_sprite.rotor3.set_data(points[0, 5:6], points[0, 5:6])
            # uav_sprite.rotor3.set_3d_properties(points[2, 5:6])
            # uav_sprite.rotor1.set_data(points[0, 5:6], points[0, 5:6])
            # uav_sprite.rotor1.set_3d_properties(points[2, 5:6])
            # uav_sprite.rotor2.set_data(points[0, 4:5], points[0, 4:5])

        plt.pause(0.0000000000001)

    def __del__(self):
        plt.close(self.fig)

    def close(self):
        plt.close(self.fig)
        self.fig = None
