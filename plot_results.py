# %% [markdown]
# ## Visualizing result from tensorboard.
# #### from tutorial: https://www.tensorflow.org/tensorboard/dataframe_api
#
# #### The following packages are required:
# ```
# pip install tensorboard pandas
# pip install matplotlib seaborn
# ```

# %%
import argparse
import pandas as pd
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import itertools

# plt.style.use(r"configs/paper_plot_style.mplstyle")
plt.style.use("default")
import seaborn as sns
import json
import os
import numpy as np

from matplotlib.patches import Circle
from pathlib import Path
from uav_sim.utils.utils import np_mad, max_abs_diff

PATH = Path(__file__).parent.absolute().resolve()


def get_sa_sat(data, max_dt_std=0.1):
    sa_sat = []
    data_done = np.array(data["uav_done"])
    data_done_time = np.array(data["uav_done_time"])
    for num_epsisode in range(data["num_episodes"]):
        output = 0
        if (
            all(data_done[:, num_epsisode])
            and np.std(data_done_time[:, num_epsisode]) <= max_dt_std
        ):
            output = 1
        sa_sat.append(output)

    return np.array(sa_sat).mean()


def get_data(all_progress):
    # data_dict = {parameter: [] for parameter in parameter_list}
    data_dict = {}

    for progress in all_progress:
        with open(str(progress.absolute()), "r") as f:
            try:
                data = json.loads(f.readlines()[-1])
            except Exception as e:
                f.seek(0)
                # data = json.loads(f.readlines()[-2])
                print(f"error reading {progress.absolute()} skipping.")
                continue
            data["target_v"] = data["env_config"]["target_v"]
            # data["safe_action"] = data["exp_config"]["safe_action_type"]
            data["name"] = data["exp_config"]["name"]
            data["num_obs"] = data["env_config"]["max_num_obstacles"]
            data["num_uavs"] = data["env_config"]["num_uavs"]
            data["seed"] = data["env_config"]["seed"]
            data["tf"] = data["env_config"]["time_final"]
            data["max_dt"] = data["env_config"]["t_go_max"]
            data["max_dt_std"] = data["env_config"]["max_dt_std"]
            data["max_dt_std"] = 0.5
            data["uav_collision_eps"] = (
                data["uav_collision"] / data["num_uavs"] / data["num_episodes"]
            )
            data["obs_collision_eps"] = (
                data["obs_collision"] / data["num_uavs"] / data["num_episodes"]
            )
            data["uav_reward_eps"] = (
                data["uav_reward"] / data["num_uavs"] / data["num_episodes"]
            )
            data["uav_crashed_eps"] = (
                data["uav_crashed"] / data["num_uavs"] / data["num_episodes"]
            )
            # data["uav_done"] = np.mean(data["uav_done"], axis=1).sum()
            # sum up to to the number of uavs in the mean and gives the average across episodes
            data["uav_sa_sat_cal"] = get_sa_sat(data, data["max_dt_std"])
            data["uav_done"] = np.mean(data["uav_done"], axis=0).mean()
            data["uav_sa_sat"] = np.mean(data["uav_sa_sat"], axis=0).mean()
            data["uav_done_dt"] = np.mean(np.abs(data["uav_done_dt"]))
            data["uav_done_time_std"] = np.std(data["uav_done_time"], axis=0).mean()
            data["uav_done_time_max"] = max_abs_diff(
                data["uav_done_time"], axis=0
            ).mean()
            # using mean absolute difference here to better show dispersion
            # data["uav_done_time_std"] = np_mad(data["uav_done_time"], axis=0).mean()

            # data["uav_done_dt"] = np.mean(data["uav_done_dt"])
            # uav_done_time = np.nan_to_num(
            #     np.array(data["uav_done_time"], dtype=np.float64), nan=100
            # )
            # data["uav_done_time"] = np.nanmean(uav_done_time)
            # data["tf_error"] = np.nanmean(np.abs(uav_done_time - data["tf"]))
            for k, v in data.items():
                if k not in data_dict:
                    data_dict[k] = []
                data_dict[k].append(v)
            data_dict["file_name"] = progress.absolute()

    df = pd.DataFrame(data_dict)
    return df


def plot_groups(groups, items, output_folder, plot_type="box", skip_legend=False):
    if plot_type == "bar":
        plot_func = sns.barplot
    elif plot_type == "box":
        plot_func = sns.boxplot
    else:
        raise NameError("unknown plot_type")

    categories = list(items.keys())
    for group in groups:
        group_to_plot = group["group"].get_group(group["group_key"])
        group_to_plot.name = group_to_plot.name.astype("category")

        avg_group = group_to_plot.groupby("name")[categories].mean()
        avg_group.rename(columns=items, inplace=True)
        # avg_group.to_csv(
        #     os.path.join(output_folder, f"df_{group['group_title']}.csv"),
        #     float_format="{:.2f}".format,
        # )
        avg_group.to_latex(
            os.path.join(output_folder, f"df_{group['group_title']}.tex"),
            index=False,
            formatters={},
            float_format="{:.2f}".format,
        )

        for key, value in items.items():
            fig, ax = plt.subplots()

            ax = plot_func(
                group_to_plot,
                hue="name",
                x=group["group_x"],
                y=key,
                ax=ax,
            )
            if key == "uav_done_time_std":
                ax.axhline(y=group_to_plot.max_dt_std.mean(), color="k")

            if key == "uav_done_time_max":
                ax.axhline(y=group_to_plot.max_dt_std.mean() * 2, color="k")

            if key == "uav_done_dt":
                ax.axhline(y=group_to_plot.max_dt.mean(), color="k")

            if key in ["uav_done", "uav_sa_sat", "uav_sa_sat_cal"]:
                ax.axhline(y=1.0, color="k")

            ax.set_ylabel(value)
            ax.set_xlabel(group["x_label"])
            # if item == "episode_reward":
            # ax.invert_yaxis()
            ax.grid()

            if skip_legend:
                # don't plot legends here. see below
                ax.legend().remove()
            else:
                ax.legend()

            fig.savefig(
                os.path.join(
                    output_folder, f"img_{plot_type}_{group['group_title']}_{key}.png"
                )
            )
            plt.close(fig)

    # We're going to create a separate figure with legends.
    # https://stackoverflow.com/questions/4534480/get-legend-as-a-separate-picture-in-matplotlib
    # then create a new image
    # adjust the figure size as necessary
    figsize = (18, 2)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax_leg.legend(handles, labels, loc="center", ncol=len(labels))
        # ax_leg.legend(handles, labels_to_plot, loc="center", ncol=3)
        # ax_leg.legend(["Safe_True", "safe_false"], loc="center", ncol=3)
        # hide the axes frame and the x/y labels
        ax_leg.axis("off")
        fig_leg.savefig(os.path.join(output_folder, "plt_label.png"))
    plt.close(fig_leg)


def plot_uav_states(
    results,
    env_config,
    num_episode=0,
    skip_legend=False,
    save_figs=False,
    image_folder=None,
    plt_prefix=None,
    show_plot=True,
):
    num_uavs = env_config["num_uavs"]
    num_obstacles = env_config["max_num_obstacles"]
    target_r = env_config["target_r"]
    pad_r = env_config["pad_r"]
    obstacle_radius = env_config["obstacle_radius"]
    uav_collision_list = results["episode_data"]["uav_collision_list"][num_episode]
    obstacle_collision_list = results["episode_data"]["obstacle_collision_list"][
        num_episode
    ]
    uav_done_list = results["episode_data"]["uav_done_list"][num_episode]
    uav_done_dt_list = results["episode_data"]["uav_done_dt_list"][num_episode]
    uav_dt_go_list = results["episode_data"]["uav_dt_go_list"][num_episode]
    uav_t_go_list = results["episode_data"]["uav_t_go_list"][num_episode]
    rel_pad_dist = results["episode_data"]["rel_pad_dist"][num_episode]
    rel_pad_vel = results["episode_data"]["rel_pad_vel"][num_episode]
    uav_reward = results["episode_data"]["uav_reward"][num_episode]
    time_step_list = results["episode_data"]["time_step_list"][num_episode]

    uav_state = np.array(results["episode_data"]["uav_state"])[num_episode]
    target_state = np.array(results["episode_data"]["target_state"])[num_episode]
    rel_pad_state = np.array(results["episode_data"]["rel_pad_state"])[num_episode]
    obstacle_state = np.array(results["episode_data"]["obstacle_state"])[num_episode]

    c_map = plt.get_cmap("tab10")

    all_axes = []
    all_figs = []

    fig = plt.figure()
    all_figs.append(fig)
    ax_leg = fig.add_subplot(611)  # legend
    ax21 = fig.add_subplot(612)
    ax22 = fig.add_subplot(613)
    ax23 = fig.add_subplot(614)
    ax231 = fig.add_subplot(615)
    ax24 = fig.add_subplot(616)

    fig = plt.figure()
    all_figs.append(fig)
    ax_1_leg = fig.add_subplot(511)  # legend
    ax = fig.add_subplot(512)  # uav collision
    ax1 = fig.add_subplot(513)  # ncfo collision
    ax3 = fig.add_subplot(514)  # delta_r
    ax31 = fig.add_subplot(515)  # delta_v

    fig = plt.figure()
    all_figs.append(fig)
    ax5 = fig.add_subplot(111, projection="3d")

    all_axes.extend([ax21, ax22, ax23, ax24, ax, ax1, ax3, ax31])

    c_idx = 0
    for idx in range(num_uavs):
        uav_c = c_map(c_idx)
        c_idx += 1

        ax21.plot(time_step_list, uav_done_list[idx], label=f"uav_{idx}")
        ax21.set_ylabel("Done")
        ax22.plot(time_step_list, uav_done_dt_list[idx], label=f"uav_{idx}")
        ax22.set_ylabel("Done $\Delta t$")
        ax23.plot(time_step_list, uav_dt_go_list[idx], label=f"uav_{idx}")
        ax23.set_ylabel("$\Delta t\_go$")
        ax231.plot(time_step_list, uav_t_go_list[idx], label=f"uav_{idx}")
        ax231.set_ylabel("$t\_go$")
        ax24.plot(time_step_list, uav_reward[idx], label=f"uav_{idx}")
        ax24.set_ylabel("Reward")
        handles, labels = ax21.get_legend_handles_labels()
        ax_leg.legend(handles, labels, loc="center", ncol=len(labels))
        ax_leg.axis("off")

        ax.plot(time_step_list, uav_collision_list[idx], label=f"uav_{idx}")
        ax.set_ylabel("UAV_col")
        ax1.plot(time_step_list, obstacle_collision_list[idx], label=f"uav_{idx}")
        ax1.set_ylabel("NCFO_col")
        ax3.plot(time_step_list, rel_pad_dist[idx], label=f"uav_{idx}")
        ax3.set_ylabel("$\parallel \Delta \mathbf{r} \parallel$")
        ax31.plot(time_step_list, rel_pad_vel[idx], label=f"uav_{idx}")
        ax31.set_ylabel("$\parallel \Delta \mathbf{v} \parallel$")
        handles, labels = ax.get_legend_handles_labels()
        ax_1_leg.legend(handles, labels, loc="center", ncol=len(labels))
        ax_1_leg.axis("off")

        ax5.plot(
            uav_state[idx, :, 0],
            uav_state[idx, :, 1],
            uav_state[idx, :, 2],
            label=f"uav_{idx}",
            color=uav_c,
        )

        uav_pad = Circle(
            (
                rel_pad_state[idx, 0, 0],
                rel_pad_state[idx, 0, 1],
                rel_pad_state[idx, 0, 2],
            ),
            pad_r,
            fill=False,
            color=uav_c,
        )
        ax5.add_patch(uav_pad)
        art3d.patch_2d_to_3d(uav_pad, z=rel_pad_state[idx, 0, 2], zdir="z")
    radius = obstacle_radius
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]

    target_c = c_map(c_idx)
    tareget_circle = Circle(
        (target_state[0, 0], target_state[0, 1]), target_r, fill=False, color=target_c
    )
    ax5.add_patch(tareget_circle)
    art3d.patch_2d_to_3d(tareget_circle, z=target_state[0, 2], zdir="z")

    for obs_id in range(num_obstacles):
        center = obstacle_state[obs_id, 0, 0:3]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        ax5.plot_wireframe(x, y, z, color="r", alpha=0.1)

    ax5.axis("equal")
    ax5.set_xlabel("X (m)")
    ax5.set_ylabel("Y (m)")
    ax5.set_zlabel("Z (m)")
    # ax5.legend()

    for ax_ in all_axes:
        ax_.set_xlabel("t (s)")

        ax_.label_outer()

        ax_.legend().remove()
        # if skip_legend:
        # ax_.legend().remove()
        # else:
        # ax_.legend()

    if save_figs:
        suffixes = [
            # "uav_ncfo_col.png",
            "uav_done.png",
            "r_v_time.png",
            "uav_3d_states.png",
        ]

        for fig_, suffix in zip(all_figs, suffixes):
            file_name = image_folder / f"{plt_prefix}_{suffix}"
            fig_.savefig(file_name)
            plt.close(fig_)

        figsize = (18, 2)
        fig_leg = plt.figure(figsize=figsize)
        ax_leg = fig_leg.add_subplot(111)
        # add the legend from the previous axes
        handles, labels = all_axes[0].get_legend_handles_labels()
        ax_leg.legend(
            handles, [f"UAV {idx}" for idx in range(num_uavs)], loc="center", ncol=4
        )
        # hide the axes frame and the x/y labels
        ax_leg.axis("off")
        fig_leg.savefig(image_folder / "uav_labels.png")

    if show_plot:
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_folder", help="Path to experiments")
    parser.add_argument("--img_folder", help="Folder to output plots")
    parser.add_argument(
        "--exp_config",
        help="experiment config",
        default=f"{PATH}/configs/exp_basic_cfg.json",
    )
    parser.add_argument(
        "--skip_legend",
        help="Don't plot legends on indvidual plots. ",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--plot_type", help="plot either box or bar plot", type=str, default="bar"
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    with open(args.exp_config, "rt") as f:
        exp_config = json.load(f)

    basedir_path = Path(args.exp_folder)
    basedir_list = list(basedir_path.glob("**/result.json"))

    if args.img_folder is not None:
        image_folder = basedir_path / args.img_folder
    else:
        image_folder = basedir_path / "images"

    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    df = get_data(basedir_list)

    exp_config["labels"] = [
        (run["name"], run["label"]) for run in exp_config["exp_config"]["runs"]
    ]

    df["name"] = df["name"].replace(
        {name: label for (name, label) in exp_config["labels"]}
    )

    names = [label[1] for label in exp_config["labels"]]
    df = df[df["name"].isin(names)]
    groups_to_plot = exp_config["groups_to_plot"]

    for idx, group in enumerate(groups_to_plot):
        val = group["group_key"]
        group_key = val[0] if len(val) == 1 else tuple(val)
        groups_to_plot[idx].update(
            {
                "group": df.groupby(group["group"]),
                "group_key": group_key,
            }
        )

    items_to_plot = exp_config["items_to_plot"]

    sns.color_palette("tab10")
    plot_groups(
        groups_to_plot,
        items_to_plot,
        image_folder,
        plot_type=args.plot_type,
        skip_legend=args.skip_legend,
    )

    obs_group = df.groupby(["seed", "num_uavs", "num_obs", "name", "target_v", "tf"])
    obs_group.groups.keys()

    target_v = exp_config["env_config"]["target_v"]
    time_final = exp_config["env_config"]["time_final"]
    num_uavs = exp_config["env_config"]["num_uavs"]
    max_num_obstacles = exp_config["env_config"]["max_num_obstacles"]
    seeds = exp_config["exp_config"]["seeds"]
    names = [label[1] for label in exp_config["labels"]]

    groups_to_plot = list(
        itertools.product(
            [
                seeds[0],
            ],
            [num_uavs[0]],
            [
                max_num_obstacles[0],
            ],
            names,
            target_v,
            [
                time_final[0],
            ],
        )
    )

    # TODO: convert to dataframe, pad the data to make them all the same lengths. plot the mean and std
    for group_to_plot in groups_to_plot:
        # There's only one group so only one key
        num_uavs = obs_group.get_group(group_to_plot)["env_config"].to_numpy()[0][
            "num_uavs"
        ]

        env_config = obs_group.get_group(group_to_plot)["env_config"].to_numpy()[0]
        data = obs_group.get_group(group_to_plot)["episode_data"].to_numpy()[0]

        def get_padded_array(d):
            # get max number of episode length
            # max_len = max([len(y) for eps in d for y in eps])
            # new_array = np.array(
            #     [y + [0] * (max_len - len(y)) for eps in d for y in eps]
            # )
            num_episodes = len(d)
            min_episode_len = min([len(y) for eps in d for y in eps])
            # pad data to be equal size
            new_array = np.array([y[:min_episode_len] for eps in d for y in eps])
            # (num_uavs, num_episodes, episode_length)
            new_array = new_array.reshape((num_uavs, num_episodes, -1))
            return new_array

        results = {"episode_data": {}}
        time_step_list = data["time_step_list"]
        results["episode_data"]["uav_collision_list"] = get_padded_array(
            data["uav_collision_list"]
        )
        results["episode_data"]["obstacle_collision_list"] = get_padded_array(
            data["obstacle_collision_list"]
        )
        results["episode_data"]["rel_pad_dist"] = get_padded_array(data["rel_pad_dist"])
        results["episode_data"]["rel_pad_vel"] = get_padded_array(data["rel_pad_vel"])

        results["episode_data"] = data
        plt_prefix = {
            "seed": group_to_plot[0],
            "uavs": group_to_plot[1],
            "obs": group_to_plot[2],
            "sa": group_to_plot[3].lower().replace("-", "_"),
            "tgt_v": group_to_plot[4],
        }

        plt_prefix = "_".join([f"{k}_{str(v)}" for k, v in plt_prefix.items()])

        plot_uav_states(
            results,
            env_config,
            num_episode=0,
            skip_legend=args.skip_legend,
            save_figs=True,
            image_folder=image_folder,
            plt_prefix=plt_prefix,
            show_plot=False,
        )


if __name__ == "__main__":
    main()
