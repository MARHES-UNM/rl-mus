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
from tkinter import W
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use(r"configs/paper_plot_style.mplstyle")
import seaborn as sns
import json
from scipy import stats
import os
import numpy as np

from pathlib import Path


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
            # data["safe_action"] = data["env_config"]["use_safe_action"]
            data["safe_action"] = data["exp_config"]["safe_action_type"]
            data["num_obs"] = data["env_config"]["num_obstacles"]
            data["seed"] = data["env_config"]["seed"]
            data["uav_done"] = np.average(data["uav_done"], axis=1).sum()
            uav_done_time = np.nan_to_num(
                np.array(data["uav_done_time"], dtype=np.float64), nan=100
            )
            # print(uav_done_time)
            data["uav_done_time"] = np.nanmean(uav_done_time)
            for k, v in data.items():
                if k not in data_dict:
                    data_dict[k] = []
                data_dict[k].append(v)
            data_dict["file_name"] = progress.absolute()

    df = pd.DataFrame(data_dict)
    return df


def plot_groups(groups, items, output_folder, labels_to_plot, plot_type="box"):
    if plot_type == "bar":
        plot_func = sns.barplot
    elif plot_type == "box":
        plot_func = sns.boxplot
    else:
        raise NameError("unknown plot_type")
    for group in groups:
        for key, value in items.items():
            # fig, ax = plt.subplots(figsize=(12, 10))
            fig, ax = plt.subplots()
            # ax.set_prop_cycle('color', sns.color_palette("colorblind",len(labels_to_plot)))
            # ax.set_prop_cycle('color', sns.color_palette("tab10",len(labels_to_plot)))
            # print(f'group_key{group["group_key"]}')
            group_to_plot = group["group"].get_group(group["group_key"])
            group_to_plot.safe_action = group_to_plot.safe_action.astype("category")
            group_to_plot.safe_action = group_to_plot.safe_action.cat.set_categories(
                labels_to_plot
            )
            ax = plot_func(
                group_to_plot, hue="safe_action", x=group["group_x"], y=key, ax=ax
            )
            ax.set_ylabel(value)
            ax.set_xlabel(group["x_label"])
            # if item == "episode_reward":
            #     ax.invert_yaxis()
            ax.grid()
            ax.legend()
            # don't plot legends here. see below
            # ax.legend_.remove()

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
    figsize = (12, 3)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    handles, labels = ax.get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc="center", ncol=3)
    # ax_leg.legend(handles, labels_to_plot, loc="center", ncol=3)
    # ax_leg.legend(["Safe_True", "safe_false"], loc="center", ncol=3)
    # hide the axes frame and the x/y labels
    ax_leg.axis("off")
    fig_leg.savefig(os.path.join(output_folder, "plt_label.png"))
    plt.close(fig_leg)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_folder", help="Path to experiments")

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    basedir_path = Path(args.exp_folder)
    basedir_list = list(basedir_path.glob("**/result.json"))
    image_folder = basedir_path / "images"

    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    df = get_data(basedir_list)

    groups_to_plot = [
        {
            "group": df.groupby(["target_v"]),
            "group_x": "num_obs",
            "group_key": (0),
            "group_title": "vary_num_obs_tgt_v_0",
            "x_label": "Number of Obstacles",
        },
        {
            "group": df.groupby(["target_v"]),
            "group_x": "num_obs",
            "group_key": (1),
            "group_title": "vary_num_obs_tgt_v_1",
            "x_label": "Number of Obstacles",
        },
        {
            "group": df.groupby(["num_obs", "target_v"]),
            "group_x": "seed",
            "group_key": (30, 0.0),
            "group_title": "vary_seed_num_obs_30",
            "x_label": "Seed",
        },
        {
            "group": df.groupby(["num_obs"]),
            "group_x": "target_v",
            "group_key": (30),
            "group_title": "vary_tgt_num_obs_30",
            "x_label": "Target Vel (m/s)",
        },
    ]

    items_to_plot = {
        "uav_collision": "Total UAV Collisions",
        "obs_collision": "Total NCFO Collision",
        "uav_done_time": "Time Landed",
        "uav_done": "UAV Landed",
    }

    sns.color_palette("colorblind")

    # labels_to_plot = ["Safe_Action_None", "Safe_Action_CBF", "Safe_Action_NNCBF"]
    labels_to_plot = ["none", "cbf", "nn_cbf"]

    plot_groups(groups_to_plot, items_to_plot, image_folder, labels_to_plot)

    obs_group = df.groupby(["seed", "num_obs", "safe_action", "target_v"])
    obs_group.groups.keys()

    safe_action = ["none", "cbf", "nn_cbf"]
    target_v = [0.0, 1.0]

    groups_to_plot = []
    for action in safe_action:
        for v in target_v:
            groups_to_plot.append((5000, 30, action, v))

    groups_to_plot

    # num_uavs = 4
    # %%
    # TODO: convert to dataframe, pad the data to make them all the same lengths. plot the mean and std
    for group_to_plot in groups_to_plot:
        # There's only one group so only one key
        num_episodes = obs_group.get_group(group_to_plot)["num_episodes"].to_numpy()[0]
        num_uavs = obs_group.get_group(group_to_plot)["env_config"].to_numpy()[0][
            "num_uavs"
        ]
        data = obs_group.get_group(group_to_plot)["episode_data"].to_numpy()[0]

        def get_padded_array(d):
            # get max number of episode length
            # max_len = max([len(y) for eps in d for y in eps])
            # new_array = np.array(
            #     [y + [0] * (max_len - len(y)) for eps in d for y in eps]
            # )
            min_episode_len = min([len(y) for eps in d for y in eps])
            # pad data to be equal size
            new_array = np.array([y[:min_episode_len] for eps in d for y in eps])
            # (num_uavs, num_episodes, episode_length)
            new_array = new_array.reshape((num_uavs, num_episodes, -1))
            return new_array

        time_step_list = get_padded_array(data["time_step_list"])[0, 0]
        uav_collision_list = get_padded_array(data["uav_collision_list"])
        obstacle_collision_list = get_padded_array(data["obstacle_collision_list"])
        rel_pad_dist = get_padded_array(data["rel_pad_dist"])
        rel_pad_vel = get_padded_array(data["rel_pad_vel"])

        all_axes = []
        all_figs = []
        for i in range(4):
            fig = plt.figure(figsize=(12, 8))
            all_axes.append(fig.add_subplot(111))
            all_figs.append(fig)

        # TODO: plot mean and stad for each param
        for idx in range(num_uavs):
            all_axes[0].plot(
                time_step_list,
                uav_collision_list[idx].mean(axis=0),
                label=f"uav_{idx}",
            )
            all_axes[1].plot(
                time_step_list,
                obstacle_collision_list[idx].mean(axis=0),
                label=f"uav_{idx}",
            )
            all_axes[2].plot(
                time_step_list,
                rel_pad_dist[idx].mean(axis=0),
                label=f"uav_{idx}",
            )
            all_axes[3].plot(
                time_step_list,
                rel_pad_vel[idx].mean(axis=0),
                label=f"uav_{idx}",
            )

        all_axes[0].set_ylabel("UAV collisions")
        all_axes[1].set_ylabel("NCFO collisions")
        all_axes[2].set_ylabel("$\parallel \Delta \mathbf{r} \parallel$")
        all_axes[3].set_ylabel("$\parallel \Delta \mathbf{v} \parallel$")

        for ax_ in all_axes:
            ax_.set_xlabel("t (s)")
            ax_.legend()

        plt_prefix = {
            "seed": group_to_plot[0],
            "obs": group_to_plot[1],
            "sa": group_to_plot[2],
            "tgt_v": group_to_plot[3],
        }

        plt_prefix = "_".join([f"{k}_{str(v)}" for k, v in plt_prefix.items()])

        suffixes = ["uav_col.png", "ncfo_col.png", "r_time.png", "v_time.png"]

        for fig_, suffix in zip(all_figs, suffixes):
            file_name = image_folder / f"{plt_prefix}_{suffix}"
            fig_.savefig(file_name)
            plt.close(fig_)

    figsize = (10, 3)
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

    # plt.show()


if __name__ == "__main__":
    main()
