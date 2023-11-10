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

PATH = Path(__file__).parent.absolute().resolve()


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
            data["uav_done"] = np.average(data["uav_done"], axis=1).sum()
            data["uav_done_dt"] = np.mean(np.abs(data["uav_done_dt"]))
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
    for group in groups:
        for key, value in items.items():
            fig, ax = plt.subplots()
            # ax.set_prop_cycle('color', sns.color_palette("colorblind",len(labels_to_plot)))
            # ax.set_prop_cycle('color', sns.color_palette("tab10",len(labels_to_plot)))
            # print(f'group_key{group["group_key"]}')
            group_to_plot = group["group"].get_group(group["group_key"])
            group_to_plot.name = group_to_plot.name.astype("category")
            # group_to_plot.safe_action = group_to_plot.safe_action.astype("category")
            # group_to_plot.safe_action = group_to_plot.safe_action.cat.set_categories(
            # labels_to_plot
            # )
            if group["group_x"] == "name":
                ax = plot_func(
                    group_to_plot,
                    x=group["group_x"],
                    y=key,
                    ax=ax,
                )
            else:
                ax = plot_func(
                    group_to_plot,
                    hue="name",
                    x=group["group_x"],
                    y=key,
                    ax=ax,
                )

            if key == "uav_done_dt":
                ax.axhline(y=group_to_plot.max_dt.mean(), color="k")

            if key == "uav_done":
                ax.axhline(y=group_to_plot.num_uavs.mean(), color="k")

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
    figsize = (12, 3)
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_folder", help="Path to experiments")
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
    image_folder = basedir_path / "images"

    if not image_folder.exists():
        image_folder.mkdir(parents=True, exist_ok=True)

    df = get_data(basedir_list)

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
    sns.color_palette("colorblind")

    # safe_action_type = exp_config["exp_config"]["safe_action_type"]
    # labels_to_plot = exp_config["labels_to_plot"]
    # df["safe_action"] = df["safe_action"].replace(
    #     {sa_type: label for sa_type, label in zip(safe_action_type, labels_to_plot)}
    # )
    plot_groups(
        groups_to_plot,
        items_to_plot,
        image_folder,
        plot_type=args.plot_type,
        skip_legend=args.skip_legend,
    )

    obs_group = df.groupby(["seed", "num_obs", "name", "target_v"])
    obs_group.groups.keys()

    target_v = exp_config["env_config"]["target_v"]
    max_num_obstacles = exp_config["env_config"]["max_num_obstacles"]
    seeds = exp_config["exp_config"]["seeds"]
    runs = exp_config["exp_config"]["runs"]

    groups_to_plot = []
    for run in runs:
        for v in target_v:
            groups_to_plot.append((seeds[0], max_num_obstacles[0], run["name"], v))

    # TODO: convert to dataframe, pad the data to make them all the same lengths. plot the mean and std
    for group_to_plot in groups_to_plot:
        # There's only one group so only one key
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
            num_episodes = len(d)
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
            if args.skip_legend:
                # don't plot legends here. see below
                ax_.legend().remove()
            else:
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


if __name__ == "__main__":
    main()
