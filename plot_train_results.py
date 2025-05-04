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
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use(r"configs/paper_plot_style.mplstyle")
# plt.style.use("paper_plot_style.mplstyle")
import seaborn as sns
import json
from scipy import stats
import tensorboard as tb
import re
import os

import pathlib

# %% [markdown]
# ## Helper functions


# %%
def parse_key(reg_exp, str_to_parse):
    """_summary_

    Args:
        reg_exp (_type_): _description_
        str_to_parse (_type_): _description_

    Returns:
        _type_: _description_
    """
    found = re.findall(reg_exp, str_to_parse)
    result = "seed_n/a"
    if found:
        result = found[0].replace("=", "_")
    print(f"found: {result}")
    return result


# %%
# Get
def get_progress(basedir, experiment_name, filter_list):
    """Give a base directory, the get progress.csv file and parse it

    Args:
        basedir (_type_): _description_
        string_pattern (_type_): _description_
        filter_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    basedir_path = pathlib.Path(basedir)
    all_progress = basedir_path.glob("**/progress.csv")

    df_list = []
    df_keys = []

    for progress in all_progress:
        print(progress.absolute())
        # filter_list = ["policy_reward_mean/pursuer", "training_iteration"]

        df_list.append(pd.read_csv(str(progress.absolute())).filter(items=filter_list))
        df_keys.append(experiment_name)

    df = pd.concat(df_list, keys=df_keys, names=["experiment_name"]).reset_index()

    return df


# %% [markdown]
# ### Constants
#

# %%
# CONSTANTS
image_output_folder = r"sim_results/train_results"

if not os.path.exists(image_output_folder):
    os.makedirs(image_output_folder, exist_ok=True)


filter_list = [
    "training_iteration",
    "time_total_s",
    "timesteps_total",
    "custom_metrics/obstacle_collisions_mean",
    "custom_metrics/uav_done_time_std_mean",
    "custom_metrics/uav_collisions_mean",
    "custom_metrics/uav_dt_go_mean",
    "custom_metrics/num_uav_landed_mean",
    "custom_metrics/uav_done_dt_mean",
    "custom_metrics/uav_rel_dist_mean",
    "custom_metrics/uav_sa_sat_mean",
    "policy_reward_mean/shared_policy",
]
labels_to_plot = [
    "PPO",
    "PPO_CUR",
]

# %%
basedir = r"checkpoints/uav_cur"
df_ppo_no_cur = get_progress(basedir, "uav_cur", filter_list)
print(df_ppo_no_cur.head)


# %%
basedir = r"checkpoints/uav_same_time"
df_ppo_cur = get_progress(basedir, "uav_same_time", filter_list)
print(df_ppo_cur.head)

# %%
# print keys
df_local = pd.concat([df_ppo_cur, df_ppo_no_cur])
df_groups = df_local.groupby("experiment_name")
keys = df_groups.groups.keys()
print(keys)

# %%
labels_keys = {
    "POETIC": "uav_same_time",
    # "POETIC_NO_REN": "uav_goal",
    "POETIC_CUR": "uav_cur",
}

# %%
# training_iteration
# time_total_s
# timesteps_total
# custom_metrics/obstacle_collisions_mean
# custom_metrics/uav_collisions_mean
# custom_metrics/uav_dt_go_mean
# custom_metrics/num_uav_landed_mean
# custom_metrics/uav_done_dt_mean
# custom_metrics/uav_rel_dist_mean
# policy_reward_mean/shared_policy
parameter_keys = {
    "Reward ": "policy_reward_mean/shared_policy",
    "UAV Dest": "custom_metrics/num_uav_landed_mean",
    "UAV Collisions": "custom_metrics/uav_collisions_mean",
    "NCFO Collisions": "custom_metrics/obstacle_collisions_mean",
    "UAV $\Delta t$": "custom_metrics/uav_done_dt_mean",
    "UAV $\Delta t std$": "custom_metrics/uav_done_time_std_mean",
    "UAV Sat": "custom_metrics/uav_sa_sat_mean",
}

# %%

# sns.color_palette("colorblind")


def plot_parameters(
    df_groups,
    labels_to_plot,
    parameter_keys,
    labels_keys,
    fig_name,
    window_size=5,
    x_variable="training_iteration",
    x_label="Training Iteration",
):
    for parameter_key, parameter in parameter_keys.items():

        # fig, ax = plt.subplots(figsize=(12, 10))
        fig, ax = plt.subplots()
        # ax.set_prop_cycle('color', sns.color_palette("tab10",len(labels_to_plot)))

        # for label in labels_to_plot:
        for label, key in labels_keys.items():
            # get the key
            # key = labels_keys[label]
            # get the serie to plot
            serie_to_plot = df_groups.get_group(key)
            x_var = serie_to_plot[x_variable].to_numpy()
            variable = serie_to_plot[parameter]

            running_mean = (
                variable.rolling(window_size, min_periods=1).mean().to_numpy()
            )
            running_std = variable.rolling(window_size, min_periods=1).std().to_numpy()
            ax.fill_between(
                x_var,
                running_mean + running_std,
                running_mean - running_std,
                alpha=0.25,
            )
            # sns.lineplot(x=x_var, y=running_mean)
            ax.plot(x_var, running_mean, label=label)
            # ax.grid()
            # ax.legend()
            ax.set_xlabel(x_label)
            ax.set_ylabel(parameter_key)

            fig.savefig(
                os.path.join(
                    image_output_folder, f"{fig_name}_{parameter.replace('/', '_')}.png"
                )
            )

    figsize = (18, 3)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    ax_leg.legend(*ax.get_legend_handles_labels(), loc="center", ncol=3)
    # hide the axes frame and the x/y labels
    ax_leg.axis("off")
    fig_leg.savefig(os.path.join(image_output_folder, "ppo_train_legend.png"))


plot_parameters(
    df_groups,
    labels_to_plot,
    parameter_keys,
    labels_keys,
    x_variable="timesteps_total",
    x_label="Training Time Step",
    fig_name="ppo_vs_ppo_no_cur",
)

# %%
# Get the total time it takes to train the models
for label, key in labels_keys.items():
    # get the serie to plot
    serie_to_plot = df_groups.get_group(key)
    print(f"model_name: {label}")
    time_total_s = serie_to_plot["time_total_s"].max()
    m, s = divmod(time_total_s, 60)
    h, m = divmod(m, 60)
    # time_total_h = time_total_s / 3600
    print(f"total time (s): {serie_to_plot['time_total_s'].max()}")
    print(f"h:m:s: \t{h:.0f}:{m:.0f}:{s:.2f}")
    # print(f"total time (m): {time_total_m}")
    # print(f"total time (h): {time_total_h}")
