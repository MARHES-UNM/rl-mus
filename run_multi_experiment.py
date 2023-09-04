from datetime import datetime
import subprocess
from pathlib import Path

import os
import concurrent.futures
from functools import partial
import logging
import json


formatter = "%(asctime)s: %(name)s - %(levelname)s - <%(module)s:%(funcName)s:%(lineno)d> - %(message)s"
logging.basicConfig(
    # filename=os.path.join(app_log_path, log_file_name),
    format=formatter
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

max_num_episodes = 10
max_num_cpus = os.cpu_count() - 1


PATH = Path(__file__).parent.absolute().resolve()


def run_experiment(exp_config):
    logger.debug(f"exp_config:{exp_config}")
    default_config = f"{PATH}/configs/sim_config.cfg"
    with open(default_config, "rt") as f:
        config = json.load(f)

    config["exp_config"].update(exp_config["exp_config"])
    config["safety_layer_cfg"].update(exp_config["safety_layer_cfg"])
    config["env_config"].update(exp_config["env_config"])

    output_folder = os.path.join(log_dir, exp_config["exp_name"])
    exp_file_config = os.path.join(output_folder, "exp_sim_config.cfg")
    fname = os.path.join(output_folder, "result.json")

    config["fname"] = fname
    config["write_experiment"] = True
    experiment_num = exp_config["experiment_num"]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(exp_file_config, "w") as f:
        json.dump(config, f)

    args = [
        "python",
        "run_experiment.py",
        "--log_dir",
        f"{output_folder}",
        "--load_config",
        str(exp_file_config),
        "--max_num_episodes",
        str(max_num_episodes),
        "--experiment_num",
        str(experiment_num),
    ]

    rv = subprocess.call(args)
    # rv = subprocess.call(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    logger.debug(f"{exp_config['exp_name']} done running.")

    return rv


if __name__ == "__main__":
    branch_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode()
    )

    dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    log_dir = Path(f"./results/test_results/exp_{dir_timestamp}_{branch_hash}")

    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    target_v = [0.0, 1.0]
    # use_safe_action = [False, True]
    safe_action_type = ["none", "cbf", "nn_cbf"]
    num_obstacles = [20, 30]
    seeds = [0, 5000, 173]
    checkpoint_dir = r"/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-09-01-06-54_6a6ba7e/debug/train_safety_layer_00757_00000_0_eps=0.0001,eps_deriv=0.0000,lr=0.0029,weight_decay=0.0000_2023-09-01_06-55-02/checkpoint_000244/checkpoint"
    checkpoint_dir = r"/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-09-04-00-12_fd3b073/debug/train_safety_layer_545fd_00007_7_num_obstacles=8,target_v=1.0000,loss_action_weight=0.0800_2023-09-04_04-48-25/checkpoint_000199/checkpoint"
    checkpoint_dir = r"/home/prime/Documents/workspace/uav_sim/results/safety_layer/safety_layer2023-09-04-00-12_fd3b073/debug/train_safety_layer_545fd_00001_1_num_obstacles=8,target_v=0.0000,loss_action_weight=0.0100_2023-09-04_00-12-59/checkpoint_000199/checkpoint"

    # target_v = [0.0
    # safe_action_type = [None, "cbf", "nn_cbf"]
    # num_obstacles = [30]
    # seeds = [0]

    exp_configs = []
    experiment_num = 0
    for seed in seeds:
        for target in target_v:
            for action_type in safe_action_type:
                for num_obstacle in num_obstacles:
                    exp_config = {}
                    exp_config["exp_config"] = {"safe_action_type": action_type}
                    exp_config["safety_layer_cfg"] = {"checkpoint_dir": checkpoint_dir}

                    exp_config["env_config"] = {
                        "target_v": target,
                        "num_obstacles": num_obstacle,
                        "seed": seed,
                    }
                    file_prefix = {
                        "tgt_v": target,
                        "sa": action_type,
                        "o": num_obstacle,
                        "s": seed,
                    }
                    file_prefix = "_".join(
                        [f"{k}_{str(v)}" for k, v in file_prefix.items()]
                    )
                    exp_config["exp_name"] = f"exp_{experiment_num}_{file_prefix}"
                    exp_config["experiment_num"] = experiment_num

                    exp_configs.append(exp_config)
                    experiment_num += 1

    starter = partial(run_experiment)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_cpus) as executor:
        future_run_experiment = [
            executor.submit(starter, exp_config) for exp_config in exp_configs
        ]
        for future in concurrent.futures.as_completed(future_run_experiment):
            rv = future.result()
