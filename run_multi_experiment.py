import argparse
from datetime import datetime
import subprocess
from pathlib import Path
from time import time

import os
import concurrent.futures
from functools import partial
import logging
import json
import itertools
from uav_sim.utils.utils import get_git_hash


formatter = "%(asctime)s: %(name)s - %(levelname)s - <%(module)s:%(funcName)s:%(lineno)d> - %(message)s"
logging.basicConfig(
    # filename=os.path.join(app_log_path, log_file_name),
    format=formatter
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

max_num_cpus = os.cpu_count() - 1

PATH = Path(__file__).parent.absolute().resolve()


def run_experiment(exp_config, log_dir, max_num_episodes):
    logger.debug(f"exp_config:{exp_config}")
    default_config = f"{PATH}/configs/sim_config.cfg"
    with open(default_config, "rt") as f:
        config = json.load(f)

    config["exp_config"].update(exp_config["exp_config"])
    if config["exp_config"]["safe_action_type"] == "nn_cbf":
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


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, help="folder to log experiment")
    parser.add_argument(
        "--exp_config",
        help="load experiment configuration.",
        default=f"{PATH}/configs/exp_basic_cfg.json",
    )
    parser.add_argument("--num_eps", help="Maximum number of episodes to run for.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()

    with open(args.exp_config, "rt") as f:
        exp_config = json.load(f)

    if not args.log_dir:
        branch_hash = get_git_hash()

        dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

        args.log_dir = Path(f"./results/test_results/exp_{dir_timestamp}_{branch_hash}")

    if not args.log_dir.exists():
        args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.num_eps:
        max_num_episodes = args.num_eps
    else:
        max_num_episodes = exp_config["exp_config"]["max_num_episodes"]

    target_v = exp_config["env_config"]["target_v"]
    max_num_obstacles = exp_config["env_config"]["max_num_obstacles"]
    seeds = exp_config["exp_config"]["seeds"]
    time_final = exp_config["env_config"]["time_final"]
    runs = exp_config["exp_config"]["runs"]
    run_nums = [i for i in range(len(runs))]

    exp_configs = []
    experiment_num = 0

    exp_items = list(
        itertools.product(seeds, target_v, run_nums, max_num_obstacles, time_final)
    )

    for exp_item in exp_items:
        seed = exp_item[0]
        target = exp_item[1]
        run_num = exp_item[2]
        num_obstacle = exp_item[3]
        t_final = exp_item[4]

        exp_config = {}
        exp_config["exp_config"] = {
            "name": runs[run_num]["name"],
            "run": runs[run_num]["run"],
            "checkpoint": runs[run_num]["checkpoint"],
            "safe_action_type": runs[run_num]["safe_action_type"],
        }
        if exp_config["exp_config"]["safe_action_type"] == "nn_cbf":
            checkpoint_dir = runs[run_num].get("sa_checkpoint_dir", None)
            exp_config["safety_layer_cfg"] = {
                "checkpoint_dir": checkpoint_dir,
                "seed": seed,
            }

        exp_config["env_config"] = {
            "target_v": target,
            "max_num_obstacles": num_obstacle,
            "seed": seed,
            "time_final": t_final,
        }

        file_prefix = {
            "tgt_v": target,
            "r": runs[run_num]["run"],
            "sa": runs[run_num]["safe_action_type"],
            "o": num_obstacle,
            "s": seed,
            "tf": t_final,
        }

        file_prefix = "_".join([f"{k}_{str(v)}" for k, v in file_prefix.items()])
        exp_config["exp_name"] = f"exp_{experiment_num}_{file_prefix}"
        exp_config["experiment_num"] = experiment_num

        exp_configs.append(exp_config)
        experiment_num += 1

    starter = partial(
        run_experiment, max_num_episodes=max_num_episodes, log_dir=args.log_dir
    )

    start_time = time()

    print("==========================================================")
    print("----------------------------------------------------------")
    print(f"Start time: {datetime.fromtimestamp(start_time)}")
    print("==========================================================")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_cpus) as executor:
        future_run_experiment = [
            executor.submit(starter, exp_config=exp_config)
            for exp_config in exp_configs
        ]
        for future in concurrent.futures.as_completed(future_run_experiment):
            rv = future.result()

    args = [
        "python",
        "plot_results.py",
        "--exp_folder",
        f"{args.log_dir}",
        "--exp_config",
        f"{args.exp_config}",
    ]

    logger.debug(f"plotting results")
    rv = subprocess.call(args)
    print("==========================================================")
    print(f"Finished All experiments. Time spent: {(time() - start_time) // 1} secs")
    print("==========================================================")
    logger.debug(f"Done!")
