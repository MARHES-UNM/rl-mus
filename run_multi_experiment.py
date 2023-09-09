import argparse
from datetime import datetime
import subprocess
from pathlib import Path

import os
import concurrent.futures
from functools import partial
import logging
import json
from uav_sim.utils.utils import get_git_hash


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


def run_experiment(exp_config, log_dir, max_num_episodes):
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


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, help="folder to log experiment")
    parser.add_argument(
        "--exp_config",
        help="load experiment configuration.",
        default=f"{PATH}/configs/exp_basic_cfg.json",
    )
    parser.add_argument("--nn_cbf_dir", help="checkpoint for learned cbf")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()

    if args.exp_config:
        with open(args.exp_config, "rt") as f:
            args.exp_config = json.load(f)

    if not args.log_dir:
        branch_hash = get_git_hash()

        dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

        args.log_dir = Path(f"./results/test_results/exp_{dir_timestamp}_{branch_hash}")

    if not args.log_dir.exists():
        args.log_dir.mkdir(parents=True, exist_ok=True)

    max_num_episodes = args.exp_config["exp_config"]["max_num_episodes"]
    target_v = args.exp_config["env_config"]["target_v"]
    safe_action_type = args.exp_config["exp_config"]["safe_action_type"]
    num_obstacles = args.exp_config["env_config"]["num_obstacles"]
    seeds = args.exp_config["exp_config"]["seed"]

    if args.nn_cbf_dir is not None:
        checkpoint_dir = args.nn_cbf_dir
    else:
        checkpoint_dir = args.exp_config["safety_layer_cfg"]["checkpoint_dir"]

    exp_configs = []
    experiment_num = 0
    for seed in seeds:
        for target in target_v:
            for action_type in safe_action_type:
                for num_obstacle in num_obstacles:
                    exp_config = {}
                    exp_config["exp_config"] = {"safe_action_type": action_type}
                    exp_config["safety_layer_cfg"] = {
                        "checkpoint_dir": checkpoint_dir,
                        "seed": seed,
                    }

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

    starter = partial(
        run_experiment, max_num_episodes=max_num_episodes, log_dir=args.log_dir
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_cpus) as executor:
        future_run_experiment = [
            executor.submit(starter, exp_config=exp_config)
            for exp_config in exp_configs
        ]
        for future in concurrent.futures.as_completed(future_run_experiment):
            rv = future.result()
