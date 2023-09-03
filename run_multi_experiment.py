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

max_num_episodes = 500
max_num_cpus = os.cpu_count() - 1


PATH = Path(__file__).parent.absolute().resolve()


def run_experiment(exp_config):
    logger.debug(f"exp_config:{exp_config}")
    default_config = f"{PATH}/configs/sim_config.cfg"
    with open(default_config, "rt") as f:
        config = json.load(f)

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
    use_safe_action = [False, True]
    num_obstacles = [20, 30]
    seeds = [0, 5000, 173]

    target_v = [0.0]
    use_safe_action = [False, True]
    num_obstacles = [30]
    seeds = [0]

    exp_configs = []
    experiment_num = 0
    for seed in seeds:
        for target in target_v:
            for action in use_safe_action:
                for num_obstacle in num_obstacles:
                    exp_config = {}
                    env_config = {
                        "target_v": target,
                        "use_safe_action": action,
                        "num_obstacles": num_obstacle,
                        "seed": seed,
                    }
                    exp_config["env_config"] = env_config

                    file_prefix = {
                        "tgt_v": target,
                        "sa": action,
                        "o": num_obstacle,
                        "s": seed,
                    }
                    file_prefix = "_".join(
                        [
                            f"{experiment_num}_{k}_{str(int(v))}"
                            for k, v in file_prefix.items()
                        ]
                    )
                    exp_config["exp_name"] = f"exp_{file_prefix}"
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
