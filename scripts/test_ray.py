from ray.rllib.algorithms.ppo import PPOConfig

import ray
ENV_VARIABLES = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONWARNINGS": "ignore::DeprecationWarning",
}
my_runtime_env = {"env_vars": ENV_VARIABLES}
# ray.init()
ray.init(runtime_env=my_runtime_env)


config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment("Taxi-v3")
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(evaluation_num_workers=1)
)


algo = config.build()  # 2. build the algorithm,

for _ in range(5):
    print(algo.train())  # 3. train it,

algo.evaluate()  # 4. and evaluate it.
