from ray import tune
from uav_sim.envs.uav_sim import UavSim

tune.register_env(
    "rl-mus-v0",
    lambda env_config: UavSim(env_config=env_config),
)
