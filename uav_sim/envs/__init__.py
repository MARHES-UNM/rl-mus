from ray import tune
from uav_sim.envs.uav_sim import UavSim
from uav_sim.envs.uav_rl_ren import UavRlRen
from uav_sim.envs.curriculum_uav_sim import CurriculumEnv

tune.register_env(
    "multi-uav-sim-v0",
    lambda env_config: UavSim(env_config=env_config),
)

tune.register_env(
    "multi-uav-ren-v0",
    lambda env_config: UavRlRen(env_config=env_config),
)

tune.register_env(
    "multi-uav-sim-curriculum-v0",
    lambda env_config: CurriculumEnv(config=env_config),
)
