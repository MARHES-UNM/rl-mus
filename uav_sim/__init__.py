from gym.envs.registration import register

register(id="uav-sim-v0", entry_point="uav_sim.envs:UavSim")