import unittest

from uav_sim.envs.uav_sim import UavSim
from uav_sim.utils.safety_layer import SafetyLayer


class TestSafetyLayer(unittest.TestCase):
    def setUp(self):
        self.env = UavSim({"num_uavs": 2, "num_obstacles": 1})
        config = {"replay_buffer_size": 100, "batch_size": 64, "num_epochs": 1}
        self.sl = SafetyLayer(self.env, config=config)

    def test_init(self):
        self.sl._sample_steps(200)
        self.sl._train_batch()

        pass


if __name__ == "__main__":
    unittest.main()
