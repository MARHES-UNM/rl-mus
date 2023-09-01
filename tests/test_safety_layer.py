import unittest

from uav_sim.envs.uav_sim import UavSim
from uav_sim.utils.safety_layer import SafetyLayer


class TestSafetyLayer(unittest.TestCase):
    def setUp(self):
        self.env = UavSim({"num_uavs": 2, "num_obstacles": 1, "time_final": 10.0})
        config = {
            "replay_buffer_size": 64 * 10,
            "batch_size": 32,
            "num_training_iter": 10,
        }
        # config = {
        #     "num_eval_steps": 3000,
        #     "num_training_steps": 12800,
        #     "replay_buffer_size": 100000,
        #     "batch_size": 512,
        #     "num_training_iter": 10,
        # }
        self.sl = SafetyLayer(self.env, config=config)

    def test_init(self):
        self.sl._sample_steps(200)
        self.sl._train_batch()

    def test_train(self):
        self.sl.train()


if __name__ == "__main__":
    unittest.main()
