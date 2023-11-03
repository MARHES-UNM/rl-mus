import unittest

from uav_sim.envs.uav_sim import UavSim
from uav_sim.utils.safety_layer import SafetyLayer


class TestSafetyLayer(unittest.TestCase):
    def setUp(self):
        self.env = UavSim({"num_uavs": 4, "num_obstacles": 4, "time_final": 10.0})
        self.config = {
            "replay_buffer_size": 64 * 10,
            "batch_size": 4,
            "num_epochs": 1,
            "device": "cpu",
            "num_iter_per_epoch": 2,
            "num_training_steps": 60,
            "num_eval_steps": 15,
            "use_rl": False,
        }
        self.sl = SafetyLayer(self.env, config=self.config)

    def test_init(self):
        self.sl._sample_steps(200)
        self.sl._train_batch()

    def test_train(self):
        self.sl.train()

    def test_rl_train(self):
        self.config = {
            "replay_buffer_size": 64 * 10,
            "batch_size": 4,
            "num_epochs": 1,
            "device": "cpu",
            "num_iter_per_epoch": 2,
            "num_training_steps": 60,
            "num_eval_steps": 15,
            "use_rl": True,
        }
        self.sl = SafetyLayer(self.env, config=self.config)
        self.sl.train()

    def test_get_mask(self):
        self.sl._sample_steps(200)
        batch = self.sl._replay_buffer.sample(self.sl._batch_size)
        constraints = self.sl._as_tensor(batch["constraint"])
        safe_mask, unsafe_mask, mid_mask = self.sl._get_mask(constraints)

        print(safe_mask)


if __name__ == "__main__":
    unittest.main()
