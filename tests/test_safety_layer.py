import unittest

from uav_sim.envs.uav_sim import UavSim
from uav_sim.utils.safety_layer import SafetyLayer
import numpy as np
import tempfile
from ray import tune
import os


class TestSafetyLayer(unittest.TestCase):
    def setUp(self):
        self.env = UavSim({"num_uavs": 4, "num_obstacles": 4, "time_final": 10.0})
        self.config = {
            "replay_buffer_size": 4,
            "batch_size": 4,
            "num_epochs": 1,
            "device": "cpu",
            "num_iter_per_epoch": 1,
            "num_training_steps": 1,
            "num_eval_steps": 1,
            "use_rl": False,
        }

        self.sl = SafetyLayer(self.env, config=self.config)

    def test_init(self):
        self.sl._sample_steps(200)
        self.sl._train_batch()

    def test_train(self):
        self.sl.train()

    @unittest.skip
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
        self.env.uavs[0].state[0:3] = np.array([1, 1, 1])
        self.env.uavs[2].state[0:3] = np.array([1, 1, 1])
        # for i in range(self.env.num_uavs):
        # self.env.uavs[i].state[0:3] = np.zeros([1, 1, 1])
        self.sl._env.get_time_coord_action = lambda x: np.zeros(3)
        actions = {i: np.zeros(3) for i in range(self.sl._env.num_uavs)}
        obs, rew, done, _, info = self.sl._env.step(actions)
        self.sl._env.reset = lambda: (obs, info)
        for i in range(2):
            self.sl._sample_steps(1)
            batch = self.sl._replay_buffer.sample(self.sl._batch_size)
            constraints = self.sl._as_tensor(batch["constraint"])
            safe_mask, unsafe_mask, mid_mask = self.sl._get_mask(constraints)

            print(f"batch state: {batch['state']}")
            print(f"constrainsts: {constraints}")
            print(f"safe: {safe_mask}")
            print(f"unsafe: {unsafe_mask}")

    def test_saving(self):
        dir_to_write_to = tempfile.TemporaryDirectory()
        self.config = {
            "replay_buffer_size": 4,
            "batch_size": 4,
            "num_epochs": 2,
            "checkpoint_freq": 1,
            "device": "cpu",
            "num_iter_per_epoch": 1,
            "num_training_steps": 1,
            "num_eval_steps": 1,
            "use_rl": False,
            "tune_run": True,
        }

        def temp_train_sl(config, checkpoint_dir=None):
            if checkpoint_dir:
                config["safety_layer_cfg"]["checkpoint_dir"] = os.path.join(
                    checkpoint_dir, "checkpoint"
                )
            self.sl = SafetyLayer(self.env, config=self.config)
            self.sl.train()

        results = tune.run(
            temp_train_sl,
            stop={
                "training_iteration": self.config["num_epochs"],
                "time_total_s": 60,
            },
            resources_per_trial={"cpu": 1, "gpu": 0},
            config=self.config,
            local_dir=dir_to_write_to.name,
            name="temp_file",
        )

        print(f"temporary directory: {dir_to_write_to.name}")


if __name__ == "__main__":
    unittest.main()
