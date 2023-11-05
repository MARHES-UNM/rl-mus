import json
from tkinter import W
import unittest
import numpy as np
from uav_sim.envs.curriculum_uav_sim import CurriculumEnv
from pathlib import Path
from ray.rllib.utils import check_env

current_path = Path(__file__).parent.absolute().resolve()


class TestUtils(unittest.TestCase):
    def setUp(self):
        with open(
            current_path / ".." / "configs/sim_config.cfg",
            "rt",
        ) as f:
            self.config = json.load(f)
        self.cur_uav_sim = CurriculumEnv(self.config["env_config"])

    def test_reset(self):
        check_env(self.cur_uav_sim)
        obs, info = self.cur_uav_sim.reset()
        self.assertEqual(self.cur_uav_sim.env._beta, 1)


if __name__ == "__main__":
    unittest.main()
