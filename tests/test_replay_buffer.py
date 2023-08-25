import unittest
import numpy as np
from uav_sim.envs.uav_sim import UavSim
from uav_sim.utils import utils
from uav_sim.utils.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.replay_buffer = ReplayBuffer(5)
        self.env = UavSim()

    def test_adding_buffer(self):
        self.assertEqual(self.replay_buffer._buffer_size, 5)

        actions = self.env.action_space.sample()

        obs = self.env.reset()
        obs_next, reward, done, info = self.env.step(actions)

        for (_, action), (_, observation), (_, observation_next) in zip(
            actions.items(), obs.items(), obs_next.items()
        ):
            buffer_dictionary = {}
            for k, v in observation.items():
                buffer_dictionary[k] = v

            buffer_dictionary["action"] = action

            for k, v in observation_next.items():
                new_key = f"{k}_next"
                buffer_dictionary[new_key] = v

            # print(buffer_dictionary)
            self.replay_buffer.add(buffer_dictionary)

        self.assertEqual(self.replay_buffer._current_index, self.env.num_uavs)

        seq_data = list(self.replay_buffer.get_sequential(1))

        for (i, action), (_, observation), (_, observation_next) in zip(
            actions.items(), obs.items(), obs_next.items()
        ):
            np.testing.assert_almost_equal(seq_data[i]["action"].squeeze(), action)

    def test_sampling_buffer(self):
        self.fill_buffer()

        data = self.replay_buffer.sample(10)

    def fill_buffer(self):
        for _ in range(20):
            obs = self.env.reset()
            actions = self.env.action_space.sample()

            obs_next, reward, done, info = self.env.step(actions)

            for (_, action), (_, observation), (_, observation_next) in zip(
                actions.items(), obs.items(), obs_next.items()
            ):
                buffer_dictionary = {}
                for k, v in observation.items():
                    buffer_dictionary[k] = v

                buffer_dictionary["action"] = action

                for k, v in observation_next.items():
                    new_key = f"{k}_next"
                    buffer_dictionary[new_key] = v

                # print(buffer_dictionary)
                self.replay_buffer.add(buffer_dictionary)


if __name__ == "__main__":
    unittest.main()
