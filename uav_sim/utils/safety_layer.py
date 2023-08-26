import os
from datetime import datetime
import random
from time import time

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from uav_sim.utils.replay_buffer import ReplayBuffer
from torch.optim import Adam
import ray.tune as tune

PATH = os.path.dirname(os.path.abspath(__file__))


class SafetyLayer:
    def __init__(self, env, config={}):
        self._env = env

        self._config = config

        self._parse_config()
        random.seed(self._seed)
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        # self._action_dim = self._env.action_space[0].shape[0]

        # orig_space = getattr(
        #     self._env.observation_space[0],
        #     "original_space",
        #     self._env.observation_space[0],
        # )
        # assert (
        #     isinstance(orig_space, gym.spaces.Dict)
        #     and "observations" in orig_space.spaces
        # )
        # self._observation_dim = orig_space["observations"].shape[0]

        # self.model = SafeActionModel(self._observation_dim, self._action_dim)

        self.cbf_model = None
        self.nn_action_model = None
        # self.cbf_model = CBFModel()
        # self.nn_action_model = NNController()

        # TODO: add method to upload saved models
        # if self._checkpoint_dir:
        #     self.model.load_model(self._checkpoint_dir)

        # self._cbf_optimizer = Adam(self.cbf_model.parameters(), lr=self._lr)
        # self._nn_action_optimizer = Adam(self.nn_action_model.parameters(), lr=self._lr)

        self._replay_buffer = ReplayBuffer(self._replay_buffer_size)

        self._train_global_step = 0
        self._eval_global_step = 0

        # # use gpu if available
        # # https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk
        # self._device = "cpu"
        # if torch.cuda.is_available():
        #     print("using cuda")
        #     self._device = "cuda"
        # self.cbf_model.to(self._device)
        # self.nn_action_model.to(self._device)

    def _parse_config(self):
        # TODO: update config inputs
        # default 1000000
        self._replay_buffer_size = self._config.get("replay_buffer_size", 1000000)
        self._episode_length = self._config.get("episode_length", 400)
        self._lr = self._config.get("lr", 0.0001)
        # default 256
        self._batch_size = self._config.get("batch_size", 256)
        self._num_eval_steps = self._config.get("num_eval_step", 1500)
        self._num_steps_per_epoch = self._config.get("num_steps_per_epoch", 6000)
        self._num_epochs = self._config.get("num_epochs", 25)
        self._report_tune = self._config.get("report_tune", False)
        self._seed = self._config.get("seed", 123)
        self._checkpoint_dir = self._config.get("checkpoint_dir", None)

    def _as_tensor(self, ndarray, requires_grad=False):
        tensor = torch.Tensor(ndarray)
        tensor.requires_grad = requires_grad
        tensor = tensor.to(self._device)
        return tensor

    def _sample_steps(self, num_steps):
        episode_length = 0

        obs = self._env.reset()

        for _ in range(num_steps):
            actions = self._env.action_space.sample()
            obs_next, _, done, _ = self._env.step(actions)

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
                self._replay_buffer.add(buffer_dictionary)

            obs = obs_next
            episode_length += 1

            # self._env.render()
            if done["__all__"] or (episode_length == self._episode_length):
                obs = self._env.reset()
                episode_length = 0

    def _get_mask(self, constraints):
        safe_mask = [[arr >= (0.0).any() for arr in constraints]]
        unsafe_mask = [[arr < (0.0).any() for arr in constraints]]

        return safe_mask, unsafe_mask

    def _evaluate_batch(self, batch):
        """Gets the observation and calculate h and action from model.

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        state = self._as_tensor(batch["state"])
        rel_pad = self._as_tensor(batch["rel_pad"])
        other_uav_obs = self._as_tensor(batch["other_uav_obs"])
        obstacles = self._as_tensor(batch["obstacles"])
        constraints = self._as_tensor(batch["constraint"])
        u_nominal = self._as_tensor(batch["action"])
        state_next = self._as_tensor(batch["state_next"])
        other_uav_obs_next = self._as_tensor(batch["other_uav_obs_next"])
        obstacles_next = self._as_tensor(batch["obstacles_next"])

        safe_mask, unsafe_mask = self._get_mask(constraints)

        h = self.model(state, other_uav_obs, obstacles)
        u = self.model(state, rel_pad, other_uav_obs, obstacles, u_nominal)

        # TODO: calculate the the nomimal state using https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9681233
        # state_nominal = state + f_linear(state, u) * self._env.dt

        h_next = self.model(state_next, other_uav_obs_next, obstacles_next)
        h_deriv = (h_next - h) / self._env.dt + h

        h_safe = h[safe_mask]
        h_unsafe = h[unsafe_mask]

        eps_safe = 0.1
        eps_unsafe = 0.1
        eps_deriv = 0.01
        action_weight = 0.01

        loss = (
            0.0
            + (F.relu(eps_safe - h_safe)).mean()
            + (F.relu(eps_unsafe + h_unsafe)).mean()
            + (F.relu(eps_deriv - h_deriv)).mean()
            + action_weight * ((u - u_nominal) ** 2).mean()
        )

        observation = self._as_tensor(batch["observations"])
        target_safe_action = self._as_tensor(batch["safe_action"])

        predict_safe_action = self.model(observation)

        loss = self._loss_function(predict_safe_action, target_safe_action)

        return loss

    def _train_batch(self):
        """Sample batch from replay buffer and calculate loss

        Returns:
            loss function
        """
        batch = self._replay_buffer.sample(self._batch_size)

        # zero parameter gradients
        self._optimizer.zero_grad()

        # forward + backward + optimize
        loss = self._evaluate_batch(batch)
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def evaluate(self):
        """Validation Step"""
        # sample steps
        self._sample_steps(self._num_eval_steps)

        self.model.eval()
        loss = list(
            map(
                lambda x: x.item(),
                [
                    self._evaluate_batch(batch)
                    for batch in self._replay_buffer.get_sequential(self._batch_size)
                ],
            )
        )

        loss = np.mean(np.array(loss))

        self._replay_buffer.clear()

        self._eval_global_step += 1
        self.model.train()

        return loss

    # def get_action(self, obs):
    #     obs_tensor = self._as_tensor(obs)
    #     with torch.no_grad():
    #         return self.model(obs_tensor).cpu().numpy()

    def train(self):
        """Train Step"""
        start_time = time()

        print("==========================================================")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        for epoch in range(self._num_epochs):
            # sample episodes from whole epoch
            self._sample_steps(self._num_steps_per_epoch)

            loss = self._train_batch()
            loss = np.mean(
                np.array(
                    [
                        self._train_batch(batch)
                        for batch in self._replay_buffer.get_sequential(
                            self._batch_size
                        )
                    ]
                )
            )
            self._replay_buffer.clear()
            self._train_global_step += 1
            training_iteration = self._train_global_step

            print(f"Finished epoch {epoch} with loss: {loss}. Running validation ...")

            # val_loss = self.evaluate()
            # print(f"validation completed, average loss {val_loss}")

            # if self._report_tune:
            #     tune.report(
            #         training_loss=loss,
            #         training_iteration=training_iteration,
            #         validation_loss=val_loss,
            #     )

            #     if epoch % 5 == 0:
            #         with tune.checkpoint_dir(epoch) as checkpoint_dir:
            #             path = os.path.join(checkpoint_dir, "checkpoint")
            #             torch.save(
            #                 (self.model.state_dict(), self._optimizer.state_dict()),
            #                 path,
            #             )
        print("==========================================================")
        print(
            f"Finished training constraint model. Time spent: {(time() - start_time) // 1} secs"
        )
        print("==========================================================")
