import os
from datetime import datetime
import random
from time import time

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from torch import nn
from uav_sim.networks.cbf import CBF
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

        self._init_model()

        # TODO: add method to upload saved models
        if self._checkpoint_dir:
            self.model.load_model(self._checkpoint_dir)

        # TODO: load the buffer with the save states
        if self._load_buffer:
            pass

        self._optimizer = Adam(
            self.model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )

        self._replay_buffer = ReplayBuffer(self._replay_buffer_size)

        self._train_global_step = 0
        self._eval_global_step = 0

        # use gpu if available
        # https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk
        self._device = "cpu"
        if torch.cuda.is_available():
            print("using cuda")
            self._device = "cuda"
        self.model.to(self._device)

    def _parse_config(self):
        # TODO: update config inputs
        # default 1000000
        self._replay_buffer_size = self._config.get("replay_buffer_size", 1000000)
        self._episode_length = self._config.get("episode_length", 400)
        self._lr = self._config.get("lr", 0.01)
        self._weight_decay = self._config.get("weight_decay", 1e-5)
        # default 256
        self._batch_size = self._config.get("batch_size", 64)
        self._num_eval_steps = self._config.get("num_eval_steps", 1500)
        self._num_training_steps = self._config.get("num_training_steps", 6000)
        self._num_training_iter = self._config.get("num_training_iter", 25)
        self._report_tune = self._config.get("report_tune", False)
        self._seed = self._config.get("seed", 123)
        self._checkpoint_dir = self._config.get("checkpoint_dir", None)
        self._load_buffer = self._config.get("buffer", None)
        self._n_hidden = self._config.get("n_hidden", 32)
        self.eps = self._config.get("eps", 0.1)
        self.eps = self._config.get("eps", 0.1)
        self.eps_action = self._config.get("eps_action", 0.2)
        self.eps_deriv = self._config.get("eps_deriv", 0.03)
        self.loss_action_weight = self._config.get("loss_action_weight", 0.08)

    def _init_model(self):
        obs_space = self._env.observation_space
        n_state = obs_space[0]["state"].shape[0]
        n_rel_pad_state = obs_space[0]["rel_pad"].shape[0]
        k_obstacle = obs_space[0]["obstacles"].shape[1]  # (num_obstacle, state)
        m_control = self._env.action_space[0].shape[0]

        self.model = CBF(
            n_state, n_rel_pad_state, k_obstacle, m_control, self._n_hidden
        )

    def _as_tensor(self, ndarray, requires_grad=False):
        tensor = torch.Tensor(ndarray)
        tensor.requires_grad = requires_grad
        tensor = tensor.to(self._device)
        return tensor

    # TODO: create static buffer so that we can sample from saved data.
    def _sample_steps(self, num_steps):
        episode_length = 0
        num_episodes = 0

        results = {
            "uav_collision": 0.0,
            "obs_collision": 0.0,
            "uav_done": 0.0,
            "uav_done_time": 0.0,
        }

        obs = self._env.reset()

        for _ in range(num_steps):
            # actions = self._env.action_space.sample()
            actions = {}
            for i in range(self._env.num_uavs):
                nom_action = self._env.get_time_coord_action(
                    self._env.uavs[i]
                ).squeeze()

                actions[i] = self.get_action(obs[i], nom_action)
                # actions[i] = nom_action

            obs_next, _, done, info = self._env.step(actions)

            for k, v in info.items():
                results["uav_collision"] += v["uav_collision"]
                results["obs_collision"] += v["obstacle_collision"]

            for i in range(self._env.num_uavs):
                buffer_dictionary = {}
                for k, v in obs[i].items():
                    buffer_dictionary[k] = v

                buffer_dictionary["u_nominal"] = nom_action

                for k, v in obs_next[i].items():
                    new_key = f"{k}_next"

                    buffer_dictionary[new_key] = v
                self._replay_buffer.add(buffer_dictionary)

            obs = obs_next
            episode_length += 1

            # self._env.render()
            if done["__all__"] or (episode_length == self._episode_length):
                num_episodes += 1
                for k, v in info.items():
                    results["uav_done"] += v["uav_landed"]
                    results["uav_done_time"] += v["uav_done_time"]

                obs = self._env.reset()
                episode_length = 0

        num_episodes += 1
        results["obs_collision"] = (
            results["obs_collision"] / num_episodes / self._env.num_uavs
        )
        results["uav_collision"] = (
            results["uav_collision"] / num_episodes / self._env.num_uavs
        )
        results["uav_done"] = results["uav_done"] / num_episodes / self._env.num_uavs
        results["uav_done_time"] = (
            results["uav_done_time"] / num_episodes / self._env.num_uavs
        )
        results["num_ts_per_episode"] = num_steps / num_episodes
        return results

    def _get_mask(self, constraints):
        safe_mask = self._as_tensor([(arr >= 0.0).any() for arr in constraints]).float()
        unsafe_mask = self._as_tensor(
            [(arr < 0.0).any() for arr in constraints]
        ).float()

        mid_mask = (1 - safe_mask) * (1 - unsafe_mask)

        return safe_mask, unsafe_mask, mid_mask

    def f_dot_torch(self, state, action):
        u = action.clone()
        u[:, 2] = 1 / self._env.uavs[0].m * u[:, 2] - self._env.uavs[0].g

        A = np.zeros((12, 12), dtype=np.float32)
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0

        B = np.zeros((12, 3), dtype=np.float32)
        B[3, 0] = 1.0
        B[4, 1] = 1.0
        B[5, 2] = 1.0
        A_T = self._as_tensor(A.T)
        B_T = self._as_tensor(B.T)

        dxdt = torch.matmul(state, A_T) + torch.matmul(u, B_T)

        return dxdt

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
        u_nominal = self._as_tensor(batch["u_nominal"])
        state_next = self._as_tensor(batch["state_next"])
        rel_pad_next = self._as_tensor(batch["rel_pad_next"])
        other_uav_obs_next = self._as_tensor(batch["other_uav_obs_next"])
        obstacles_next = self._as_tensor(batch["obstacles_next"])

        safe_mask, unsafe_mask, mid_mask = self._get_mask(constraints)

        h, u = self.model(state, rel_pad, other_uav_obs, obstacles, u_nominal)

        # TODO: calculate the the nomimal state using https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9681233
        state_next = state + self.f_dot_torch(state, u) * self._env.dt

        h_next, _ = self.model(
            state_next, rel_pad_next, other_uav_obs_next, obstacles_next, u_nominal
        )
        h_deriv = (h_next - h) / self._env.dt + h

        num_safe = torch.sum(safe_mask)
        num_unsafe = torch.sum(unsafe_mask)
        num_mid = torch.sum(mid_mask)

        loss_h_safe = torch.sum(F.relu(self.eps - h) * safe_mask) / (1e-5 + num_safe)
        loss_h_dang = torch.sum(F.relu(h + self.eps) * unsafe_mask) / (
            1e-5 + num_unsafe
        )

        acc_h_safe = torch.sum((h >= 0).float() * safe_mask) / (1e-5 + num_safe)
        acc_h_dang = torch.sum((h < 0).float() * unsafe_mask) / (1e-5 + num_unsafe)

        loss_deriv_safe = torch.sum(F.relu(self.eps_deriv - h_deriv) * safe_mask) / (
            1e-5 + num_safe
        )
        loss_deriv_dang = torch.sum(F.relu(self.eps_deriv - h_deriv) * unsafe_mask) / (
            1e-5 + num_unsafe
        )
        loss_deriv_mid = torch.sum(F.relu(self.eps_deriv - h_deriv) * mid_mask) / (
            1e-5 + num_mid
        )

        acc_deriv_safe = torch.sum((h_deriv > 0).float() * safe_mask) / (
            1e-5 + num_safe
        )
        acc_deriv_dang = torch.sum((h_deriv > 0).float() * unsafe_mask) / (
            1e-5 + num_unsafe
        )
        acc_deriv_mid = torch.sum((h_deriv > 0).float() * mid_mask) / (1e-5 + num_mid)

        err_action = torch.mean(torch.abs(u - u_nominal))

        loss_action = torch.mean(F.relu(torch.abs(u - u_nominal) - self.eps_action))

        loss = (
            loss_h_safe
            + loss_h_dang
            + loss_deriv_safe
            + loss_deriv_dang
            + loss_deriv_mid
            + loss_action * self.loss_action_weight
        )

        # TODO: use a dictionary to store acc_h_items instead.
        return loss, (
            acc_h_safe.detach().cpu().numpy(),
            acc_h_dang.detach().cpu().numpy(),
            acc_deriv_safe.detach().cpu().numpy(),
            acc_deriv_dang.detach().cpu().numpy(),
            acc_deriv_mid.detach().cpu().numpy(),
            err_action.detach().cpu().numpy(),
        )

    def _train_batch(self):
        """Sample batch from replay buffer and calculate loss

        Returns:
            loss function
        """
        batch = self._replay_buffer.sample(self._batch_size)

        # forward + backward + optimize
        loss, acc_stats = self._evaluate_batch(batch)

        # zero parameter gradients
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss, acc_stats

    def parse_results(self, results):
        loss_array = []
        acc_stat_array = []
        for x in results:
            loss_array.append(x[0].item())
            acc_stat_array.append([acc for acc in x[1]])

        loss = np.array(loss_array).mean()
        acc_stat = np.array(acc_stat_array).mean(axis=0)

        return loss, acc_stat

    def evaluate(self):
        """Validation Step"""
        # sample steps
        self._sample_steps(self._num_eval_steps)

        self.model.eval()

        eval_results = [
            self._evaluate_batch(batch)
            for batch in self._replay_buffer.get_sequential(self._batch_size)
        ]

        loss, acc_stat = self.parse_results(eval_results)

        self._replay_buffer.clear()

        self._eval_global_step += 1
        self.model.train()

        return loss, acc_stat

    def get_action(self, obs, action):
        state = torch.unsqueeze(self._as_tensor(obs["state"]), dim=0)
        rel_pad = torch.unsqueeze(self._as_tensor(obs["rel_pad"]), dim=0)
        other_uav_obs = torch.unsqueeze(self._as_tensor(obs["other_uav_obs"]), dim=0)
        obstacles = torch.unsqueeze(self._as_tensor(obs["obstacles"]), dim=0)
        constraint = torch.unsqueeze(self._as_tensor(obs["constraint"]), dim=0)
        u_nominal = torch.unsqueeze(self._as_tensor(action.squeeze()), dim=0)
        with torch.no_grad():
            _, u = self.model(state, rel_pad, other_uav_obs, obstacles, u_nominal)

            return u.detach().cpu().numpy().squeeze()

    def train(self):
        """Train Step"""
        start_time = time()

        print("==========================================================")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        for training_iter in range(self._num_training_iter):
            # sample episodes for training iteration
            sample_stats = self._sample_steps(self._num_training_steps)

            # iterate through the buffer and get batches at a time
            train_results = [
                self._train_batch()
                for _ in range(self._num_training_steps // self._batch_size)
            ]

            loss, train_acc_stats = self.parse_results(train_results)
            self._replay_buffer.clear()
            self._train_global_step += 1

            print(
                f"Finished training iter {training_iter} with loss: {loss}. Running validation ..."
            )

            val_loss, val_acc_stats = self.evaluate()
            print(f"validation completed, average loss {val_loss}")

            # TODO: fix report h items
            if self._report_tune:
                tune.report(
                    training_iteration=self._train_global_step,
                    train_loss=loss,
                    train_acc_h_safe=train_acc_stats[0],
                    train_acc_h_dang=train_acc_stats[1],
                    train_acc_h_deriv_safe=train_acc_stats[2],
                    train_acc_h_deriv_dang=train_acc_stats[3],
                    train_acc_h_deriv_mid=train_acc_stats[4],
                    train_err_action=train_acc_stats[5],
                    val_loss=val_loss,
                    val_acc_h_safe=val_acc_stats[0],
                    val_acc_h_dang=val_acc_stats[1],
                    val_acc_h_deriv_safe=val_acc_stats[2],
                    val_acc_h_deriv_dang=val_acc_stats[3],
                    val_acc_h_deriv_mid=val_acc_stats[4],
                    val_err_action=val_acc_stats[5],
                    **sample_stats,
                )

                with tune.checkpoint_dir(step=training_iter) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(
                        (self.model.state_dict(), self._optimizer.state_dict()),
                        path,
                    )

        print("==========================================================")
        print(
            f"Finished training constraint model. Time spent: {(time() - start_time) // 1} secs"
        )
        print("==========================================================")
