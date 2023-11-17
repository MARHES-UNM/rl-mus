import torch
from torch import nn
import numpy as np


class CBF(nn.Module):
    def __init__(self, n_state, n_hidden=32):
        super(CBF, self).__init__()
        self.n_state = n_state
        self.conv0 = nn.Conv1d(n_state, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        # self.conv4 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 1, 1)
        self.fc0 = nn.Linear(128, 1)
        self.activation = nn.ReLU()

    def forward(self, state, other_uav_obs, obstacles):
        state = torch.unsqueeze(state, 2)  # (bs, n_state, 1)
        other_uav_obs = other_uav_obs.permute(
            0, 2, 1
        )  # (bs, uav_state, num_other_uavs)
        obstacles = obstacles.permute(0, 2, 1)  # (bs, obstacle_state, num_obstacles)

        other_uav_state_diff = (
            state[:, : self.n_state, :] - other_uav_obs[:, : self.n_state, :]
        )
        obstacle_state_diff = (
            state[:, : self.n_state, :] - obstacles[:, : self.n_state, :]
        )

        x = torch.cat((other_uav_state_diff, obstacle_state_diff), dim=2)
        x = self.activation(self.conv0(x))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))  # (bs, 128, n_state)
        x = self.activation(self.conv3(x))
        # x = self.activation(self.conv4(x))
        # x, _ = torch.max(x, dim=2)  # (bs, 128)
        # x = self.fc0(x)
        x = self.conv4(x)
        h = torch.squeeze(x, dim=1)  # (bs, n_state)
        return h


class NN_Action(nn.Module):
    # def __init__(self, n_state, k_obstacle, m_control, preprocess_func=None, output_scale=1.0):
    def __init__(self, n_state, m_control, num_o, n_hidden=32):
        super().__init__()
        self.n_state = n_state

        self.conv0 = nn.Conv1d(n_state, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.fc0_ = nn.Linear(num_o, 1)
        self.fc0 = nn.Linear(128 + m_control + n_state, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, m_control)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()

    def forward(self, state, rel_pad, other_uav_obs, obstacles, u_nominal):
        """
        args:
            state (bs, n_state)
            obstacle (bs, k_obstacle, n_state)
            u_nominal (bs, m_control)
            state_error (bs, n_state)
        returns:
            u (bs, m_control)
        """
        state = torch.unsqueeze(state, 2)  # (bs, n_state, 1)
        other_uav_obs = other_uav_obs.permute(
            0, 2, 1
        )  # (bs, uav_state, num_other_uavs)
        obstacles = obstacles.permute(0, 2, 1)  # (bs, obstacle_state, num_obstacles)

        other_uav_state_diff = (
            state[:, : self.n_state, :] - other_uav_obs[:, : self.n_state, :]
        )
        obstacle_state_diff = (
            state[:, : self.n_state, :] - obstacles[:, : self.n_state, :]
        )

        x = torch.cat((other_uav_state_diff, obstacle_state_diff), dim=2)
        x = self.activation(self.conv0(x))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))  # (bs, 128, k_obstacle)
        # x = torch.squeeze(self.activation(self.fc0_(x)), dim=-1)
        x, _ = torch.max(x, dim=2)  # (bs, 128)
        x = torch.cat([x, u_nominal, rel_pad], dim=1)  # (bs, 128 + m_control)
        x = self.activation(self.fc0(x))
        x = self.activation(self.fc1(x))
        x = self.output_activation(self.fc2(x)) * 5.0
        u = x + u_nominal
        return u
