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
        self.conv4 = nn.Conv1d(128, 1, 1)
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
        x = self.conv4(x)
        h = torch.squeeze(x, dim=1)  # (bs, n_state)
        return h


class NN_Action(nn.Module):
    """https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html


    Args:
        nn (_type_): _description_
    """

    def __init__(self, n_state, n_rel_pad, k_obstacle, m_control, n_hidden):
        super(NN_Action, self).__init__()

        self.phi_other_uav_obs = nn.Sequential(
            nn.Linear(n_state, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )
        self.rho_other_uav_obs = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU())

        self.phi_obstacles = nn.Sequential(
            nn.Linear(k_obstacle, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )
        self.rho_obstacles = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU())

        self.last_state = nn.Sequential(
            # uav_state + rel_pad_state + other_uav_states + obstacles
            nn.Linear(
                n_state + n_rel_pad + n_hidden + n_hidden,
                n_hidden,
            ),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        # # self.h_fn = nn.Linear(n_hidden, 1)
        # self.h_fn = nn.Sequential(
        #     nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1)
        # )
        self.action_fn = nn.Sequential(
            nn.Linear(n_hidden + m_control, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, m_control),
        )

        # TODO: fix init weights
        # self.init_weights()

    def init_weights(self):
        """
        https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
        """
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, state, rel_pad, other_uav_obs, obstacles, u_nominal):
        """

        Args:
            state (_type_): (bs, n_state)
            rel_pad (_type_): (bs, )
            other_uav_obs (_type_): (bs, n_other_uav_obs, state)
            obstacles (_type_): _description_
            u_nominal (_type_): _description_

        Returns:
            _type_: _description_
        """
        x_other_uav_obs = self.phi_other_uav_obs(
            other_uav_obs
        )  # (bs, n_hidden, n_other_uavs)
        x_other_uav_obs = torch.amax(x_other_uav_obs, dim=1)  # (bs, n_hidden)
        x_other_uav_obs = self.rho_other_uav_obs(x_other_uav_obs)

        x_obstacles = self.phi_obstacles(obstacles)  # (bs, n_hidden, k_obstacle)
        x_obstacles = torch.amax(x_obstacles, dim=1)  # (bs, n_hidden)
        x_obstacles = self.rho_obstacles(x_obstacles)  # (bs, n_hidden)

        x = torch.cat((state, rel_pad, x_other_uav_obs, x_obstacles), dim=1)
        x = self.last_state(x)  # (bs, n_hidden)

        # h_out = self.h_fn(x)
        # h_out = torch.squeeze(h_out, dim=1)

        action_out = torch.cat((x, u_nominal), dim=1)
        action_out = self.action_fn(action_out)

        # return h_out, action_out
        return action_out
