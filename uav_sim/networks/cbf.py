import torch

from torch import nn
import numpy as np


class CBF(nn.Module):
    """https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html


    Args:
        nn (_type_): _description_
    """

    def __init__(self, n_state, n_rel_pad, k_obstacle, m_control, n_hidden):
        super(CBF, self).__init__()

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
            nn.Linear(
                n_state
                + n_rel_pad
                + n_hidden
                + n_hidden,  # uav_state + other_uav_states + obstacles
                n_hidden,
            ),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        self.h_fn = nn.Linear(n_hidden, 1)
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

        h_out = self.h_fn(x)

        action_out = torch.cat((x, u_nominal), dim=1)
        action_out = self.action_fn(action_out)

        return h_out, action_out

    def load_model(self, model_dir, device="cpu"):
        model_state, _ = torch.load(model_dir, map_location=torch.device(device))
        self.load_state_dict(model_state)
