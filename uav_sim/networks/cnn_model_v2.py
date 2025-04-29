from uav_sim.networks.base_model import BaseModel

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from torch import nn
import torch


class TorchCnnModel(BaseModel):
    def __init__(
        self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
    ):
        BaseModel.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
        )

        self.n_agent_state = model_config["custom_model_config"]["n_agent_state"]
        self.max_action_val = model_config["custom_model_config"]["max_action_val"]
        # model based on this.
        # https://arxiv.org/pdf/1611.06455
        self.conv0 = nn.Conv1d(self.n_agent_state, 128, 1)
        self.bn_conv0 = nn.BatchNorm1d(128)
        self.conv1 = nn.Conv1d(128, 256, 1)
        self.bn_conv1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, 1)
        self.bn_conv2 = nn.BatchNorm1d(128)
        # self.conv3 = nn.Conv1d(256,512, 1)
        # TODO: create encoder decoder architecture here
        # self.conv0 = nn.Conv1d(self.n_agent_state, 32, 1)
        # self.conv1 = nn.Conv1d(32, 64, 1)
        # self.conv2 = nn.Conv1d(64, 128, 1)
        # self.conv3 = nn.Conv1d(512, 256, 1)
        # self.conv3 = nn.Conv1d(256, 128, 1)
        # self.conv3 = nn.Conv1d(256, 512, 1)
        # self.conv3 = nn.Conv1d(512, 1024, 1)
        # to concat obs and other_uavs + agent_state + target_state + done_dt
        self.fc0 = nn.Linear(128 + self.n_agent_state + self.n_agent_state + 1, 128)
        self.fc1 = nn.Linear(128, 128)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()

        self.policy_fn = nn.Linear(128, num_outputs)
        self.value_fn = nn.Linear(128, 1)
        self._value_out = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):

        agent_state = input_dict["obs"]["state"]
        other_uav_state = input_dict["obs"]["other_uav_obs"]
        obstacles = input_dict["obs"]["obstacles"]
        target_state = input_dict["obs"]["rel_pad"]
        done_dt = input_dict["obs"]["dt_go_error"]

        agent_state = torch.unsqueeze(agent_state, 2)  # (bs, n_agent_state, 1)
        other_uav_state = other_uav_state.permute(
            0, 2, 1
        )  # (bs, n_agent_state, num_other_uavs)
        obstacles = obstacles.permute(0, 2, 1)  # (bs, n_agent_state, num_obstacles)

        other_uav_state_diff = (
            agent_state[:, : self.n_agent_state, :]
            - other_uav_state[:, : self.n_agent_state, :]
        )
        obstacle_state_diff = (
            agent_state[:, : self.n_agent_state, :]
            - obstacles[:, : self.n_agent_state, :]
        )

        x = torch.cat((other_uav_state_diff, obstacle_state_diff), dim=2)
        x = self.activation(self.bn_conv0(self.conv0(x)))
        x = self.activation(self.bn_conv1(self.conv1(x)))
        x = self.activation(self.bn_conv2(self.conv2(x)))
        # apply a pooling function to squash observation
        x, _ = torch.max(x, dim=2)  # (bs, 128)
        x = torch.cat((x, torch.squeeze(agent_state, -1), target_state, done_dt), dim=1)
        x = self.activation(self.fc0(x))
        x = self.activation(self.fc1(x))

        # Save for value function
        self._value_out = self.value_fn(x)
        logits = self.output_activation(self.policy_fn(x)) * self.max_action_val

        return logits, state

    @override(ModelV2)
    def value_function(self):
        return self._value_out.flatten()
