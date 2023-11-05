from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn


class BaseModel(TorchModelV2, nn.Module):
    """"""

    def __init__(
        self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
        )
        nn.Module.__init__(self)

        self.conv0 = nn.Conv1d(n_state, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.fc0 = nn.Linear(128 + m_control + n_state, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, m_control)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()

        self.policy_fn = nn.Linear(64, num_outputs)
        self.value_fn = nn.Linear(64, 1)

    def forward(self, input_dict, state, seq_lens):
        raise NotImplementedError

    def value_function(self):
        raise NotImplementedError
