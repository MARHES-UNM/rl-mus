from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn


class BaseModel(TorchModelV2, nn.Module):
    """_summary_

    Args:
        TorchModelV2 (_type_): _description_
        nn (_type_): _description_
    """

    def __init__(
        self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
        )
        nn.Module.__init__(self)

        self.orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(self.orig_space, gym.spaces.Dict)
            and "observations" in self.orig_space.spaces
        )
        self.hidden_layer_size = model_config["custom_model_config"][
            "hidden_layer_size"
        ]

        self.num_agent_states = model_config["custom_model_config"]["num_agent_states"]
        self.num_obstacle_states = model_config["custom_model_config"][
            "num_obstacle_states"
        ]
        self.num_evader_other_agent_states = model_config["custom_model_config"][
            "num_evader_other_agent_states"
        ]

        # get number of entities in environment
        self.num_evaders = model_config["custom_model_config"]["num_evaders"]
        self.num_obstacles = model_config["custom_model_config"]["num_obstacles"]
        self.num_agents = model_config["custom_model_config"]["num_agents"]

        # max number of entities in environment
        self.max_num_obstacles = model_config["custom_model_config"][
            "max_num_obstacles"
        ]
        self.max_num_agents = model_config["custom_model_config"]["max_num_agents"]
        self.max_num_evaders = model_config["custom_model_config"]["max_num_evaders"]

        self.use_safe_action = model_config["custom_model_config"].get(
            "use_safe_action", False
        )

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
