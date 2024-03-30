from torch import nn
from uav_sim.networks.base_model import BaseModel

class TorchFixModel(BaseModel):
    def __init__(
        self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
    ):
        BaseModel.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
        )

        self.in_layer = nn.Sequential(
            nn.Linear(self.orig_space["observations"].shape[0], self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
        )

        self.policy_fn = nn.Linear(self.hidden_layer_size, num_outputs)
        self.value_fn = nn.Linear(self.hidden_layer_size, 1)

    def forward(self, input_dict, state, seq_lens):
        x = self.in_layer(input_dict["obs"]["observations"])
        self._value_out = self.value_fn(x)
        logits = self.policy_fn(x)

        return logits, state

    def value_function(self):
        return self._value_out.flatten()
