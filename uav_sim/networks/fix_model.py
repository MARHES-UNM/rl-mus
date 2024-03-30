from uav_sim.networks.base_model import BaseModel

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override

class TorchFixModel(BaseModel):
    def __init__(
        self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
    ):
        BaseModel.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
        )

        # Base of the model
        self.model = TorchFC(obs_space, act_space, num_outputs, model_config, name)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()
