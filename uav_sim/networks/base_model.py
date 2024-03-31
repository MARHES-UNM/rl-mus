from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
import gymnasium as gym



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
            and "state" in self.orig_space.spaces
            and "rel_pad" in self.orig_space.spaces
            and "other_uav_obs" in self.orig_space.spaces
            and "obstacles" in self.orig_space.spaces
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        raise NotImplementedError

    @override(ModelV2)
    def value_function(self):
        raise NotImplementedError
