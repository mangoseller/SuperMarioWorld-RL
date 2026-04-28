import io
import torch as t
import torch.nn as nn

from muzero.dynamics import DynamicsHead, DynamicsModel
from muzero.encoder import MuZeroEncoder
from muzero.heads import PolicyHead, RewardHead, ValueHead


def serialize_worker_weights(weights):
    buffer = io.BytesIO()
    t.save(weights, buffer)
    return buffer.getvalue()


def deserialize_worker_weights(payload):
    if isinstance(payload, (bytes, bytearray)):
        return t.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    return payload


class MuZeroNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = MuZeroEncoder(config)
        self.dynamics = DynamicsModel(config)
        self.dynamics_head = DynamicsHead(config)
        self.policy = PolicyHead(config)
        self.value = ValueHead(config)
        self.reward = RewardHead(config)

    def weights_for_workers(self):
        return {
            "encoder": {
                key: value.detach().cpu()
                for key, value in self.encoder.state_dict().items()
            },
            "dynamics": {
                key: value.detach().cpu()
                for key, value in self.dynamics.state_dict().items()
            },
            "policy": {
                key: value.detach().cpu()
                for key, value in self.policy.state_dict().items()
            },
            "value": {
                key: value.detach().cpu()
                for key, value in self.value.state_dict().items()
            },
            "reward": {
                key: value.detach().cpu()
                for key, value in self.reward.state_dict().items()
            },
        }
