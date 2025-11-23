import torch as t
import torch.nn as nn
import numpy as np

class ImpalaSmall(nn.Module):
    def __init__(self, num_actions=13):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=8,
            stride=4
        )
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels = 16,
            out_channels=32, 
            kernel_size=4,
            stride=2
        )
        self.activation2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.FC = nn.Linear(2592, 256)
        self.activation3 = nn.ReLU()
        self.feature_extractor = nn.Sequential(
            self.conv1,
            self.activation1,
            self.conv2,
            self.activation2,
            self.flatten,
            self.FC,
            self.activation3
        )
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1) 
        self._initialize_weights()

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.policy_head(x), self.value_head(x)

    def _initialize_weights(self): # TODO: Generalize this to both models & comment better

        """Apply orthogonal intialization to weights with appropiate gain,
        sqrt(2) for convolutional and FC layers, policy head 0.01,
        value_head 1.0"""

        for layer in [self.conv1, self.conv2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        nn.init.orthogonal_(self.FC.weight, gain=np.sqrt(2))
        nn.init.constant_(self.FC.bias, 0)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0)


