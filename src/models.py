import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
    # Normalize before convolution     
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.se = SEBlock(channels)

    def forward(self, x):

        out = F.silu(self.gn1(x))
        out = self.conv1(out)
        out = F.silu(self.gn2(out))
        out = self.conv2(out)
        out = self.se(out)
        return x + out # Residual

class SEBlock(nn.Module):

# Squeeze and Excitation block
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

class ModelBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_residual=2):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_channels) for _ in range(num_residual)]
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_blocks(x)
        return x

class ImpalaLike(nn.Module):

    def __init__(self, num_actions=14):
        super().__init__()

        self.block1 = ModelBlock(4, 16, num_residual=1)
        self.block2 = ModelBlock(16, 32, num_residual=1)
        self.block3 = ModelBlock(32, 64, num_residual=2)
        
        self.reduce = nn.Conv2d(64, 16, kernel_size=1)
        self.flatten = nn.Flatten()


        self.fc = nn.Linear(1936, 512)
        self.ln = nn.LayerNorm(512)


        self.policy_head = nn.Linear(512, num_actions)
        self.value_head = nn.Linear(512, 1)

        self._initialize_weights()

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.silu(self.reduce(x))
        
        x = self.flatten(x)
        x = F.silu(self.fc(x))
        x = self.ln(x)

        return self.policy_head(x), self.value_head(x)

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)


class ConvolutionalSmall(nn.Module):
    def __init__(self, num_actions=14):
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

    def _initialize_weights(self): 

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


