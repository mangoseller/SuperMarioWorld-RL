import torch as t
import torch.nn as nn
import numpy as np
from model_components import (
    FourierCoordConv,
    ResidualBlock,
    RandomShifts,
    SpatialSoftmax,
    PixelControlHead
)

class TransPala(nn.Module): # TODO Give these less stupid names
    """Wider, more spatial awareness, pixel_control SSL.
    ~3.8 Million Parameters"""

    def __init__(self, num_actions=14):
        super().__init__()
        
        self.aug = RandomShifts(pad=4)
        self.coord_conv = FourierCoordConv(scales=[1.0, 2.0]) 

        # Input: 4 + 8 (2 scales * 4 coords) = 12 channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(12, 48, kernel_size=3, padding=1),
            nn.MaxPool2d(2), # 84 -> 42
            ResidualBlock(48, dilation=1),
            ResidualBlock(48, dilation=1)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.MaxPool2d(2), # 42 -> 21
            ResidualBlock(96, dilation=2),
            ResidualBlock(96, dilation=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            ResidualBlock(192, dilation=4), # Global context
            ResidualBlock(192, dilation=8)  
        )

        # 192 channels * 2 coords = 384 inputs
        self.spatial_softmax = SpatialSoftmax(21, 21)
        
        self.trunk = nn.Sequential(
            nn.Linear(384, 1024),
            nn.LayerNorm(1024),
            nn.SiLU()
        )
        
        self.policy_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(1024, num_actions)
        )
        self.value_head = nn.Linear(1024, 1)
        
        # Pixel Control Head attached to the trunk features
        self.pixel_control_head = PixelControlHead(1024, grid_size=7)
        self._init_weights()

    def forward(self, x, return_pixel_control=False):

        x = self.aug(x)
        x = self.coord_conv(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.spatial_softmax(x)
        x = self.trunk(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        if return_pixel_control:
            pixel_pred = self.pixel_control_head(x)
            return policy, value, pixel_pred
        
        return policy, value

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.orthogonal_(self.pixel_control_head.spatial[-2].weight, gain=1.0)
   
class ImpalaLike(nn.Module):
 # ~1.1 million parameters, no pixel control 
    def __init__(self, num_actions=14):
        super().__init__()
        
        self.aug = RandomShifts(pad=4)
        self.coord_conv = FourierCoordConv(scales=[2.0]) 

        # Input: 4 (Gray) + 4 (Fourier) = 8 channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2), # 84 -> 42
            ResidualBlock(32, dilation=1),
            ResidualBlock(32, dilation=1)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2), # 42 -> 21
            ResidualBlock(64, dilation=2),
            ResidualBlock(64, dilation=2)
        )
        
        # Widened to 128 to hit ~1M params and increase logic capacity
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # No pooling, keep 21x21 for spatial softmax
            ResidualBlock(128, dilation=2),
            ResidualBlock(128, dilation=4) # Wide receptive field
        )

        # 128 channels * 2 coordinates (x,y) = 256 input features
        self.spatial_softmax = SpatialSoftmax(21, 21)
        
        self.trunk = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.SiLU()
        )
        
        self.policy_head = nn.Linear(512, num_actions)
        self.value_head = nn.Linear(512, 1)
        
        self._init_weights()

    def forward(self, x, return_pixel_control=False):
        x = self.aug(x)
        x = self.coord_conv(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.spatial_softmax(x)
        x = self.trunk(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)

        if return_pixel_control:
            # Return dummy if requested by PPO to prevent crashes
            return policy, value, None 

        return policy, value

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)


class ConvolutionalSmall(nn.Module):
    # Very light, ~600k params
    def __init__(self, num_actions=14):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.activation2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.FC = nn.Linear(2592, 256)
        self.activation3 = nn.ReLU()
        self.feature_extractor = nn.Sequential(
            self.conv1, self.activation1,
            self.conv2, self.activation2,
            self.flatten, self.FC, self.activation3
        )
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1) 
        self._initialize_weights()

    def forward(self, x, return_pixel_control=False):
        x = self.feature_extractor(x)
        if return_pixel_control:
            return self.policy_head(x), self.value_head(x), None
        return self.policy_head(x), self.value_head(x)

    def _initialize_weights(self): 

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

