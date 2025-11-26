import torch as t
import torch.nn as nn
import numpy as np

from model_components import (
    FourierCoordConv,
    ModelBlock,
    RandomShifts,
    SpatialAttentionPool,
    PixelControlHead
)

class TransPala(nn.Module):

    def __init__(self, num_actions=14, dropout=0.1):
        super().__init__()

        self.aug = RandomShifts(pad=4)
        # 3 Scales * 4 Coords = 12 Channels
        self.coord_conv = FourierCoordConv(scales=[1.0, 2.0, 4.0])

        # Stem: 4 (GrayStack) + 12 (Fourier) = 16 Input Channels
        self.stem = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.SiLU()
        )

        # Progressive stochastic depth: 0 -> 0.05 -> 0.1 -> 0.15
        self.block1 = ModelBlock(32, 64, num_residual=2, 
                                 drop_prob_start=0.0, drop_prob_end=0.05,
                                 use_spatial_attention=False)

        self.block2 = ModelBlock(64, 128, num_residual=3,
                                 drop_prob_start=0.05, drop_prob_end=0.1,
                                 use_spatial_attention=True)

        self.block3 = ModelBlock(128, 256, num_residual=5,
                                 drop_prob_start=0.1, drop_prob_end=0.15,
                                 use_spatial_attention=True,
                                 use_axial_attention=True)

        # Attention pooling
        self.pool = SpatialAttentionPool(256, num_heads=8, dropout=dropout)

        # [MODIFIED] Reduced trunk size from 2048 to 1024
        self.trunk = nn.Sequential(
            nn.Linear(256, 1024),
            nn.LayerNorm(1024),
            nn.SiLU()
        )

        self.policy_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, num_actions)
        )

        self.value_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 1)
        )
        
        self.pixel_control_head = PixelControlHead(1024, grid_size=7)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.orthogonal_(self.pixel_control_head.spatial[-2].weight, gain=1.0)

    def forward(self, x, return_pixel_control=False):
        x = self.aug(x)
        x = self.coord_conv(x)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # (B, 256, H, W) -> (B, H*W, 256)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        
        x = self.pool(x)
        x = self.trunk(x)

        policy = self.policy_head(x)
        value = self.value_head(x)
        
        if return_pixel_control:
            pixel_pred = self.pixel_control_head(x)
            return policy, value, pixel_pred
        
        return policy, value
    

class ImpalaLike(nn.Module):
    """
    Smaller version for lighter training.
    Updated to include TransPala features (SpatialAttention, PixelControl) while remaining lean.
    """
    def __init__(self, num_actions=14, dropout=0.1):
        super().__init__()
        
        self.aug = RandomShifts(pad=4)
        self.coord_conv = FourierCoordConv(scales=[1.0, 2.0, 4.0])

        self.block1 = ModelBlock(16, 32, num_residual=2)
        self.block2 = ModelBlock(32, 64, num_residual=2)
        
        self.block3 = ModelBlock(64, 128, num_residual=2, use_spatial_attention=True)
        
        self.embed_dim = 128
        
        # Replaced static query attention with TransPala's smarter SpatialAttentionPool
        self.pool = SpatialAttentionPool(self.embed_dim, num_heads=4, dropout=dropout)

        self.trunk = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, num_actions)
        )
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )
        
        self.pixel_control_head = PixelControlHead(512, grid_size=7)

        self._initialize_weights()

    def forward(self, x, return_pixel_control=False):
        x = self.aug(x)
        x = self.coord_conv(x) 

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # (B, 128, H, W) -> (B, H*W, 128)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        
        x = self.pool(x)
        x = self.trunk(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        if return_pixel_control:
            pixel_pred = self.pixel_control_head(x)
            return policy, value, pixel_pred

        return policy, value
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        # [MODIFIED] Init for pixel control
        nn.init.orthogonal_(self.pixel_control_head.spatial[-2].weight, gain=1.0)


class ConvolutionalSmall(nn.Module):
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

    def forward(self, x):
        x = self.feature_extractor(x)
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
