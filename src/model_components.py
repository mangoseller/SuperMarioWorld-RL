import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat, rearrange, einsum

class FourierCoordConv(nn.Module):
    # Concatenate spacial/coordinate information to each pixel on the screen. Fourier features more expressive than raw (x, y) values

    def __init__(self, scales=[1.0, 2.0, 4.0]):
        super().__init__()
        self.scales = scales
        
    def forward(self, x):
        b, c, h, w = x.shape
        coord_channels = []
        
        # Create coordinate grid
        y_range = t.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        x_range = t.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        y_coords = repeat(y_range, 'h -> h w', w=w)
        x_coords = repeat(x_range, 'w -> h w', h=h)
       
        # Create 4 channels for each scale - positional encoding
        for scale in self.scales:
            coord_channels.extend([
                t.sin(scale * np.pi * y_coords),
                t.cos(scale * np.pi * y_coords),
                t.sin(scale * np.pi * x_coords),
                t.cos(scale * np.pi * x_coords)
            ])
        
        # Combine and batch
        coords = t.stack(coord_channels, dim=0)
        coords = repeat(coords, 'c h w -> b c h w', b=b)

        return t.cat([x, coords], dim=1)

class ResidualBlock(nn.Module):
    """Pre-activation ResNet block
       Supports kernel expansion to increase receptive field without adding parameters"""

    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels) # GroupNorm is more accurate and stable than batch norm for small batches
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x):

        residual = x
        out = F.silu(self.gn1(x))
        out = self.conv1(out)
        out = F.silu(self.gn2(out))
        out = self.conv2(out)

        return out + residual

class SpatialSoftmax(nn.Module):

    """Convert feature maps into (x, y) coordinates of the most activated spots.
    Significantly cheaper than Attention, but provides precise spatial reasoning.
    Avoids destroying spatial information when transitioning from convolutional layers to FC layers."""

    def __init__(self, height, width):

        super().__init__()

        self.height = height
        self.width = width
        
        # Create coordinate grids 
        y_grid, x_grid = t.meshgrid(
            t.linspace(-1., 1., self.height),
            t.linspace(-1., 1., self.width),
            indexing='ij'
        )

        # Register as buffers and flatten
        self.register_buffer('x_grid', x_grid.flatten())
        self.register_buffer('y_grid', y_grid.flatten())

    def forward(self, x):
        # x: (B, C, H, W)

        # Flatten spatial dims into position dim
        x = rearrange(x, 'b c h w -> b c (h w)')
        probs = F.softmax(x, dim=-1)
        
        # Compute evaluated value of coord - (sum of prob * coord)
        get_expected_value = lambda coord: einsum(probs, coord, 'b c pos, pos -> b c')

        expected_x = get_expected_value(self.x_grid)
        expected_y = get_expected_value(self.y_grid)

        return t.cat([expected_x, expected_y], dim=-1)

class PixelControlHead(nn.Module):

    # SSL task to predict pixel changes to force the agent to learn in-game physics.

    def __init__(self, input_dim, grid_size=7):

        super().__init__()
        self.grid_size = grid_size
        self.channels = 32
        
        self.deconv = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 32 * self.grid_size * self.grid_size)
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Softplus() 
        )
        
    def forward(self, features):
        # features: (b, input_dim)
        x = self.deconv(features)
        
        # Unpack x into spatial feature maps
        x = rearrange(x, 'b (c h w) -> b c h w', c=self.channels, h=self.grid_size)
        x = self.spatial(x)

        # Remove channel dim
        return rearrange(x, 'b 1 h w -> b h w')


class RandomShifts(nn.Module):
    # Data augmentation, but carefully so as to not corrupt the state fundamentally 
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):

        if not self.training:
            return x
        n, c, h, w = x.size()
        
        # Pad image - replicate avoids producing black borders
        x_padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), 'replicate')
        
        # Sample a grid from the padded image
        h_pad = h + 2 * self.pad
        w_pad = w + 2 * self.pad

        eps = 1.0 / h_pad
        arange = t.linspace(-1.0 + eps, 1.0 - eps, h_pad, device=x.device, dtype=x.dtype)[:h] 

        # Create mesh grid
        base_grid = t.stack([
            repeat(arange, 'w -> h w', h=h),
            repeat(arange, 'h -> h w', w=w)
        ], dim=-1)

        # Add batch dim
        base_grid = repeat(base_grid, 'h w c -> n h w c', n=n)

        # Generate Random shifts - for each image, generate a single shift
        shift = t.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)

        # Normalize shift to [-1, 1] range based on padded dimensions
        shift *= 2.0 / h_pad
        # Apply shift
        grid = base_grid + shift
        
        return F.grid_sample(x_padded, grid, padding_mode='zeros', align_corners=False, mode='nearest')



