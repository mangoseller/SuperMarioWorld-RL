import torch as t
import torch.nn as nn

class ImpalaSmall(nn.Module):
    def __init__(self, num_actions=13):
        super().__init__()
        # input is (4x84x84) 4 frame stacking greyscale
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

    def forward(self, x):
        x = x / 255 # Normalize pixels
        x = self.feature_extractor(x)
        policy = t.softmax(self.policy_head(x), dim=1)
        return policy, self.value_head(x)


if __name__ == "__main__":
    model = ImpalaSmall()
    test_input = t.randn(1, 4, 84, 84) * 255  # Simulate pixel values
    policy, value = model(test_input)
    print(f"Policy shape: {policy.shape}")  # Should be (1, num_actions)
    print(f"Value shape: {value.shape}")    # Should be (1, 1) - can squeeze this to (1,) if needed
