
from environment import env
import torch as t
from model_small import ImpalaSmall

test_env = env.reset()
print(f"Initial state shape: {test_env['pixels'].shape}")

for i in range(10):
    action = t.tensor(env.action_space.sample())
    test_env["action"] = action
    test_env = env.step(test_env)
    print(f"Step {i}: reward={test_env['next']['reward'].item():.1f}, done={test_env['next']['done'].item()}")

model = ImpalaSmall()
test_env = env.reset()
state = test_env['pixels']

with t.no_grad():
    policy, value = model(state.unsqueeze(0))
    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
