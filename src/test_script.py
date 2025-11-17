import torch as t
from environment import make_training_env
import time

def test_parallel_resets(num_envs=4, max_steps=10000):
    """Test that parallel envs reset correctly on death"""
    env = make_training_env(num_envs=num_envs)
    environment = env.reset()
    state = environment['pixels']
    
    if num_envs == 1 and state.dim() == 3:
        state = state.unsqueeze(0)
    
    episodes_completed = [0] * num_envs
    episode_lengths = [0] * num_envs
    all_episode_lengths = [[] for _ in range(num_envs)]
    
    print(f"Testing {num_envs} parallel environments...")
    
    for step in range(max_steps):
        action = t.randint(0, 13, (num_envs,))
        action_onehot = t.nn.functional.one_hot(action, num_classes=13).float()
        
        environment["action"] = action_onehot.squeeze(0) if num_envs == 1 else action_onehot
        environment = env.step(environment)
        
        dones = environment["next"]["done"]
        if dones.dim() == 0:
            dones = dones.unsqueeze(0)
        
        # Track episode lengths
        for i in range(num_envs):
            episode_lengths[i] += 1
            
            if dones[i].item():
                episodes_completed[i] += 1
                all_episode_lengths[i].append(episode_lengths[i])
                episode_lengths[i] = 0
                print(f"  Env {i}: Episode {episodes_completed[i]} completed "
                      f"(length: {all_episode_lengths[i][-1]})")
        
        # Stop when each env has completed at least 3 episodes
        if all(count >= 3 for count in episodes_completed):
            break
    
    # Report results
    print(f"\n{'='*50}")
    print(f"Test completed after {step+1} steps")
    for i in range(num_envs):
        avg_length = sum(all_episode_lengths[i]) / len(all_episode_lengths[i]) if all_episode_lengths[i] else 0
        print(f"Env {i}: {episodes_completed[i]} episodes, avg length: {avg_length:.1f}")
        print(f"  Lengths: {all_episode_lengths[i]}")
    
    # Assertions to verify correct behavior
    assert all(count >= 3 for count in episodes_completed), \
        f"Not all envs completed 3 episodes: {episodes_completed}"
    
    # Check that episode lengths are reasonable (not stuck)
    for i, lengths in enumerate(all_episode_lengths):
        assert all(length < 5000 for length in lengths), \
            f"Env {i} has suspiciously long episodes: {lengths}"
        assert all(length > 0 for length in lengths), \
            f"Env {i} has zero-length episodes: {lengths}"
    
    print(f"\n✓ All checks passed!")
    env.close()

if __name__ == "__main__":
    # print("Testing single env...")
    # test_parallel_resets(num_envs=1, max_steps=5000)
    
    print("\n" + "="*50)
    print("Testing 4 parallel envs...")
    test_parallel_resets(num_envs=4, max_steps=10000)
