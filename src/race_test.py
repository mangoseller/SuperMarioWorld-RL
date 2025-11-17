
import torch as t
from environment import make_training_env
import time

def test_for_race_condition(num_envs=4, max_steps=10000, num_trials=5):
    """
    Test for the race condition where episodes complete with length 1 repeatedly.
    Returns True if race condition is detected, False if tests pass.
    """
    
    print(f"Testing for race condition with {num_envs} parallel environments...")
    print(f"Running {num_trials} trials of up to {max_steps} steps each\n")
    
    race_condition_detected = False
    
    for trial in range(2, num_trials):
        print(f"Trial {trial + 1}/{num_trials}")
        
        env = make_training_env(num_envs=num_envs)
        environment = env.reset()
        state = environment['pixels']
        
        if num_envs == 1 and state.dim() == 3:
            state = state.unsqueeze(0)
        
        episodes_completed = [0] * num_envs
        episode_lengths = [0] * num_envs
        all_episode_lengths = [[] for _ in range(num_envs)]
        consecutive_short_episodes = [0] * num_envs  # Track consecutive length-1 episodes
        
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
                    
                    # Check for consecutive short episodes (potential race condition)
                    if episode_lengths[i] == 1:
                        consecutive_short_episodes[i] += 1
                        if consecutive_short_episodes[i] >= 5:  # 5 consecutive length-1 episodes
                            print(f"  ⚠️  Race condition detected in Env {i}!")
                            print(f"      {consecutive_short_episodes[i]} consecutive episodes with length 1")
                            race_condition_detected = True
                    else:
                        consecutive_short_episodes[i] = 0  # Reset counter
                    
                    episode_lengths[i] = 0
            
            # Stop when each env has completed at least 3 episodes
            if all(count >= 3 for count in episodes_completed):
                break
        
        env.close()
        
        # Report trial results
        print(f"  Trial {trial + 1} completed:")
        for i in range(num_envs):
            if all_episode_lengths[i]:
                avg_length = sum(all_episode_lengths[i]) / len(all_episode_lengths[i])
                print(f"    Env {i}: {episodes_completed[i]} episodes, avg length: {avg_length:.1f}")
                
                # Check for suspiciously many length-1 episodes
                short_episodes = sum(1 for length in all_episode_lengths[i] if length == 1)
                if short_episodes > len(all_episode_lengths[i]) * 0.5:  # More than 50% are length 1
                    print(f"      ⚠️  Warning: {short_episodes}/{len(all_episode_lengths[i])} episodes had length 1")
        
        if race_condition_detected:
            print(f"  ❌ Trial {trial + 1} FAILED - Race condition detected!\n")
            break
        else:
            print(f"  ✅ Trial {trial + 1} PASSED\n")
    
    # Final result
    print("=" * 60)
    if race_condition_detected:
        print("❌ RACE CONDITION DETECTED - The bug is still present!")
        print("   Episodes are terminating immediately after reset.")
        return True
    else:
        print("✅ ALL TESTS PASSED - No race condition detected!")
        print("   The HandleMarioLifeLoss wrapper appears to be working correctly.")
        return False

def main():
    # Test with different numbers of parallel environments
    configs = [
        # (1, 5000, 3),   # Single env, shorter test
        (4, 10000, 5),  # 4 parallel envs, standard test
        (8, 10000, 3),  # 8 parallel envs
    ]
    
    all_passed = True
    
    for num_envs, max_steps, num_trials in configs:
        print("\n" + "=" * 60)
        print(f"Testing configuration: {num_envs} environments")
        print("=" * 60)
        
        if test_for_race_condition(num_envs, max_steps, num_trials):
            all_passed = False
            break
    
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    if all_passed:
        print("✅ All configurations passed - the fix appears to be working!")
    else:
        print("❌ Race condition still present - further debugging needed")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
