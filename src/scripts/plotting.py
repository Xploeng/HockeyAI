import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("src/")

from collections import defaultdict

def smooth(y, window=10):
    """Apply smoothing to the data"""
    box = np.ones(window) / window
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def plot_rewards_from_memory(checkpoint, output_dir=None, window_size=10):
    """
    Load checkpoints, extract rewards from memory buffer, and plot them
    
    Args:
        output_dir: Directory to save plots (defaults to checkpoint_dir)
        window_size: Window size for smoothing
    """
    if output_dir is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
    
    # Extract memory from the agent state dict
    agent_state_dict = checkpoint['agent_state_dict']
    if 'memory' not in agent_state_dict:
        print("Memory not found in checkpoint")
        return
    
    memory = agent_state_dict['memory']
    
    # Extract rewards from memory
    rewards = []
    for transition in memory.memory:
        if transition is not None and hasattr(transition, 'reward'):
            reward = transition.reward
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            rewards.append(reward)
    
    if not rewards:
        print("No rewards found in memory")
        return
    
    # Calculate episode rewards by grouping consecutive rewards until a done=True is encountered
    episode_rewards = []
    current_episode_reward = 0
    
    for transition in memory.memory:
        if transition is None:
            continue
            
        reward = transition.reward
        done = transition.done
        
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        if isinstance(done, torch.Tensor):
            done = done.item()
            
        current_episode_reward += reward
        
        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0
    
    # Add the last episode if it's not complete
    if current_episode_reward > 0:
        episode_rewards.append(current_episode_reward)
    
    # Plot individual rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.5, label='Individual Rewards')
    
    if len(rewards) > window_size:
        smooth_rewards = smooth(rewards, window=window_size)
        smooth_steps = range(window_size, len(rewards) + 1)
        plt.plot(smooth_steps, smooth_rewards, label=f'Smoothed (window={window_size})')
    
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Individual Rewards During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'individual_rewards.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot episode rewards if available
    if episode_rewards:
        plt.figure(figsize=(10, 6))
        episodes = range(1, len(episode_rewards) + 1)
        plt.plot(episodes, episode_rewards, alpha=0.5, label='Episode Rewards', linewidth=2)
        
        if len(episode_rewards) > window_size:
            smooth_ep_rewards = smooth(episode_rewards, window=window_size)
            smooth_episodes = range(window_size, len(episode_rewards) + 1)
            plt.plot(smooth_episodes, smooth_ep_rewards, label=f'Smoothed (window={window_size})', linewidth=2)
        
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Total Reward', fontsize=14)
        plt.title('', fontsize=18)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(-10, 10)  # Set y-axis limits between -10 and 10
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'episode_rewards.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Calculate and plot reward statistics
    reward_stats = {
        'mean': np.mean(rewards),
        'median': np.median(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'std': np.std(rewards)
    }
    
    print("Reward Statistics:")
    for stat, value in reward_stats.items():
        print(f"{stat}: {value:.4f}")
    
    # Plot reward distribution
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=50, alpha=0.7)
    plt.axvline(reward_stats['mean'], color='r', linestyle='dashed', linewidth=2, label=f"Mean: {reward_stats['mean']:.4f}")
    plt.axvline(reward_stats['median'], color='g', linestyle='dashed', linewidth=2, label=f"Median: {reward_stats['median']:.4f}")
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'reward_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot moving average of rewards over time
    window_sizes = [10, 50, 100]
    plt.figure(figsize=(10, 6))
    
    for ws in window_sizes:
        if len(rewards) > ws:
            smooth_r = smooth(rewards, window=ws)
            smooth_s = range(ws, len(rewards) + 1)
            plt.plot(smooth_s, smooth_r, label=f'Window size: {ws}')
    
    plt.xlabel('Step')
    plt.ylabel('Moving Average Reward')
    plt.title('Reward Moving Averages')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'reward_moving_averages.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}")
    
    return rewards, episode_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot rewards from memory buffer in checkpoints")
    parser.add_argument("--window_size", type=int, default=100, help="Window size for smoothing")
    
    args = parser.parse_args()
    path1 = "/home/tluebbing/workspace/studies/HockeyAI/src/outputs/sac_hockey_weak/checkpoints/sac_hockey_weak_last.ckpt"
    path2 = "/home/tluebbing/workspace/studies/HockeyAI/src/outputs/sac_hockey_weak/episdoe_statistics"
    plot_rewards_from_memory(path1, path2, args.window_size)
