import os
import gymnasium as gym
import torch

from gymnasium import spaces


def write_checkpoint(
    agent,
    optimizer,
    episode: int,
    dst_path: str,
):
    """
    Writes a checkpoint including agent, optimizer state dictionaries along with current episode statistics
    and ReplayMemory to a file.

    :param agent: The agent and all its state dictionaries
    :param optimizer: The pytorch optimizer
    :param epsiode: Current training epsiode
    :param dst_path: Path where the checkpoint is written to
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(
        obj={
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode + 1,
        },
        f=dst_path,
    )


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, bins=5):
        """A wrapper for converting a 1D continuous actions into discrete ones.
        Args:
            env: The environment to apply the wrapper
            bins: number of discrete actions
        """
        assert isinstance(env.action_space, spaces.Box)
        super().__init__(env)
        self.bins = bins
        self.orig_action_space = env.action_space
        self.action_space = spaces.Discrete(self.bins)

    def action(self, action):
        """discrete actions from low to high in 'bins'
        Args:
            action: The discrete action
        Returns:
            continuous action
        """
        return self.orig_action_space.low + action / (self.bins - 1.0) * (
            self.orig_action_space.high - self.orig_action_space.low
        )
