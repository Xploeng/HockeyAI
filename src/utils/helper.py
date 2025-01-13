import os
import threading
import warnings

from copy import deepcopy
import gymnasium as gym
import torch

from gymnasium import spaces


def load_checkpoint(cfg, agent, checkpoint_path, device):
    """
    Loads a checkpoint for the given agent if it exists and continues training is enabled.

    Args:
        cfg (object): Configuration object containing training settings.
        agent (torch.nn.Module): The agent whose state is to be loaded.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): The device on which to load the checkpoint.

    Returns:
        int: The episode number from which to continue training, or 0 if no checkpoint is loaded.
    """
    if cfg.agent.training.continue_training and os.path.exists(checkpoint_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.load_state_dict(**checkpoint)
        return checkpoint.get("episode", 0)
    return 0


def save_checkpoint(agent, checkpoint_path, episode):
    """
    Saves a checkpoint of the agent's state.

    Args:
        agent (object): The agent whose state is to be saved.
        checkpoint_path (str): The file path where the checkpoint will be saved.
        episode (int): The current episode number.

    Returns:
        None
    """
    agent_cp = deepcopy(agent)
    thread = threading.Thread(
        target=write_checkpoint,
        args=(agent_cp, episode, checkpoint_path),
    )
    thread.start()
    thread.join()


def write_checkpoint(
    agent,
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
