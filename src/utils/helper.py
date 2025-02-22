import os
import sys
import threading
import warnings

from copy import deepcopy
import gymnasium as gym
import numpy as np
import torch

from gymnasium import spaces
from icecream import ic


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
        print(f"Loading checkpoint from {checkpoint_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        agent.load_state_dict(**checkpoint)
        return checkpoint.get("episode", 0)
    else:
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

class OpponentWrapper:
    def __init__(self, opponent, env):
        self.env = env
        self.opponent = opponent

        from agents.ddpg import DDPG
        from agents.rainbow import Rainbow
        from agents.sac import SAC

        if isinstance(opponent, DDPG):
            self.opp_type = "ddpg"
        elif isinstance(opponent, SAC):
            self.opp_type = "sac"
        elif isinstance(opponent, Rainbow):
            self.opp_type = "rainbow"
        else:
            self.opp_type = "basic"

    def act(self, state: torch.Tensor):
        """
        Returns the opponent's action for a given state.

        For basic opponents, if the state is a torch.Tensor, it is converted
        to a NumPy array. For agent opponents, appropriate conversions and
        method calls (like select_action) are performed.
        """
        if self.opp_type == "basic":
            return self._act_basic(state)
        elif self.opp_type == "rainbow":
            return self._act_rainbow(state)
        elif self.opp_type == "ddpg" or self.opp_type == "sac":
            return self._act_ddpg_sac(state)
        else:
            raise ValueError(f"Unsupported opponent type: {self.opp_type}")

    def _act_basic(self, state):
        """Selects an action for a basic opponent. Requires a NumPy array."""
        if isinstance(state, torch.Tensor):
            state = state.squeeze().cpu().numpy()

        return self.opponent.act(state)

    def _act_rainbow(self, state):
        """Selects an action for an agent opponent."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.opponent.device)
        action = self.opponent.select_action(state)
        # ic(action)
        # action = self.env.discrete_to_continous_action(action.item())
        action = [self.env.discrete_to_continous_action(a.item()) for a in action]
        action = torch.tensor(action, dtype=torch.float32, device=self.opponent.device).squeeze()
        return action

    def _act_ddpg_sac(self, state):
        """Selects an action for an agent opponent."""
        return self.opponent.select_action(state)

    # def act(self, state: torch.Tensor):
    #     action = None

    #     if self.opp_type == 'basic':
    #         if isinstance(state, torch.Tensor): # rainbow state representation
    #             action = self.opponent.act(state.squeeze().cpu().numpy())
    #         else: # ddpg state representation
    #             action = self.opponent.act(state)

    #     elif self.opp_type == 'agent':
    #         if isinstance(self.opponent, Rainbow):
    #             if not isinstance(state, torch.Tensor):
    #                 state = torch.tensor(state, dtype=torch.float32, device=self.opponent.device)
    #             action = self.opponent.select_action(state)
    #             # ic(action)
    #             # action = self.env.discrete_to_continous_action(action.item())
    #             action = [self.env.discrete_to_continous_action(a.item()) for a in action]
    #             action = torch.tensor(action, dtype=torch.float32, device=self.opponent.device).squeeze()
    #         elif isinstance(self.opponent, DDPG):
    #             action = self.opponent.select_action(state)

    #     return action
