import os
import threading
import warnings

from copy import deepcopy
import gymnasium as gym
import torch

from gymnasium import spaces

from agents.agent import Agent

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
    def __init__(self, opponent, env, requires_continues_action_space, device):
        self.env = env
        self.opponent = opponent
        self.requires_continues_action_space = requires_continues_action_space
        self.device = device
        
        self.opp_type = 'agent' if isinstance(opponent, Agent) else 'basic'
        
    def act(self, state: torch.Tensor):
        action = None
        if self.opp_type == 'basic':
            action = self.opponent.act(state)
        elif self.opp_type == 'agent':
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.opponent.select_action(state)
            if not self.requires_continues_action_space:
                action = self.env.discrete_to_continous_action(action.item())
            else:
                action = action.squeeze().cpu().numpy()
        return action
    
def discrete_to_continuous_action(discrete_action, bins, keep_mode=False):
    """Converts a discrete action index into a smooth continuous action vector.
    
    If `keep_mode` is True, an additional binary shooting action is included.

    Args:
        discrete_action (int): The index of the discrete action.
        bins (int): Number of bins per action dimension.
        keep_mode (bool): Whether to include a 4th dimension for shooting.

    Returns:
        list: A list of 3 (or 4 if `keep_mode=True`) continuous values in range [-1, 1], with the last being binary (0 or 1).
    
    Raises:
        ValueError: If discrete_action is out of range.
    """
    if bins < 1:
        raise ValueError("Bins must be at least 1.")

    # Adjust bin size if keep_mode is enabled (adds an extra binary dimension)
    if keep_mode:
        total_bins = bins ** 3 * 2  # Extra factor of 2 for the shooting action
    else:
        total_bins = bins ** 3

    if not (0 <= discrete_action < total_bins):
        raise ValueError(f"Discrete action {discrete_action} is out of bounds for bins {bins}")

    # Decode the discrete action into bins
    x_bin = (discrete_action // (bins * bins)) % bins
    y_bin = (discrete_action // bins) % bins
    angle_bin = discrete_action % bins

    # Midpoint scaling for smoother transitions
    if bins > 1:
        action_cont = [
            (x_bin + 0.5) / bins * 2 - 1,
            (y_bin + 0.5) / bins * 2 - 1,
            (angle_bin + 0.5) / bins * 2 - 1
        ]
    else:
        action_cont = [0.0, 0.0, 0.0]

    # If keep_mode is enabled, extract the shooting action
    if keep_mode:
        shoot_bin = (discrete_action // (bins ** 3)) % 2  # Binary (0 or 1)
        action_cont.append(float(shoot_bin))  # Add shooting action as continuous value

    return action_cont


