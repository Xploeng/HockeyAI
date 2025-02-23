from __future__ import annotations

import argparse
import os
import sys
import uuid
import hockey.hockey_env as h_env
import hydra
import numpy as np
import torch

from icecream import ic
from omegaconf import DictConfig


sys.path.append("src/")
from comprl.client import Agent, launch_client
# from agents import Agent
from agents.rainbow import Rainbow
from scripts.train import get_checkpoint_path
from utils.helper import DiscreteActionWrapper, load_checkpoint


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(f"game ended: {text_result} with my score: " f"{stats[0]} against the opponent with score: {stats[1]}")


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id, byteorder="big"))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(f"Game ended: {text_result} with my score: " f"{stats[0]} against the opponent with score: {stats[1]}")


class RainbowHockeyAgent(Agent):
    """A hockey agent that uses Rainbow DQN."""

    def __init__(self, config_path: str) -> None:
        super().__init__()

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the config
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg = hydra.compose(config_name=config_path)

        # Initialize environment
        self.env = h_env.HockeyEnv()
        if not cfg.agent.requires_continues_action_space:
            self.env = DiscreteActionWrapper(self.env, bins=cfg.agent.bins)

        # Initialize the Rainbow agent
        self.rainbow = hydra.utils.instantiate(
            config=cfg.agent,
            env=self.env,
            opponent=None,
            device=self.device,
            recursive=False,
        )

        # Load the checkpoint
        checkpoint_path = "/Users/ericnazarenus/Library/Mobile Documents/com~apple~CloudDocs/Uni/WS2024/Reinforcement Learning/HockeyAI/src/outputs/rainbow_hockey_bot_composite/checkpoints/rainbow_hockey_bot_composite_last.ckpt"
        cfg.agent.training.continue_training = True
        if os.path.exists(checkpoint_path):
            load_checkpoint(cfg, self.rainbow, checkpoint_path, self.device)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    def get_step(self, observation: list[float]) -> list[float]:
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.rainbow.select_action(state)
        print(action)
        continuous_action = self.env.discrete_to_continous_action(action.item())
        print(continuous_action)
        return continuous_action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id, byteorder="big"))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(f"Game ended: {text_result} with my score: " f"{stats[0]} against the opponent with score: {stats[1]}")


class SACHockeyAgent(Agent):
    """A hockey agent that uses Soft Actor-Critic."""

    def __init__(self, config_path: str) -> None:
        super().__init__()

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the config
        with hydra.initialize(version_base=None, config_path="../configs"):
            cfg = hydra.compose(config_name=config_path)

        print(cfg)
        # Initialize environment
        self.env = h_env.HockeyEnv()

        # Initialize the SAC agent
        self.sac = hydra.utils.instantiate(
            config=cfg.agent,
            env=self.env,
            opponent=None,
            device=self.device,
            mode="opponent",
            recursive=False,
        )

        cfg.agent.training.continue_training = True
        # Load the checkpoint
        # checkpoint_path = "/home/tluebbing/workspace/studies/HockeyAI/src/outputs/sac_hockey_bot_play_v0/checkpoints/sac_hockey_bot_play_v0_last.ckpt"
        checkpoint_path = "/home/tluebbing/workspace/studies/HockeyAI/src/outputs/sac_hockey_bot_play_v0/checkpoints/sac_hockey_bot_play_v0_last.ckpt"
        if os.path.exists(checkpoint_path):
            episode = load_checkpoint(cfg, self.sac, checkpoint_path, self.device)
            print(f"Loaded checkpoint from {checkpoint_path}")
            print(f"Checkpoint loaded from episode {episode}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    def get_step(self, observation: list[float]) -> list[float]:
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.sac.select_action(state.squeeze(0), deterministic=True)
        return action.tolist()

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id, byteorder="big"))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(f"Game ended: {text_result} with my score: " f"{stats[0]} against the opponent with score: {stats[1]}")


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["weak", "strong", "random", "rainbow", "sac"],
        default="weak",
        help="Which agent to use.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="rainbow_hockey_client.yaml",
        help="Path to config file for Rainbow/SAC agent.",
    )
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "rainbow":
        agent = RainbowHockeyAgent(config_path=args.config)
    elif args.agent == "sac":
        agent = SACHockeyAgent(config_path=args.config)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
