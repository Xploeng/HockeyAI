import os
import sys

from dataclasses import asdict
from pathlib import Path
import gymnasium as gym
import hockey
import hydra
import numpy as np
import torch
import yaml

from gymnasium import spaces
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from tqdm import tqdm


sys.path.append("src/")
from agents import Agent
from utils import DiscreteActionWrapper, OpponentWrapper, load_checkpoint
from utils.visuals import (
    EpisodeStatistics,
    plot_q_function_all_dims,
    plot_rewards,
    plot_wins_vs_losses,
    save_gif,
    save_json,
)


def get_checkpoint_path(agent_name):
    return os.path.join(
        "src/outputs",
        agent_name,
        "checkpoints",
        f"{agent_name}_last.ckpt",
    )

def initialize_opponent(cfg: DictConfig, env, device: torch.device):
    if cfg.env.opponent_type == "AgentOpponent":
        opp_cfg_pth = Path(".") / "src" / "outputs" / cfg.env.opponent.name / ".hydra" / "config.yaml"
        with open(opp_cfg_pth) as file:
            opp_cfg = DictConfig(yaml.safe_load(file))

        # enable checkpoint loading
        opp_cfg.agent.training.continue_training = True
        opp_cfg.agent.mode = 'opponent'

        opp: Agent = initialize_agent(
            cfg=opp_cfg,
            env=env,
            device=device,
            checkpoint_path=get_checkpoint_path(cfg.env.opponent.name),
        )
        return OpponentWrapper(opp, env)
    elif cfg.env.opponent_type == "BasicOpponent":
        return OpponentWrapper(hydra.utils.instantiate(cfg.env.opponent), env)

def initialize_environment(cfg: DictConfig):
    if cfg.env.name == "Hockey-v0":
        env = hockey.hockey_env.HockeyEnv()
    else:
        env = gym.make(cfg.env.name, render_mode="rgb_array")

    # Check if env continuous and agent not continuous -> wrap env
    agent_continuous = cfg.agent.requires_continues_action_space
    if isinstance(env.action_space, spaces.Box) and not agent_continuous:
        env = DiscreteActionWrapper(env, bins=cfg.agent.bins)
    elif isinstance(env.action_space, spaces.Discrete) and agent_continuous:
        raise ValueError(
            f"Agent requires a continuous action space, but {cfg.env} has a discrete action space.",
        )
    return env


def initialize_agent(cfg: DictConfig, env: gym.Env, device: torch.device, checkpoint_path: str, opponent= None) -> tuple[Agent, int]:
    agent_continuous = cfg.agent.requires_continues_action_space
    env_continuous = isinstance(env.action_space, spaces.Box)
    if agent_continuous and not env_continuous:
        raise ValueError("The agent requires a continuous action space, but the environment has a discrete one.")

    agent: Agent = hydra.utils.instantiate(
        config=cfg.agent,
        env=env,
        opponent=opponent,
        device=device,
        recursive=False,
    )

    cfg.agent.training.continue_training = True

    episode = load_checkpoint(cfg, agent, checkpoint_path, device)
    print(f"Loaded checkpoint from episode {episode}.")

    return agent


def evaluate_model(cfg: DictConfig, agent_cfg: DictConfig) -> None:
    device = torch.device(cfg.device)

    animation_dir = os.path.join("src/outputs", cfg.agent, "animations")
    os.makedirs(animation_dir, exist_ok=True)
    episode_stats_dir = os.path.join("src/outputs", cfg.agent, "episode_statistics")
    os.makedirs(episode_stats_dir, exist_ok=True)
    figures_dir = os.path.join("src/outputs", cfg.agent, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    checkpoint_path = get_checkpoint_path(agent_cfg.agent.name)

    # Initialize the environment and agent
    agent_cfg.env = cfg.env
    env= initialize_environment(agent_cfg)
    opp = initialize_opponent(cfg, env, device) if cfg.env.name == "Hockey-v0" else None
    agent = initialize_agent(agent_cfg, env, device, checkpoint_path, opp)

    # Fresh recordings (clear training recordings)
    agent.memory.clear()

    # Plot the Q-function for all state dimensions
    # plot_q_function_all_dims(agent, cfg.env, figures_dir)

    wins = 0
    draws = 0
    losses = 0

    # Start the evaluation
    all_episode_stats = {}
    print(f"\nEvaluating the model on {cfg.episodes} episodes.")
    if cfg.hockey:
        op_name = agent_cfg.env.opponent.name if cfg.env.opponent_type == "AgentOpponent" else "BasicOpponent"
        print("Hockey mode enabled.")
        print(f"Evaluating {cfg.agent} against {op_name}.")
    for episode in tqdm(range(cfg.episodes)):
        frames, info = agent.evaluate_episode(render=cfg.render)

        # Keep track of episode statistics and save the animation
        rewards, states = agent.memory.rewards, agent.memory.states
        episode_stats = EpisodeStatistics(episode=episode, rewards=rewards, states=states, info=info)
        all_episode_stats[episode] = asdict(episode_stats)
        agent.memory.clear()

        if cfg.hockey:
            if info["winner"] == 1:
                wins += 1
            elif info["winner"] == 0:
                draws += 1
            else:
                losses += 1
        if cfg.render:
            gif_path = os.path.join(animation_dir, f"episode_{episode}.gif")
            save_gif(frames, gif_path)

    # Save the episode statistics as json
    stats_file_path = os.path.join(episode_stats_dir, f"episode_statistics_{agent_cfg.agent.name}.json")
    save_json(all_episode_stats, stats_file_path)
    print(f"Episode statistics saved to {stats_file_path}")

    # Plot the results as a pie chart
    if cfg.hockey:
        plot_wins_vs_losses(wins, draws, losses, figures_dir, show=cfg.show_figures)

@hydra.main(config_path="../configs/", config_name="config_eval", version_base=None)
def run_evaluations(cfg: DictConfig) -> None:
    device = cfg.device
    silent = cfg.silent
    episodes = cfg.episodes
    show_figures = cfg.show_figures

    GlobalHydra.instance().clear()
    config_path = os.path.join("../outputs", cfg.agent, ".hydra")
    with hydra.initialize(version_base=None, config_path=config_path):
        agent_cfg = hydra.compose(config_name="config")
        agent_cfg.device = device
        agent_cfg.verbose = not silent

    if agent_cfg.seed:
        np.random.seed(agent_cfg.seed)
        torch.manual_seed(agent_cfg.seed)

    evaluate_model(cfg=cfg, agent_cfg=agent_cfg)

    agent_output_dir = os.path.join("src/outputs", agent_cfg.agent.name)
    plot_rewards(
        agent_output_dir,
        n_episodes=episodes,
        show=show_figures,
    )

if __name__ == "__main__":
    run_evaluations()#
