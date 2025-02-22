import os
import sys
import gymnasium as gym
import yaml
import hockey
import hydra
import numpy as np
import torch
import torch.utils.tensorboard as tb

from gymnasium import spaces
from omegaconf import DictConfig
from tqdm import tqdm


sys.path.append("src/")
from agents import Agent
from utils.helper import DiscreteActionWrapper, OpponentWrapper, load_checkpoint, save_checkpoint

def get_checkpoint_path(agent_name):
    return os.path.join(
        "src/outputs",
        agent_name,
        "checkpoints",
        f"{agent_name}_last.ckpt",
    )

def initialize_opponent(cfg: DictConfig, env, device: torch.device):
    if cfg.env.opponent_type == "AgentOpponent":
        opp_cfg_pth = os.path.join("./src/outputs", cfg.env.opponent.name, ".hydra/config.yaml")
        with open(opp_cfg_pth, 'r') as file:
            opp_cfg = DictConfig(yaml.safe_load(file))
        
        # enable checkpoint loading
        opp_cfg.agent.training.continue_training = True
        opp_cfg.agent.mode = 'opponent'
        
        opp, _ = initialize_agent(cfg=opp_cfg, env=env, device=device, checkpoint_path=get_checkpoint_path(cfg.env.opponent.name))
        return OpponentWrapper(opp, env, opp_cfg.agent.requires_continues_action_space)
    elif cfg.env.opponent_type == "BasicOpponent":
        return OpponentWrapper(hydra.utils.instantiate(cfg.env.opponent), env, opp_cfg.agent.requires_continues_action_space)

def initialize_environment(cfg: DictConfig):
    if cfg.env.name == "Hockey-v0":
        env = hockey.hockey_env.HockeyEnv()
    else:
        env = gym.make(cfg.env.name)

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

    start_episode = load_checkpoint(cfg, agent, checkpoint_path, device)

    return agent, start_episode


@hydra.main(config_path="../configs/", config_name="config", version_base=None)
def run_training(cfg: DictConfig):
    """
    Orchestrates the training process for the specified number of episodes.

    Args:
        cfg (object): Configuration object containing training parameters.
        agent (object): The agent to train.
        env (object): The environment in which the agent operates.
        device (str): The device for computation (CPU/GPU).

    Returns:
        None
    """
    writer = tb.SummaryWriter(log_dir=os.path.join("src/outputs", cfg.agent.name, "tensorboard"))

    if cfg.seed:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    env = initialize_environment(cfg)
    opponent = initialize_opponent(cfg, env, device) if cfg.env.name == "Hockey-v0" else None
    agent, start_episode = initialize_agent(cfg, env, device, get_checkpoint_path(cfg.agent.name), opponent=opponent)

    print(f"Starting training from episode {start_episode} to {start_episode + cfg.agent.training.episodes}")
    for episode in tqdm(range(start_episode, start_episode + cfg.agent.training.episodes)):
        episode += start_episode

        agent.train_episode()

        loss = agent.losses[-1] if agent.losses else 0
        writer.add_scalar("Loss", loss, global_step=agent.steps_done)
        writer.add_scalar("Episode", episode, global_step=agent.steps_done)
        writer.add_scalar("Reward", agent.reward, global_step=episode)

        if cfg.agent.training.save_agent and episode % cfg.agent.training.save_interval == 0:
            save_checkpoint(agent, get_checkpoint_path(cfg.agent.name), episode)

    writer.flush()
    writer.close()
    env.close()


if __name__ == "__main__":
    run_training()
