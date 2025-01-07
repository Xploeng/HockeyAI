import os
import sys
import threading

from copy import deepcopy
import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.utils.tensorboard as tb

from gymnasium import spaces
from omegaconf import DictConfig
from tqdm import tqdm


sys.path.append("src/")
from agents import Agent
from utils import DiscreteActionWrapper, ReplayMemory, write_checkpoint


class Trainer:
    """Handles the training pipeline for agents."""

    def __init__(self, cfg: DictConfig, agent, env, device):
        self.cfg = cfg
        self.agent = agent
        self.env = env
        self.device = device
        self.writer = tb.SummaryWriter(log_dir=os.path.join("src/outputs", cfg.agent.name, "tensorboard"))
        self.checkpoint_path = os.path.join(
            "src/outputs",
            cfg.agent.name,
            "checkpoints",
            f"{cfg.agent.name}_last.ckpt",
        )

    def train(self):
        start_episode = self._load_checkpoint()

        print(f"Starting training from episode {start_episode} to {self.cfg.training.episodes}")
        for episode in tqdm(range(start_episode, self.cfg.training.episodes)):
            self._train_episode(episode)
            if self.cfg.training.save_agent and episode % self.cfg.training.save_interval == 0:
                self._save_checkpoint(episode)

        self.writer.flush()
        self.writer.close()
        self.env.close()

    def _train_episode(self, episode):
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        done = False

        while not done:
            action = self.agent.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward], device=self.device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

            self.agent.record(state, action, next_state, reward, done)
            self.agent.optimize(**self.cfg.training)

            state = next_state

        loss = self.agent.losses[-1] if self.agent.losses else 0
        self.writer.add_scalar("Loss", loss, global_step=self.agent.steps_done)
        self.writer.add_scalar("Episode", episode, global_step=self.agent.steps_done)

    def _load_checkpoint(self):
        if self.cfg.training.continue_training and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.agent.load_state_dict(**checkpoint)
            return checkpoint.get("episode", 0)
        return 0

    def _save_checkpoint(self, episode):
        agent_cp = deepcopy(self.agent)
        thread = threading.Thread(
            target=write_checkpoint,
            args=(agent_cp, self.agent.optimizer, episode, self.checkpoint_path),
        )
        thread.start()
        thread.join()


@hydra.main(config_path="../configs/", config_name="config", version_base=None)
def run_training(cfg: DictConfig):
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    if cfg.seed:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    env = gym.make(cfg.env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    agent = Agent.from_config(cfg.agent, env)
    if isinstance(env.action_space, gym.spaces.Box) and not agent.is_continueous:
        env = DiscreteActionWrapper(env, bins=cfg.bins)

    trainer = Trainer(cfg, agent, env, device)
    trainer.train()


