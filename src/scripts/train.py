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
from tqdm import tqdm


sys.path.append("src/")
from agents import Agent
from utils import DiscreteActionWrapper, ReplayMemory, write_checkpoint


@hydra.main(config_path="../configs/", config_name="config", version_base=None)
def run_training(cfg):

    device = torch.device(cfg.device)
    print(f"\nDevice: {device}")
    print(f"Environment: {cfg.env}")
    print(f"Agent: {cfg.agent.name} of type {cfg.agent._target_}")

    env = gym.make(cfg.env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if isinstance(env.action_space, spaces.Box):
        n_actions = cfg.bins
        env = DiscreteActionWrapper(env, bins=cfg.bins)
    elif isinstance(env.action_space, spaces.Discrete):
        n_actions = env.action_space.n
    if isinstance(env.observation_space, spaces.Box):
        n_oberservations = env.observation_space.shape[0]
    else:
        state, info = env.reset()
        n_oberservations = len(state)

    if cfg.seed:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # Initialize the replay buffer
    memory: ReplayMemory = hydra.utils.instantiate(config=cfg.memory)

    # Initialize the policy and target network
    policy_net = hydra.utils.instantiate(
        config=cfg.network,
        n_observations=n_oberservations,
        n_actions=n_actions,
    ).to(device=device)
    target_net = hydra.utils.instantiate(
        config=cfg.network,
        n_observations=n_oberservations,
        n_actions=n_actions,
    ).to(device=device)
    target_net.load_state_dict(policy_net.state_dict())

    # Initialize the optimizer
    optimizer = hydra.utils.instantiate(
        config=cfg.training.optimizer,
        params=policy_net.parameters(),
    )
    # Initialize the loss function
    criterion = hydra.utils.instantiate(config=cfg.training.criterion)

    # Initialize the agent
    agent: Agent = hydra.utils.instantiate(
        config=cfg.agent,
        n_actions=n_actions,
        n_observations=n_oberservations,
        policy_net=policy_net,
        target_net=target_net,
        optimizer=optimizer,
        criterion=criterion,
        memory=memory,
        env=env,
        device=device,
        episodes=cfg.training.episodes,
    )

    # Load checkpoint from file to continue training or initialize training scalars
    checkpoint_path = os.path.join(
        "src/outputs",
        cfg.agent.name,
        "checkpoints",
        f"{cfg.agent.name}_last.ckpt",
    )
    if cfg.training.continue_training:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.load_state_dict(**checkpoint)
        start_episode = checkpoint["episode"]
    else:
        start_episode = 0

    # Write the model configurations to the model save path
    os.makedirs(os.path.join("src/outputs", cfg.agent.name), exist_ok=True)

    # Initialize tensorbaord to track scalars
    writer = tb.SummaryWriter(
        log_dir=os.path.join("src/outputs", cfg.agent.name, "tensorboard"),
    )

    # training loop
    print("Starting training")
    print(f"Training from {start_episode} to {cfg.training.episodes} episodes")
    for episode in tqdm(range(start_episode, cfg.training.episodes)):
        # Track the episode number and learning rate
        writer.add_scalar(
            tag="Episode",
            scalar_value=episode,
            global_step=agent.steps_done,
        )
        writer.add_scalar(
            tag="Learning Rate",
            scalar_value=optimizer.state_dict()["param_groups"][0]["lr"],
            global_step=agent.steps_done,
        )

        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        while not done:
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

            # Store the transition in memory
            agent.record(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize(**cfg.training, episode=episode)

            loss = agent.losses[-1] if len(agent.losses) > 0 else 0
            writer.add_scalar(
                tag="Loss",
                scalar_value=loss,
                global_step=agent.steps_done,
            )
        # Write checkpoint to file, using a separate thread
        if cfg.training.save_agent and episode % cfg.training.save_interval == 0:
            agent_cp, optimizer_cp = deepcopy(agent), deepcopy(optimizer)
            thread = threading.Thread(
                target=write_checkpoint,
                args=(agent_cp, optimizer_cp, episode, checkpoint_path),
            )
            thread.start()
    try:
        thread.join()
        writer.flush()
        writer.close()
        env.close()
    except NameError:
        pass


if __name__ == "__main__":
    run_training()
