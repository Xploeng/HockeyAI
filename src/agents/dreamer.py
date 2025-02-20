from utils.models import WorldModel
from utils.networks import Actor, Critic
from .agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import collections
import sys
from gymnasium import spaces
sys.path.append("src/")
import numpy as np
from PIL import Image


class Dreamer(Agent):
    """
    Dreamer Agent: A model-based reinforcement learning agent that learns by imagining trajectories.
    
    Key Components:
    1. World Model (~ the "imagination" part of Dreamer):
        - Encoder: Compresses observations into latent states (~ Understand what it sees)
        - RSSM (Recurrent State-Space Model): Predicts next latent states (~ Predict what will happen next)
        - Decoder: Reconstructs observations from latent states (~ Visualize its predictions)
    
    2. Actor-Critic Networks:
        - Actor: Learns policy in latent space
        - Critic: Evaluates value of imagined trajectories
    
    Training Process:
    1. World Model Training:
        - Collect real environment experiences
        - Encode observations to latent states
        - Learn state transitions using RSSM
        - Reconstruct observations
        - Minimize reconstruction loss + KL divergence
    
    2. Policy Training:
        - Start from encoded state
        - Imagine trajectories using world model
        - Optimize actor-critic using imagined outcomes
        - Actor maximizes expected imagined rewards
        - Critic learns value estimates of imagined states
    
    Computational Flow:
    1. Real Experience Collection:
        observation -> encoded state -> action -> next observation -> reward
    
    2. World Model Updates:
        - Encode: observation -> latent state (mean, logvar)
        - Transition: predict next state using RSSM
        - Decode: latent state -> reconstructed observation
        - Losses: reconstruction loss + KL divergence between prior and posterior
    
    3. Policy Learning:
        - Generate imagined trajectories
        - Compute rewards and values in latent space
        - Update actor to maximize expected returns
        - Update critic to better estimate values
    
    Key Hyperparameters:
    - hidden_dim: Size of RSSM hidden state
    - state_dim: Dimension of latent state
    - embed_dim: Dimension of encoded observations
    - kl_weight: Balances reconstruction vs KL loss
    """
    
    def __init__(self, model_config, actor_config, critic_config, training,  env, memory, device, **kwargs):
        super().__init__()  
        """
        Initialize Dreamer with necessary hyperparameters and configs.
        model_config : dict with settings for the world model
        actor_config : dict for actor network settings
        critic_config: dict for critic network settings
        env_config   : dict or environment reference, for real environment interaction
        """

        # Store configurations
        self.model_config = model_config
        self.actor_config = actor_config
        self.critic_config = critic_config
        self.env = env
        self.device = device
        self.memory = hydra.utils.instantiate(config=memory)
        self.training_cfg = training

        # Initialize networks (world model, actor, critic)
        self._build_model()
        self._build_actor_critic()

    def _build_model(self):
        """Build the world model components"""
        obs_shape = self.env.observation_space.shape
        
        # Handle both discrete and continuous action spaces
        if isinstance(self.env.action_space, spaces.Discrete):
            action_dim = self.env.action_space.n
        else:  # spaces.Box
            action_dim = self.env.action_space.shape[0]
        
        self.world_model = WorldModel(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_dim=self.model_config.get('hidden_dim', 200),
            state_dim=self.model_config.get('state_dim', 30),
            embed_dim=self.model_config.get('embed_dim', 200)
        )
        
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=self.model_config.get('learning_rate', 1e-3)
        )

    def train_world_model(self, real_experiences):
        """
        Train the world model components (encoder, transition, decoder) 
        using real environment experiences.
        real_experiences: data or batch from a replay buffer containing Transitions
        """
        # Unpack transitions from the batch
        transitions = real_experiences['transitions']
        obs = torch.FloatTensor(np.array([t.state for t in transitions])).to(self.device)
        actions = torch.FloatTensor(np.array([t.action for t in transitions])).to(self.device)
        rewards = torch.FloatTensor(np.array([t.reward for t in transitions])).to(self.device)
        
        # Get posterior and prior latent states
        posterior_state, prior_state, recon = self.world_model(obs, actions)
        
        # Reconstruction loss (~ ability to accurately reconstruct the observation from posterior latent state by decoder)
        recon_loss = F.mse_loss(recon, obs)
        
        # KL divergence loss (~ Difference of how we expected the latent state to look without observing vs how it actually looked)
        kl_loss = self.world_model.kl_loss(prior_state, posterior_state)
        
        # Predict rewards
        predicted_rewards = self.world_model.predict_reward(posterior_state)
        reward_loss = F.mse_loss(predicted_rewards, rewards.unsqueeze(-1))
        
        # Update total loss
        loss = recon_loss + self.model_config.get('kl_weight', 1.0) * kl_loss.mean() + reward_loss
        
        # Optimize
        self.world_model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()
        
        return {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.mean().item(),
            'reward_loss': reward_loss.item()
        }

    def train_world_model_sequence(self, seq_batch, sequence_length=50):
        """
        Train the world model by unrolling the RSSM over an entire sequence of length T (=sequence_length).
        Accumulate reconstruction, KL, and reward prediction losses across all timesteps in the sequence.

        Args:
            seq_batch: A dictionary containing:
                - obs_seq: (B, T, obs_dim or image shape)
                - action_seq: (B, T, action_dim)
                - reward_seq: (B, T)
            sequence_length (int): Length of the sequence unroll for RSSM training.
        
        Returns:
            A dict of losses (total, recon, kl, reward).
        """
        obs_seq = seq_batch['obs_seq']       # shape (B, T, obs_dim) or (B, T, H, W, C) for images
        action_seq = seq_batch['action_seq'] # shape (B, T, action_dim)
        reward_seq = seq_batch['reward_seq'] # shape (B, T)

        batch_size = obs_seq.shape[0]
        device = self.device

        # Move tensors to device
        obs_seq = torch.FloatTensor(obs_seq).to(device)
        action_seq = torch.FloatTensor(action_seq).to(device)
        reward_seq = torch.FloatTensor(reward_seq).to(device)

        # Initialize the RSSM state for the batch
        state = self.world_model.initial_state(batch_size, device=device)

        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_reward_loss = 0.0

        for t in range(sequence_length):
            # Grab the t-th observation, action, reward
            obs_t = obs_seq[:, t]
            act_t = action_seq[:, t]
            rew_t = reward_seq[:, t]

            # Forward pass: unroll the RSSM
            posterior_state, prior_state, recon = self.world_model(obs_t, act_t, prev_state=state)

            # Compute reconstruction loss
            # NOTE: for images, recon and obs_t might have shape (B, C, H, W); handle MSE or BCE accordingly
            if self.world_model.is_image_obs:
                # e.g. pixel-based:
                recon_loss_t = F.mse_loss(recon, obs_t.permute(0, 3, 1, 2))  # if your decoder outputs (B,C,H,W)
            else:
                # vector-based:
                recon_loss_t = F.mse_loss(recon, obs_t)

            # KL divergence
            kl_loss_t = self.world_model.kl_loss(prior_state, posterior_state)

            # Reward prediction
            predicted_reward = self.world_model.predict_reward(posterior_state)
            reward_loss_t = F.mse_loss(predicted_reward, rew_t.unsqueeze(-1))

            total_recon_loss += recon_loss_t
            total_kl_loss += kl_loss_t.mean()  # or sum(), depending on your preference
            total_reward_loss += reward_loss_t

            # Next iteration, the `posterior_state` becomes our new "prev_state"
            # so we effectively unroll in time
            state = posterior_state

        # Average the losses over time
        mean_recon_loss = total_recon_loss / sequence_length
        mean_kl_loss = total_kl_loss / sequence_length
        mean_reward_loss = total_reward_loss / sequence_length

        # Scale KL by your `kl_weight` if needed
        kl_weight = self.model_config.get('kl_weight', 1.0)
        loss = mean_recon_loss + kl_weight * mean_kl_loss + mean_reward_loss

        # Optimize
        self.world_model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()

        return {
            'loss': loss.item(),
            'recon_loss': mean_recon_loss.item(),
            'kl_loss': mean_kl_loss.item(),
            'reward_loss': mean_reward_loss.item()
        }

    def _build_actor_critic(self):
        """
        Build actor and critic networks.
        Actor: will sample actions from a learned policy in latent space.
        Critic: evaluate the value of latent states or trajectories that result from the WorldModel applying the sampled actions from Actor.
        """
        feature_dim = self.world_model.rssm.state_size + self.world_model.rssm.belief_size
        
        # Handle both discrete and continuous action spaces
        if isinstance(self.env.action_space, spaces.Discrete):
            action_dim = self.env.action_space.n
        else:  # spaces.Box
            action_dim = self.env.action_space.shape[0]

        # Initialize actor and critic networks using existing classes
        self.actor = Actor(
            input_size=feature_dim,
            hidden_size=self.actor_config.get('hidden_dim', 400),
            output_size=action_dim,
            min_stddev=self.actor_config.get('min_stddev', 0.1),
            deterministic=False
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_config.get('learning_rate', 1e-3)
        )

        self.critic = Critic(
            input_size=feature_dim,
            hidden_size=self.critic_config.get('hidden_dim', 400),
            output_size=1
        ).to(self.device)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_config.get('learning_rate', 1e-3)
        )

    def imagine_rollout(self, start_state, horizon):
        """
        Generate imagined trajectories from the world model starting from a given latent state.
        
        Args:
            start_state: Initial latent state dictionary from RSSM
            horizon: Integer length of imagination horizon
        
        Returns:
            Dictionary containing:
            - imagined_states: List of imagined latent states
            - predicted_values: List of value predictions for each state
            - predicted_rewards: List of predicted rewards for each state
            - actions: List of sampled actions
        """
        prev_state = start_state
        predicted_values = []
        imagined_states = [start_state]
        actions = []
        predicted_rewards = []
        
        for _ in range(horizon): 
            # Get action distribution from current state features
            state_features = self.world_model.get_features(prev_state)
            action_dist = self.actor(state_features)
            
            # Sample action from distribution
            action = action_dist.sample()
            actions.append(action)
            
            # Imagine next state using world model
            next_state = self.world_model.imagine(prev_state, action)
            imagined_states.append(next_state)
            
            # Get value prediction for imagined state
            next_features = self.world_model.get_features(next_state)
            state_value = self.critic(next_features)
            predicted_values.append(state_value)
            
            # Predict reward for current state
            reward = self.world_model.predict_reward(prev_state)
            
            # Store predicted reward
            predicted_rewards.append(reward)
            
            prev_state = next_state
        
        return {
            'imagined_states': imagined_states,
            'predicted_values': predicted_values,
            'predicted_rewards': predicted_rewards,
            'actions': actions
        }

    def train_actor_critic(self, imagined_data):
        """
        Train actor and critic using imagined trajectories.
        
        Args:
            imagined_data: Dictionary containing:
                - imagined_states: List of imagined latent states
                - predicted_values: List of value predictions
                - predicted_rewards: List of predicted rewards
                - actions: List of sampled actions
        """
        states = imagined_data['imagined_states']
        values = imagined_data['predicted_values']
        rewards = imagined_data['predicted_rewards']
        actions = imagined_data['actions']
        
        # Hyperparameters for return calculation
        lambda_value = 0.95  # Mixing parameter for TD(λ)
        discount = 0.99  # Discount factor for future rewards
        horizon = len(values)
        
        # Pre-compute all state features
        state_features_list = []
        returns = []
        
        with torch.no_grad():
            # Compute state features for all states
            for t in range(horizon):
                state_features = self.world_model.get_features(states[t])
                state_features_list.append(state_features)
            
            # Compute lambda-returns (going backwards)
            last_value = self.critic(state_features_list[-1])
            next_value = last_value
            next_return = last_value

            for t in reversed(range(horizon)):
                reward = rewards[t]
                value = values[t]
                
                # Calculate return using both rewards and values (lambda-weighted return)
                return_t = reward + discount * ((1 - lambda_value) * value + lambda_value * next_return)
                returns.insert(0, return_t.detach())
                next_return = return_t
        
        # Update critic
        critic_loss = torch.tensor(0.0, device=self.device)
        for t in range(horizon):
            value_pred = self.critic(state_features_list[t])
            value_pred = torch.clamp(value_pred, -100, 100)  # Add bounds to value predictions
            critic_loss = critic_loss + F.mse_loss(value_pred, returns[t])
        critic_loss = critic_loss / horizon * 0.1  # Add scaling factor

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor using advantage
        actor_loss = torch.tensor(0.0, device=self.device)
        for t in range(horizon):
            # Recompute value predictions after critic update
            with torch.no_grad():
                value_pred = self.critic(state_features_list[t])
                advantage = (returns[t] - value_pred)
            
            action_dist = self.actor(state_features_list[t])
            log_prob = action_dist.log_prob(actions[t])
            actor_loss = actor_loss - (log_prob * advantage).mean()
        
        actor_loss = actor_loss / horizon  # Average over horizon

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # Add gradient clipping
        self.actor_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }

    def train_episode(self):
        """Train the agent for one episode"""
        # Reset environment at the start of each episode
        observation, _ = self.env.reset()
        done = False
        
        # Collect experiences for one episode
        while not done:
            # Select and perform action
            action = self.select_action(observation)
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition in memory
            self.record(observation, action, next_observation, reward, done)
            
            # Move to next state
            observation = next_observation
        
        # Only optimize if we have enough samples
        if len(self.memory) >= self.training_cfg.batch_size:
            loss_dict = self.optimize()
            print(loss_dict)
            self.losses.append(sum(loss_dict.values()))
        else:
            self.losses.append(0)  # No training occurred

    def select_action(self, observation):
        """
        Select an action for the given observation.
        
        Args:
            observation: Environment observation
        """
        # Encode observation
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            # Get latent state
            if not hasattr(self, 'current_state'):
                self.current_state = self.world_model.initial_state(batch_size=1, device=self.device)
                self.prev_action = None  # Initialize prev_action as None for first step
            
            # Update state using observation and previous action
            posterior_state, _ = self.world_model.observe(obs_tensor, self.prev_action, self.current_state)
            self.current_state = posterior_state
            
            # Get action from actor
            state_features = self.world_model.get_features(posterior_state)
            action_dist = self.actor(state_features)
            
            # Get mean action and handle different action space types
            action = action_dist.mean.cpu().numpy().squeeze()
            
            # Convert action to appropriate format and store for next step
            if isinstance(self.env.action_space, spaces.Discrete):
                action = int(action)  # Convert to integer for discrete action spaces
                self.prev_action = torch.tensor([[action]], device=self.device)
            elif isinstance(self.env.action_space, spaces.Box):
                if len(self.env.action_space.shape) == 1 and self.env.action_space.shape[0] == 1:
                    action = np.array([float(action)])  # Return 1D array for environments like Pendulum
                self.prev_action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
        return action

    def optimize(self, *args, **kwargs):
        """
            Main update function that performs one iteration of Dreamer training:
            1. Samples experiences from the replay buffer
            2. Updates the world model
            3. Imagines trajectories
            4. Updates actor & critic
        """
        replay_buffer = self.memory
        batch_size = self.training_cfg.batch_size
        sequence_length = self.training_cfg.get('sequence_length', 50)

        if len(replay_buffer) < batch_size or replay_buffer.ptr == 0:
            print(f"Warning: Skipping optimization. Buffer size ({len(replay_buffer)}) < batch size ({batch_size}) or empty buffer")
            return {}  # Return empty loss dict if not enough samples

        try:
            # 1. Sample sequences (obs_seq, action_seq, reward_seq) each of length T
            seq_batch = replay_buffer.sample_sequences(batch_size, sequence_length)
            
            # 2. Train world model
            world_model_loss = self.train_world_model_sequence(seq_batch, sequence_length=sequence_length)
            
            # 3. Generate imagined trajectories
            initial_state = self.world_model.initial_state(batch_size=batch_size, device=self.device)
            imagination_horizon = 15  # Typical value from paper
            imagined_trajectories = self.imagine_rollout(initial_state, imagination_horizon)
            
            # 4. Train actor and critic on imagined outcomes
            actor_critic_losses = self.train_actor_critic(imagined_trajectories)
            
            return {
                **world_model_loss,
                **actor_critic_losses
            }
            
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            return {}

    def record(self, state, action, next_state, reward, done):
        """
        Stores a transition in memory (replay buffer)
        
        Args:
            state: Current state/observation
            action: Action taken
            next_state: Next state/observation
            reward: Reward received
            done: Whether episode ended
        """
        self.steps_done += 1
        
        # Convert inputs to appropriate format if needed
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        # Ensure action is the right shape
        if isinstance(self.env.action_space, spaces.Box) and not isinstance(action, np.ndarray):
            action = np.array([action])
        
        try:
            self.memory.push(state, action, next_state, reward, done)
        except Exception as e:
            print(f"Error storing transition in replay buffer: {str(e)}")

    @property
    def optimizer(self):
        """
        Returns a dictionary of all optimizers for checkpoint saving/loading.
        This property is used by the helper functions for saving/loading checkpoints.
        """
        return {
            "world_model": self.world_model_optimizer,
            "actor": self.actor_optimizer,
            "critic": self.critic_optimizer
        }

    def load_state_dict(self, agent_state_dict, optimizer_state_dict=None, **_):
        """
        Load saved model parameters and optimizer states
        
        Args:
            agent_state_dict: Dictionary containing model parameters
            optimizer_state_dict: Optional dictionary containing optimizer states
        """
        # Load world model parameters
        self.world_model.load_state_dict(agent_state_dict["world_model"])
        
        # Load actor-critic parameters
        self.actor.load_state_dict(agent_state_dict["actor"])
        self.critic.load_state_dict(agent_state_dict["critic"])
        
        # Load optimizer states if provided
        if optimizer_state_dict is not None:
            self.world_model_optimizer.load_state_dict(optimizer_state_dict["world_model"])
            self.actor_optimizer.load_state_dict(optimizer_state_dict["actor"])
            self.critic_optimizer.load_state_dict(optimizer_state_dict["critic"])
        
        # Load memory if exists
        if "memory" in agent_state_dict:
            self.memory = agent_state_dict["memory"]

    def state_dict(self):
        """
        Return a dictionary containing the state of each module for saving
        
        Returns:
            OrderedDict containing model parameters and memory
        """
        return collections.OrderedDict({
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "memory": self.memory
        })

    def evaluate_episode(self, render: bool = True) -> tuple[list[Image.Image], dict]:
        """
        Run one evaluation episode, collecting frames and info.
        
        Args:
            render (bool): Whether to render and collect frames
            
        Returns:
            tuple containing:
            - list of frames as PIL Images
            - info dictionary from environment
        """
        observation, info = self.env.reset()
        done = False
        frames = []

        while not done:
            # Render the environment and save the frames if requested
            if render:
                frame = self.env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame))

            # Select action using the current policy
            action = self.select_action(observation)
            
            # Step the environment
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition in memory
            self.record(observation, action, next_observation, reward, done)
            
            # Move to next state if not done
            if not done:
                observation = next_observation

        return frames, info
