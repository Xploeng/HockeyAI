import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence

class WorldModel(nn.Module):
    """
    World Model: Core component of Dreamer that learns to model environment dynamics in latent space.
    
    Architecture Components:
    1. Encoder (Observation -> Embedding):
        - Maps high-dimensional observations to compressed embeddings
        - Outputs a single embedded vector of size embed_dim
        - Used by RSSM to compute posterior state distribution
    
    2. RSSM (Recurrent State-Space Model):
        - Combines RNN and state-space modeling
        - RNN: Captures temporal dependencies in hidden states
        - State Prior: Predicts next latent state distribution
        - Handles both deterministic (hidden) and stochastic (latent) states
    
    3. Decoder (Latent State -> Observation):
        - Reconstructs observations from latent states
        - Acts as a generative model of observations
        - Helps ensure latent states capture relevant information
    
    Computational Flow:
    1. Encoding:
        observation -> encoder -> (mean, logvar) -> sampled latent state
    
    2. State Transition (RSSM):
        (state, action, hidden) -> RNN -> new hidden state -> state prior
        - Predicts distribution over next latent state
        - Uses both deterministic and stochastic paths
    
    3. Decoding:
        latent state -> decoder -> reconstructed observation
    
    Training Objectives:
    1. Reconstruction Loss: 
        - Minimize difference between real and reconstructed observations
    2. KL Divergence:
        - Between posterior (encoder) and prior (RSSM) distributions
        - Ensures meaningful and consistent latent space
    
    Parameters:
    - obs_shape: Shape of observation space
    - action_dim: Dimension of action space
    - hidden_dim: Size of RNN hidden state (default: 200)
    - state_dim: Dimension of latent state (default: 30)
    """
    def __init__(
            self, 
            obs_shape, 
            action_dim, 
            hidden_dim=200,
            state_dim=30,
            embed_dim=200,
            min_stddev=0.1,
            elu=nn.ELU
        ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Determine if input is image or vector based on shape
        self.is_image_obs = len(obs_shape) == 3  # Check if shape is (C, H, W)
        
        if self.is_image_obs:
            # Image encoder (CNN) architecture from paper : observation -> embedded state
            self.encoder = nn.Sequential(
                nn.Conv2d(obs_shape[-1], 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(1024, embed_dim)
            )
            
            # Image decoder architecture from paper: latent state -> observation reconstruction
            self.decoder = nn.Sequential(
                nn.Linear(state_dim, 1024),
                nn.Unflatten(1, (1024, 1, 1)),
                nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, obs_shape[-1], kernel_size=6, stride=2)
            )
        else:
            # Enhanced vector encoder for environments like CartPole
            self.encoder = nn.Sequential(
                nn.Linear(obs_shape[0], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim)
            )
            
            # Enhanced vector decoder
            self.decoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_shape[0])
            )
        
        # RSSM for state space modeling
        self.rssm = RSSM(
            state_size=state_dim,
            belief_size=hidden_dim,
            embed_size=embed_dim,
            action_size=action_dim,
            hidden_size=hidden_dim,
            min_stddev=min_stddev,
            elu=elu
        )

        # Add reward predictor network
        self.reward_predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def initial_state(self, batch_size, device=None):
        """Initialize RSSM state for new episodes."""
        return self.rssm.initial_state(batch_size, device)

    def observe(self, obs, action, prev_state=None):
        """
        Update latent state representation using observation and action.
        
        Args:
            obs: Current observation (image)
            action: Current action
            prev_state: Previous RSSM state (if None, initialize new state)
            
        Returns:
            posterior_state: Updated state after observing
            prior_state: Predicted state before observing
        """
        # Initialize state if needed
        if prev_state is None:
            prev_state = self.initial_state(obs.shape[0], device=obs.device)
        
        # Reshape observation if needed
        if self.is_image_obs:
            # For image observations: (B, H, W, C) -> (B, C, H, W)
            obs = obs.permute(0, 3, 1, 2)
        else:
            # For vector observations: Add batch dimension if needed
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
        
        # Encode observation
        embed = self.encoder(obs)
        # Ensure embed has correct shape (B, embed_dim)
        if len(embed.shape) == 3:
            embed = embed.squeeze(1)  # Remove extra dimension if present
        
        # Get state distributions
        prior_state = self.rssm.transition(prev_state, action)
        posterior_state = self.rssm.posterior(prev_state, action, embed)
        
        return posterior_state, prior_state

    def imagine(self, prev_state, action):
        """
        Imagine next state without observation (using prior only).
        
        Args:
            prev_state: Previous RSSM state
            action: Action to imagine taking
            
        Returns:
            prior_state: Predicted next state
        """
        prior_state = self.rssm.transition(prev_state, action)
        return prior_state

    def decode(self, state):
        """
        Decode state to observation.
        
        Args:
            state: RSSM state dictionary
            
        Returns:
            obs_reconstruction: Reconstructed observation
        """
        return self.decoder(state['sample'])

    def kl_loss(self, prior_state, posterior_state, mask=None):
        """Compute KL loss between prior and posterior states."""
        return self.rssm.kl_loss(prior_state, posterior_state, mask)

    def get_features(self, state):
        """Get features for policy/value networks."""
        return self.rssm.get_features(state)

    def forward(self, obs, action, prev_state=None):
        """
        Forward pass through world model.
        
        Args:
            obs: Current observation
            action: Current action
            prev_state: Previous RSSM state
            
        Returns:
            posterior_state: Updated state after observing
            prior_state: Predicted state before observing
            obs_reconstruction: Reconstructed observation
        """
        posterior_state, prior_state = self.observe(obs, action, prev_state)
        obs_reconstruction = self.decode(posterior_state)
        return posterior_state, prior_state, obs_reconstruction

    def predict_reward(self, state):
        """Predict reward from latent state."""
        return self.reward_predictor(state['sample'])

class RSSM(nn.Module):
    """
    Recurrent State-Space Model that implements the world model's dynamics. (~ Brain of the World Model)
    
    Architecture:
    1. Deterministic path (belief states): ** Potential TODO for project contribution: Move to Transformer **
        - GRU that processes action and state 
        - Maintains temporal dependencies
    
    2. Stochastic path (state distribution):
        - Prior: p(s_t | h_t)
        - Posterior: q(s_t | h_t, o_t)
        - Both output mean and stddev for state distribution
    
    States contain:
    - mean: Mean of state distribution
    - stddev: Standard deviation of state distribution
    - sample: Sampled state
    - belief: GRU hidden state (deterministic)
    - rnn_state: Internal RNN state
    """
    def __init__(
            self,
            state_size,
            belief_size,
            embed_size,
            action_size,
            hidden_size=200,
            min_stddev=0.1,
            elu=nn.ELU,
            num_layers=1
        ):
        super().__init__()
        self.state_size = state_size      # Stochastic state size
        self.action_size = action_size    # Available actions
        self.belief_size = belief_size    # GRU hidden size
        self.embed_size = embed_size      # Size of embedded observations
        self.min_stddev = min_stddev
        self.num_layers = num_layers
        
        # GRU for deterministic path
        self.rnn = nn.GRUCell(state_size + action_size, belief_size)
        
        # State prior network (transition model)
        self.fc_state_prior = nn.Sequential(
            nn.Linear(belief_size, hidden_size),
            elu(),
            nn.Linear(hidden_size, hidden_size),
            elu(),
            nn.Linear(hidden_size, 2 * state_size)  # mean and stddev
        )
        
        # State posterior network
        self.fc_state_posterior = nn.Sequential(
            nn.Linear(belief_size + embed_size, hidden_size),
            elu(),
            nn.Linear(hidden_size, hidden_size),
            elu(),
            nn.Linear(hidden_size, 2 * state_size)  # mean and stddev
        )

    def initial_state(self, batch_size, device=None):
        """Initialize the recurrent state."""
        return {
            'mean': torch.zeros(batch_size, self.state_size, device=device),
            'stddev': torch.zeros(batch_size, self.state_size, device=device),
            'sample': torch.zeros(batch_size, self.state_size, device=device),
            'belief': torch.zeros(batch_size, self.belief_size, device=device),
            'rnn_state': torch.zeros(batch_size, self.belief_size, device=device)
        }

    def transition(self, prev_state, prev_action):
        """
        Compute prior state distribution p(s_t | h_t) for next timestep.
        """
        # Deterministic state update
        if prev_action is None:
            prev_action = torch.zeros(prev_state['sample'].shape[0], self.action_size, device=prev_state['sample'].device)
        hidden = torch.cat([prev_state['sample'], prev_action], dim=-1)
        belief = self.rnn(hidden, prev_state['rnn_state'])
        
        # State prior
        hidden = self.fc_state_prior(belief)
        mean, stddev = torch.chunk(hidden, 2, dim=-1)
        stddev = F.softplus(stddev) + self.min_stddev
        
        # Sample state
        dist = Normal(mean, stddev)
        sample = dist.rsample()
        
        return {
            'mean': mean,
            'stddev': stddev,
            'sample': sample,
            'belief': belief,
            'rnn_state': belief
        }

    def posterior(self, prev_state, prev_action, embed):
        """
        Compute posterior state distribution q(s_t | h_t, o_t) using observation.
        """
        # Get prior state first
        prior = self.transition(prev_state, prev_action)
        
        # Compute posterior using observation
        hidden = torch.cat([prior['belief'], embed], dim=-1)
        hidden = self.fc_state_posterior(hidden)
        mean, stddev = torch.chunk(hidden, 2, dim=-1)
        stddev = F.softplus(stddev) + self.min_stddev
        
        # Sample state
        dist = Normal(mean, stddev)
        sample = dist.rsample()
        
        return {
            'mean': mean,
            'stddev': stddev,
            'sample': sample,
            'belief': prior['belief'],
            'rnn_state': prior['rnn_state']
        }

    def get_features(self, state):
        """Concatenate belief and state sample for features."""
        return torch.cat([state['sample'], state['belief']], dim=-1)

    def kl_loss(self, prior_state, posterior_state, mask=None):
        """Compute KL divergence between prior and posterior state distributions."""
        prior_dist = Normal(prior_state['mean'], prior_state['stddev'])
        posterior_dist = Normal(posterior_state['mean'], posterior_state['stddev'])
        kl_div = kl_divergence(posterior_dist, prior_dist)
        
        if mask is not None:
            kl_div = kl_div * mask
            
        return kl_div.sum(dim=-1)
