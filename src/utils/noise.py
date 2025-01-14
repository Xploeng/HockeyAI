import numpy as np


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise:
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.05, decay_period=1000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def select_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# PinkNoise class for exploration
class PinkNoise:
    def __init__(self, action_space, beta=1.0, scale=0.2):
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.beta = beta  # Pink noise parameter (1/f^beta)
        self.scale = scale  # Scaling factor for noise
        self.noise = np.zeros(self.action_dim)

    def select_action(self, action):
        """
        Add pink noise to the action.

        # ? Macht fft überhaupt sinn wenn der action_space nur 1D ist?
        # ? fft von einer Zahl ist genau die selbe Zahl (nur mit 0 als Imaginärteil)
        # ? Heißt im pendulum env macht es keinen Sinn, aber in anderen envs schon?
        # ? Oder bin ich komplett lost gerade?
        """
        white_noise = np.random.randn(self.action_dim)
        # print(white_noise)
        fft = np.fft.rfft(white_noise)
        # print(fft)
        freqs = np.arange(1, len(white_noise) + 1)
        # print(freqs)
        fft_filtered = fft / freqs ** self.beta
        # print(fft_filtered)
        pink_noise = np.fft.irfft(fft_filtered, n=len(white_noise))
        # pink_noise = np.fft.irfft(np.fft.rfft(white_noise) / np.arange(1, len(white_noise) + 1) ** self.beta)
        pink_noise = pink_noise[: self.action_dim]  # Ensure the noise has the correct dimension
        self.noise = self.scale * pink_noise
        return np.clip(action + self.noise, self.low, self.high)

    def reset(self):
        self.noise = np.zeros(self.action_dim)
