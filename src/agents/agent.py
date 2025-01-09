import collections

from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self):
        self.steps_done = 0
        self.losses = []

    @abstractmethod
    def record(self, state, action, next_state, reward, done):
        raise NotImplementedError

    @abstractmethod
    def select_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def train_episode(self):
        raise NotImplementedError

    @abstractmethod
    def optimize(self):
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, torch_state_dict):
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> collections.OrderedDict:
        raise NotImplementedError
