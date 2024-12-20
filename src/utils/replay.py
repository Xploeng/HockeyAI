import random

from collections import deque, namedtuple
import torch


Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "reward", "done"),
)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def clear(self) -> None:
        self.memory.clear()

    def sample(self, batch_size) -> list[namedtuple]:
        return random.sample(self.memory, batch_size)

    @property
    def rewards(self) -> list[float]:
        return [transition.reward for transition in self.memory]

    @property
    def states(self) -> list:
        return [
            transition.state.cpu().tolist() if isinstance(transition.state, torch.Tensor) else transition.state
            for transition in self.memory
        ]
