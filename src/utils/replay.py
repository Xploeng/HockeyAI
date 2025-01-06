import random

from collections import namedtuple
import numpy as np
import torch

from .segment_tree import MinSegmentTree, SumSegmentTree


Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "reward", "done"),
)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = np.empty([self.capacity], dtype=object)
        self.ptr, self.size = 0, 0

    def __len__(self) -> int:
        return self.size

    def push(self, *args) -> None:
        self.memory[self.ptr] = Transition(*args)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def clear(self) -> None:
        self.memory = np.array([self.capacity], dtype=object)

    def sample(self, batch_size) -> dict:
        return dict(
            transitions=np.random.choice(self.memory[: self.ptr], batch_size).tolist(),
        )

    def __getitem__(self, idx) -> namedtuple:
        if isinstance(idx, int):
            return self.memory[idx]
        else:
            return self.memory[idx].tolist()

    @property
    def rewards(self) -> list[float]:
        return [transition.reward for transition in self.memory]

    @property
    def states(self) -> list:
        return [
            (
                transition.state.cpu().tolist()
                if isinstance(transition.state, torch.Tensor)
                else transition.state
            )
            for transition in self.memory
        ]


class PrioritizedReplayMemory(ReplayMemory):
    """
    Prioritized Replay Memory.
    Adapted from:
    https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03.per.ipynb
    """

    def __init__(self, capacity, alpha=0.6):
        super().__init__(capacity)
        self.alpha = alpha
        self.max_priority, self.tree_ptr = 1.0, 0

        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def push(self, *args) -> None:
        super().push(*args)
        self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.capacity

    def sample(self, batch_size, beta: float = 0.4):
        assert len(self) >= batch_size, "Not enough samples in the memory"
        assert beta > 0, "Beta should be greater than 0"

        indices = self._sample_proportional(batch_size)

        transitions = self.memory[indices]

        weigths = np.array([self._calculate_weight(idx, beta) for idx in indices])

        return dict(transitions=transitions, weights=weigths, indices=indices)

    def update_priorities(self, indices: list[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size) -> list[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight
