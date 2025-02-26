import random

from collections import deque, namedtuple
import numpy as np
import torch

from .segment_tree import MinSegmentTree, SumSegmentTree


Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "opp_next_state", "reward", "done"),
)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = np.empty([self.capacity], dtype=object)
        self.ptr, self.size = 0, 0

    def __len__(self) -> int:
        return self.size

    def push(self, state, action, opp_action, next_state, reward, done) -> None:
        self.memory[self.ptr] = Transition(state, action, opp_action, next_state, reward, done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def clear(self) -> None:
        self.memory = np.empty([self.capacity], dtype=object)
        self.ptr, self.size = 0, 0

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
        # print(self.ptr)
        return [transition.reward.detach().cpu().item()
                if isinstance(transition.reward, torch.Tensor)
                else transition.reward
                for transition in self.memory[: self.ptr]]

    @property
    def running_avg_rewards(self, window=100):
        window_start = max(0, self.ptr - window)
        return np.mean(self.rewards[-window_start:])

    @property
    def states(self) -> list:
        return [
            (
                transition.state.cpu().tolist()
                if isinstance(transition.state, torch.Tensor)
                else transition.state
            )
            for transition in self.memory[: self.ptr]
        ]


class PrioritizedReplayMemory(ReplayMemory):
    """
    Prioritized Replay Memory.
    Adapted from:
    https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03.per.ipynb
    """

    def __init__(self, capacity, alpha=0.6, **_):
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

        priorities = priorities**self.alpha
        self.max_priority = max(self.max_priority, np.max(priorities))

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority

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


class NStepBuffer:
    """
    N-Step Replay Memory.
    Adapted from:
    https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/07.n_step_learning.ipynb
    """
    def __init__(self, capacity: int, n_steps: int, gamma: float):
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma

        self.memory = ReplayMemory(capacity)

        self.n_step_buffer = deque(maxlen=self.n_steps)

    def push(self, *args) -> Transition:
        transition = Transition(*args)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_steps:
            return

        # make a n-step transition
        reward, next_state, done = self._get_n_step_info(self.n_step_buffer, self.gamma)
        state, action = self.n_step_buffer[0][:2]

        self.memory.push(*(state, action, transition.opp_action, next_state, reward, done))

        return self.n_step_buffer[0]

    def sample_batch(self):
        return self.memory.sample()

    def sample_batch_from_idxs(self, idxs: np.ndarray) -> dict:
        return dict(transitions=self.memory[idxs])

    def _get_n_step_info(self, n_step_buffer: deque, gamma: float):
        """Return n step reward, next_state, and done."""
        # info of the last transition
        _, _, next_state, reward, done = n_step_buffer[-1]

        for transition in reversed(list(n_step_buffer)[:-1]):
            _, _, n_s, r, d = transition

            reward = r + gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done

    def __len__(self) -> int:
        return self.memory.size
