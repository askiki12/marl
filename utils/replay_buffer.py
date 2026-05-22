"""Shared replay buffer utilities.

The experiment entrypoint currently uses algorithm-local placeholder buffers,
but this module is kept import-safe for future reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence, Tuple

import numpy as np


@dataclass
class ReplayBuffer:
    """Simple circular replay buffer placeholder."""

    capacity: int
    storage: List[Tuple[Any, ...]] = field(default_factory=list)
    position: int = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: Tuple[Any, ...]) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Sequence[Tuple[Any, ...]]:
        indices = np.random.choice(len(self.storage), size=batch_size, replace=False)
        return [self.storage[index] for index in indices]