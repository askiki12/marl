"""Independent Q-Learning (IQL) skeleton for the Switch4-v0 experiment.

This module keeps algorithm-specific logic only.
Experiment-level settings such as:
- env name (Switch4-v0)
- algorithm selection (IQL / VDN)
- training steps / episodes
- evaluation episodes
- seed list

should be defined in ``train.py`` or a dedicated config layer, not in
``__init__.py``. The package initializer should only expose imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
	import torch
except ImportError as exc:  # pragma: no cover - handled at runtime in the notebook/env
	raise ImportError("PyTorch is required for the IQL implementation") from exc

from utils import DQNBase, DQNConfig


@dataclass
class IQLConfig(DQNConfig):
	"""Hyperparameters for the IQL agent.

	Experiment-level constants should still live in the training entrypoint.
	This config only stores algorithm-specific parameters.
	"""


class IQLAgent(DQNBase):
	"""Single-agent IQL learner.

	In Switch4-v0, one instance of this class will typically be created for
	each agent, and each agent learns its own Q-function independently.
	"""

	def __init__(self, config: IQLConfig) -> None:
		super().__init__(config)

	def store_transition(
		self,
		obs: np.ndarray,
		action: int,
		reward: float,
		next_obs: np.ndarray,
		done: bool,
	) -> None:
		if self.replay_buffer is None:
			raise RuntimeError("IQLAgent requires a replay buffer")
		self.replay_buffer.add((obs, action, reward, next_obs, done))

	def update(self) -> Optional[Dict[str, float]]:
		"""Run one gradient update.

		TODO: implement batched Bellman targets, gradient clipping, and
		target-network synchronization.
		"""
		if self.replay_buffer is None:
			return None
		return self._update_from_buffer(self.replay_buffer)


class IQLTrainer:
	"""Multi-agent training scaffold for Switch4-v0.

	TODO: implement the environment loop here or in train.py:
	- create env with gym.make("ma_gym:Switch4-v0")
	- split observations per agent
	- collect transitions for each agent
	- log training metrics
	- run periodic evaluation
	- save model checkpoints
	"""

	def __init__(self, agent_configs: Sequence[IQLConfig]) -> None:
		self.agents = [IQLAgent(config) for config in agent_configs]

	def act(self, observations: Sequence[np.ndarray], greedy: bool = False) -> List[int]:
		return [agent.select_action(obs, greedy=greedy) for agent, obs in zip(self.agents, observations)]

	def observe(
		self,
		observations: Sequence[np.ndarray],
		actions: Sequence[int],
		rewards: Sequence[float],
		next_observations: Sequence[np.ndarray],
		dones: Sequence[bool],
	) -> None:
		for agent, obs, action, reward, next_obs, done in zip(
			self.agents, observations, actions, rewards, next_observations, dones
		):
			agent.store_transition(obs, action, reward, next_obs, done)

	def update(self) -> List[Optional[Dict[str, float]]]:
		return [agent.update() for agent in self.agents]

	def state_dict(self) -> Dict[str, Any]:
		return {f"agent_{idx}": agent.state_dict() for idx, agent in enumerate(self.agents)}

	def load_state_dict(self, state: Dict[str, Any]) -> None:
		for idx, agent in enumerate(self.agents):
			agent.load_state_dict(state[f"agent_{idx}"])

