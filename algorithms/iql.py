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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
	import torch
	import torch.nn as nn
except ImportError as exc:  # pragma: no cover - handled at runtime in the notebook/env
	raise ImportError("PyTorch is required for the IQL implementation") from exc


@dataclass
class IQLConfig:
	"""Hyperparameters for the IQL agent.

	Experiment-level constants should still live in the training entrypoint.
	This config only stores algorithm-specific parameters.
	"""

	obs_dim: int
	action_dim: int
	hidden_dim: int = 128
	gamma: float = 0.99
	lr: float = 1e-3
	batch_size: int = 64
	buffer_size: int = 100_000
	target_update_interval: int = 200
	epsilon_start: float = 1.0
	epsilon_end: float = 0.05
	epsilon_decay: float = 0.995
	device: str = "cpu"


class QNetwork(nn.Module):
	"""Simple MLP used by each independent agent."""

	def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(obs_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, action_dim),
		)

	def forward(self, obs: torch.Tensor) -> torch.Tensor:
		return self.net(obs)


class ReplayBuffer:
	"""Lightweight placeholder replay buffer for the IQL skeleton.

	TODO: replace with a circular buffer implementation and add sampling
	for batched multi-agent transitions.
	"""

	def __init__(self, capacity: int) -> None:
		self.capacity = capacity
		self.storage: List[Tuple[Any, ...]] = []
		self.position = 0

	def __len__(self) -> int:
		return len(self.storage)

	def add(self, transition: Tuple[Any, ...]) -> None:
		if len(self.storage) < self.capacity:
			self.storage.append(transition)
		else:
			self.storage[self.position] = transition
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size: int) -> Tuple[Any, ...]:
		# TODO: return stacked tensors with shapes suitable for training.
		indices = np.random.choice(len(self.storage), size=batch_size, replace=False)
		batch = [self.storage[i] for i in indices]
		return tuple(zip(*batch))


class IQLAgent:
	"""Single-agent IQL learner.

	In Switch4-v0, one instance of this class will typically be created for
	each agent, and each agent learns its own Q-function independently.
	"""

	def __init__(self, config: IQLConfig) -> None:
		self.config = config
		self.device = torch.device(config.device)

		self.policy_net = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
		self.target_net = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr)
		self.replay_buffer = ReplayBuffer(config.buffer_size)
		self.epsilon = config.epsilon_start
		self.train_steps = 0

	def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
		"""Select an action with epsilon-greedy exploration."""

		if (not greedy) and (np.random.rand() < self.epsilon):
			return int(np.random.randint(self.config.action_dim))

		obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		with torch.no_grad():
			q_values = self.policy_net(obs_tensor)
		return int(torch.argmax(q_values, dim=-1).item())

	def store_transition(
		self,
		obs: np.ndarray,
		action: int,
		reward: float,
		next_obs: np.ndarray,
		done: bool,
	) -> None:
		self.replay_buffer.add((obs, action, reward, next_obs, done))

	def update(self) -> Optional[Dict[str, float]]:
		"""Run one gradient update.

		TODO: implement batched Bellman targets, gradient clipping, and
		target-network synchronization.
		"""

		if len(self.replay_buffer) < self.config.batch_size:
			return None

		obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = self.replay_buffer.sample(
			self.config.batch_size
		)

		obs_tensor = torch.as_tensor(np.array(obs_batch), dtype=torch.float32, device=self.device)
		action_tensor = torch.as_tensor(np.array(action_batch), dtype=torch.int64, device=self.device).unsqueeze(-1)
		reward_tensor = torch.as_tensor(np.array(reward_batch), dtype=torch.float32, device=self.device).unsqueeze(-1)
		next_obs_tensor = torch.as_tensor(np.array(next_obs_batch), dtype=torch.float32, device=self.device)
		done_tensor = torch.as_tensor(np.array(done_batch), dtype=torch.float32, device=self.device).unsqueeze(-1)

		q_values = self.policy_net(obs_tensor).gather(1, action_tensor)
		with torch.no_grad():
			next_q_values = self.target_net(next_obs_tensor).max(dim=1, keepdim=True).values
			targets = reward_tensor + self.config.gamma * (1.0 - done_tensor) * next_q_values

		loss = torch.nn.functional.mse_loss(q_values, targets)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.train_steps += 1
		if self.train_steps % self.config.target_update_interval == 0:
			self.target_net.load_state_dict(self.policy_net.state_dict())

		self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
		return {"loss": float(loss.item()), "epsilon": float(self.epsilon)}

	def state_dict(self) -> Dict[str, Any]:
		return {
			"policy_net": self.policy_net.state_dict(),
			"target_net": self.target_net.state_dict(),
			"optimizer": self.optimizer.state_dict(),
			"epsilon": self.epsilon,
			"train_steps": self.train_steps,
		}

	def load_state_dict(self, state: Dict[str, Any]) -> None:
		self.policy_net.load_state_dict(state["policy_net"])
		self.target_net.load_state_dict(state["target_net"])
		self.optimizer.load_state_dict(state["optimizer"])
		self.epsilon = float(state.get("epsilon", self.epsilon))
		self.train_steps = int(state.get("train_steps", self.train_steps))


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

