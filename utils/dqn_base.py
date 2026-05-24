"""Shared DQN building blocks for MARL algorithms.

This module keeps the duplicated agent scaffolding out of algorithm-specific
implementations so IQL and VDN can share a consistent network/config layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
	import torch
	import torch.nn as nn
except ImportError as exc:  # pragma: no cover
	raise ImportError("PyTorch is required for the DQN base implementation") from exc

from .replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
	"""Common hyperparameters for DQN-style agents."""

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
	"""Shared MLP used by DQN-style agents."""

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


class DQNBase:
	"""Common DQN agent state and utility methods.

	Algorithm-specific agents reuse this class for the model, target-network,
	epsilon schedule, and serialization logic.
	"""

	def __init__(self, config: DQNConfig, *, use_replay_buffer: bool = True) -> None:
		self.config = config
		self.device = torch.device(config.device)
		self.policy_net = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
		self.target_net = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr)
		self.replay_buffer = ReplayBuffer(config.buffer_size) if use_replay_buffer else None
		self.epsilon = config.epsilon_start
		self.train_steps = 0

	def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
		if (not greedy) and (np.random.rand() < self.epsilon):
			return int(np.random.randint(self.config.action_dim))

		obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		with torch.no_grad():
			q_values = self.policy_net(obs_tensor)
		return int(torch.argmax(q_values, dim=-1).item())

	def _transition_to_tensors(
		self,
		batch: Sequence[Tuple[Any, ...]],
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
		obs_tensor = torch.as_tensor(np.array(obs_batch), dtype=torch.float32, device=self.device)
		action_tensor = torch.as_tensor(np.array(action_batch), dtype=torch.int64, device=self.device).unsqueeze(-1)
		reward_tensor = torch.as_tensor(np.array(reward_batch), dtype=torch.float32, device=self.device).unsqueeze(-1)
		next_obs_tensor = torch.as_tensor(np.array(next_obs_batch), dtype=torch.float32, device=self.device)
		done_tensor = torch.as_tensor(np.array(done_batch), dtype=torch.float32, device=self.device).unsqueeze(-1)
		return obs_tensor, action_tensor, reward_tensor, next_obs_tensor, done_tensor

	def _standard_dqn_loss(self, batch: Sequence[Tuple[Any, ...]]) -> torch.Tensor:
		obs_tensor, action_tensor, reward_tensor, next_obs_tensor, done_tensor = self._transition_to_tensors(batch)
		q_values = self.policy_net(obs_tensor).gather(1, action_tensor)
		with torch.no_grad():
			next_q_values = self.target_net(next_obs_tensor).max(dim=1, keepdim=True).values
			targets = reward_tensor + self.config.gamma * (1.0 - done_tensor) * next_q_values
		return torch.nn.functional.mse_loss(q_values, targets)

	def _optimize_loss(self, loss: torch.Tensor) -> Dict[str, float]:
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.train_steps += 1
		if self.train_steps % self.config.target_update_interval == 0:
			self.target_net.load_state_dict(self.policy_net.state_dict())

		self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
		return {"loss": float(loss.item()), "epsilon": float(self.epsilon)}

	def _update_from_buffer(self, buffer: ReplayBuffer) -> Optional[Dict[str, float]]:
		if len(buffer) < self.config.batch_size:
			return None

		batch = buffer.sample(self.config.batch_size)
		loss = self._standard_dqn_loss(batch)
		return self._optimize_loss(loss)

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


__all__ = ["DQNBase", "DQNConfig", "QNetwork", "ReplayBuffer"]