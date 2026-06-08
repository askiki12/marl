"""QMIX implementation for the Switch4-v0 experiment.

The environment used in this project does not expose a dedicated global state
API, so the trainer builds the mixing-network state from the concatenated
agent observations collected at each transition.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
	import torch
	import torch.nn as nn
except ImportError as exc:  # pragma: no cover
	raise ImportError("PyTorch is required for the QMIX implementation") from exc

from utils import DQNBase, DQNConfig, ReplayBuffer


@dataclass
class QMIXConfig(DQNConfig):
	state_dim: int = 0
	mixing_hidden_dim: int = 32
	hypernet_hidden_dim: int = 64
	mixer_target_update_interval: int = 200


class QMIXAgent(DQNBase):
	"""Per-agent DQN learner used by the centralized QMIX trainer."""

	def __init__(self, config: QMIXConfig) -> None:
		super().__init__(config, use_replay_buffer=False)


class QMixer(nn.Module):
	"""Monotonic mixing network used by QMIX."""

	def __init__(self, n_agents: int, state_dim: int, mixing_hidden_dim: int, hypernet_hidden_dim: int) -> None:
		super().__init__()
		self.n_agents = n_agents
		self.state_dim = state_dim
		self.mixing_hidden_dim = mixing_hidden_dim
		self.hypernet_hidden_dim = hypernet_hidden_dim

		self.hyper_w1 = nn.Sequential(
			nn.Linear(state_dim, hypernet_hidden_dim),
			nn.ReLU(),
			nn.Linear(hypernet_hidden_dim, n_agents * mixing_hidden_dim),
		)
		self.hyper_b1 = nn.Linear(state_dim, mixing_hidden_dim)
		self.hyper_w2 = nn.Sequential(
			nn.Linear(state_dim, hypernet_hidden_dim),
			nn.ReLU(),
			nn.Linear(hypernet_hidden_dim, mixing_hidden_dim),
		)
		self.hyper_b2 = nn.Sequential(
			nn.Linear(state_dim, hypernet_hidden_dim),
			nn.ReLU(),
			nn.Linear(hypernet_hidden_dim, 1),
		)

	def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
		batch_size = agent_qs.size(0)
		w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.n_agents, self.mixing_hidden_dim)
		b1 = self.hyper_b1(states).view(batch_size, 1, self.mixing_hidden_dim)
		hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
		hidden = torch.nn.functional.elu(hidden)
		w2 = torch.abs(self.hyper_w2(states)).view(batch_size, self.mixing_hidden_dim, 1)
		b2 = self.hyper_b2(states).view(batch_size, 1, 1)
		q_total = torch.bmm(hidden, w2) + b2
		return q_total.view(batch_size, 1)


class QMIXTrainer:
	"""Multi-agent QMIX trainer with a joint replay buffer and target mixer."""

	def __init__(self, agent_configs: Sequence[QMIXConfig]) -> None:
		self.agents = [QMIXAgent(config) for config in agent_configs]
		if not self.agents:
			raise ValueError("QMIXTrainer requires at least one agent config")
		self.config = agent_configs[0]
		self.n_agents = len(self.agents)
		self.replay_buffer = ReplayBuffer(self.config.buffer_size)
		self.mixer = QMixer(
			n_agents=self.n_agents,
			state_dim=self.config.state_dim,
			mixing_hidden_dim=self.config.mixing_hidden_dim,
			hypernet_hidden_dim=self.config.hypernet_hidden_dim,
		).to(self.agents[0].device)
		self.target_mixer = copy.deepcopy(self.mixer)
		self.mixer_optimizer = torch.optim.Adam(self.mixer.parameters(), lr=self.config.lr)
		self.train_steps = 0

	def _agent_batches(
		self,
		batch: Sequence[tuple[Sequence[np.ndarray], Sequence[int], float, Sequence[np.ndarray], bool]],
	) -> tuple[list[list[np.ndarray]], list[list[int]], list[float], list[list[np.ndarray]], list[bool]]:
		obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
		per_agent_obs = [list(agent_values) for agent_values in zip(*obs_batch)]
		per_agent_actions = [list(agent_values) for agent_values in zip(*action_batch)]
		per_agent_next_obs = [list(agent_values) for agent_values in zip(*next_obs_batch)]
		return per_agent_obs, per_agent_actions, list(reward_batch), per_agent_next_obs, list(done_batch)

	def _joint_states(self, observations_batch: Sequence[Sequence[np.ndarray]]) -> torch.Tensor:
		joint_states = [np.concatenate([np.asarray(obs).reshape(-1) for obs in observations], axis=0) for observations in observations_batch]
		return torch.as_tensor(np.asarray(joint_states), dtype=torch.float32, device=self.agents[0].device)

	def _states_from_batch(self, state_batch: Sequence[np.ndarray] | None, observations_batch: Sequence[Sequence[np.ndarray]]) -> torch.Tensor:
		if state_batch is not None:
			state_array = np.asarray([np.asarray(state, dtype=np.float32).reshape(-1) for state in state_batch], dtype=np.float32)
			return torch.as_tensor(state_array, dtype=torch.float32, device=self.agents[0].device)
		return self._joint_states(observations_batch)

	def act(self, observations: Sequence[np.ndarray], greedy: bool = False) -> List[int]:
		return [agent.select_action(obs, greedy=greedy) for agent, obs in zip(self.agents, observations)]

	def observe(
		self,
		observations: Sequence[np.ndarray],
		actions: Sequence[int],
		rewards: Sequence[float],
		next_observations: Sequence[np.ndarray],
		dones: Sequence[bool],
		state: Optional[np.ndarray] = None,
		next_state: Optional[np.ndarray] = None,
	) -> None:
		joint_reward = float(np.sum(rewards))
		joint_done = bool(all(dones))
		if state is not None and next_state is not None:
			self.replay_buffer.add(
				(
					list(observations),
					list(actions),
					joint_reward,
					list(next_observations),
					joint_done,
					np.asarray(state, dtype=np.float32),
					np.asarray(next_state, dtype=np.float32),
				)
			)
		else:
			self.replay_buffer.add((list(observations), list(actions), joint_reward, list(next_observations), joint_done))

	def update(self) -> List[Optional[Dict[str, float]]]:
		if len(self.replay_buffer) < self.config.batch_size:
			return [None for _ in self.agents]

		batch = self.replay_buffer.sample(self.config.batch_size)
		state_batch: Sequence[np.ndarray] | None = None
		next_state_batch: Sequence[np.ndarray] | None = None
		if batch and len(batch[0]) == 7:
			obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, state_batch, next_state_batch = zip(*batch)
			per_agent_obs = [list(agent_values) for agent_values in zip(*obs_batch)]
			per_agent_actions = [list(agent_values) for agent_values in zip(*action_batch)]
			per_agent_next_obs = [list(agent_values) for agent_values in zip(*next_obs_batch)]
		else:
			per_agent_obs, per_agent_actions, reward_batch, per_agent_next_obs, done_batch = self._agent_batches(batch)
		states = self._states_from_batch(state_batch, list(zip(*per_agent_obs)))
		next_states = self._states_from_batch(next_state_batch, list(zip(*per_agent_next_obs)))
		reward_tensor = torch.as_tensor(np.asarray(reward_batch), dtype=torch.float32, device=self.agents[0].device).unsqueeze(-1)
		done_tensor = torch.as_tensor(np.asarray(done_batch), dtype=torch.float32, device=self.agents[0].device).unsqueeze(-1)

		chosen_qs: List[torch.Tensor] = []
		target_next_qs: List[torch.Tensor] = []
		for agent_index, agent in enumerate(self.agents):
			obs_tensor, action_tensor, _, next_obs_tensor, _ = agent._transition_to_tensors(
				list(
					zip(
						per_agent_obs[agent_index],
						per_agent_actions[agent_index],
						reward_batch,
						per_agent_next_obs[agent_index],
						done_batch,
					)
				)
			)
			chosen_qs.append(agent.policy_net(obs_tensor).gather(1, action_tensor))
			with torch.no_grad():
				target_next_qs.append(agent.target_net(next_obs_tensor).max(dim=1, keepdim=True).values)

		agent_qs = torch.cat(chosen_qs, dim=1)
		next_agent_qs = torch.cat(target_next_qs, dim=1)
		q_total = self.mixer(agent_qs, states)
		with torch.no_grad():
			target_q_total = self.target_mixer(next_agent_qs, next_states)
			targets = reward_tensor + self.config.gamma * (1.0 - done_tensor) * target_q_total
		loss = torch.nn.functional.mse_loss(q_total, targets)

		for agent in self.agents:
			agent.optimizer.zero_grad()
		self.mixer_optimizer.zero_grad()
		loss.backward()
		for agent in self.agents:
			agent.optimizer.step()
			agent.train_steps += 1
			if agent.train_steps % agent.config.target_update_interval == 0:
				agent.target_net.load_state_dict(agent.policy_net.state_dict())
			agent.epsilon = max(agent.config.epsilon_end, agent.epsilon * agent.config.epsilon_decay)
		self.mixer_optimizer.step()

		self.train_steps += 1
		if self.train_steps % self.config.mixer_target_update_interval == 0:
			self.target_mixer.load_state_dict(self.mixer.state_dict())

		return [{"loss": float(loss.item()), "epsilon": float(agent.epsilon)} for agent in self.agents]

	def state_dict(self) -> Dict[str, Any]:
		return {
			"agents": {f"agent_{idx}": agent.state_dict() for idx, agent in enumerate(self.agents)},
			"mixer": self.mixer.state_dict(),
			"target_mixer": self.target_mixer.state_dict(),
			"mixer_optimizer": self.mixer_optimizer.state_dict(),
			"train_steps": self.train_steps,
		}

	def load_state_dict(self, state: Dict[str, Any]) -> None:
		agent_state = state.get("agents", state)
		for idx, agent in enumerate(self.agents):
			agent.load_state_dict(agent_state[f"agent_{idx}"])
		if "mixer" in state:
			self.mixer.load_state_dict(state["mixer"])
		if "target_mixer" in state:
			self.target_mixer.load_state_dict(state["target_mixer"])
		if "mixer_optimizer" in state:
			self.mixer_optimizer.load_state_dict(state["mixer_optimizer"])
		self.train_steps = int(state.get("train_steps", self.train_steps))
