"""Value Decomposition Network (VDN) skeleton for the Switch4-v0 experiment.

This module mirrors the IQL framework shape so the training entrypoint can
switch between IQL and VDN with a unified interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from utils import DQNBase, DQNConfig, ReplayBuffer

try:
	import torch
except ImportError as exc:  # pragma: no cover
	raise ImportError("PyTorch is required for the VDN implementation") from exc


@dataclass
class VDNConfig(DQNConfig):
	pass


class VDNAgent(DQNBase):
	"""Per-agent network bundle used by the centralized VDN trainer."""

	def __init__(self, config: VDNConfig) -> None:
		super().__init__(config, use_replay_buffer=False)


class VDNTrainer:
	"""Multi-agent VDN training scaffold.

	This trainer owns the joint replay buffer and the centralized VDN update.
	"""

	def __init__(self, agent_configs: Sequence[VDNConfig]) -> None:
		self.agents = [VDNAgent(config) for config in agent_configs]
		if not self.agents:
			raise ValueError("VDNTrainer requires at least one agent config")
		self.config = agent_configs[0]
		self.replay_buffer = ReplayBuffer(self.config.buffer_size)

	def _agent_batches(
		self,
		batch: Sequence[tuple[Sequence[np.ndarray], Sequence[int], float, Sequence[np.ndarray], bool]],
	) -> tuple[list[list[np.ndarray]], list[list[int]], list[float], list[list[np.ndarray]], list[bool]]:
		obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
		per_agent_obs = [list(agent_values) for agent_values in zip(*obs_batch)]
		per_agent_actions = [list(agent_values) for agent_values in zip(*action_batch)]
		per_agent_next_obs = [list(agent_values) for agent_values in zip(*next_obs_batch)]
		return per_agent_obs, per_agent_actions, list(reward_batch), per_agent_next_obs, list(done_batch)

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
		joint_reward = float(np.sum(rewards))
		joint_done = bool(all(dones))
		self.replay_buffer.add((list(observations), list(actions), joint_reward, list(next_observations), joint_done))

	def update(self) -> List[Optional[Dict[str, float]]]:
		if len(self.replay_buffer) < self.config.batch_size:
			return [None for _ in self.agents]

		batch = self.replay_buffer.sample(self.config.batch_size)
		per_agent_obs, per_agent_actions, reward_batch, per_agent_next_obs, done_batch = self._agent_batches(batch)

		reward_tensor = torch.as_tensor(np.asarray(reward_batch), dtype=torch.float32, device=self.agents[0].device).unsqueeze(-1)
		done_tensor = torch.as_tensor(np.asarray(done_batch), dtype=torch.float32, device=self.agents[0].device).unsqueeze(-1)

		joint_q_values: List[torch.Tensor] = []
		joint_next_q_values: List[torch.Tensor] = []
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
			q_values = agent.policy_net(obs_tensor).gather(1, action_tensor)
			with torch.no_grad():
				next_q_values = agent.target_net(next_obs_tensor).max(dim=1, keepdim=True).values
			joint_q_values.append(q_values)
			joint_next_q_values.append(next_q_values)

		q_total = torch.stack(joint_q_values, dim=0).sum(dim=0)
		next_q_total = torch.stack(joint_next_q_values, dim=0).sum(dim=0)
		targets = reward_tensor + self.config.gamma * (1.0 - done_tensor) * next_q_total
		loss = torch.nn.functional.mse_loss(q_total, targets)

		for agent in self.agents:
			agent.optimizer.zero_grad()
		loss.backward()
		for agent in self.agents:
			agent.optimizer.step()
			agent.train_steps += 1
			if agent.train_steps % agent.config.target_update_interval == 0:
				agent.target_net.load_state_dict(agent.policy_net.state_dict())
			agent.epsilon = max(agent.config.epsilon_end, agent.epsilon * agent.config.epsilon_decay)

		return [{"loss": float(loss.item()), "epsilon": float(agent.epsilon)} for agent in self.agents]

	def state_dict(self) -> Dict[str, Any]:
		return {f"agent_{idx}": agent.state_dict() for idx, agent in enumerate(self.agents)}

	def load_state_dict(self, state: Dict[str, Any]) -> None:
		for idx, agent in enumerate(self.agents):
			agent.load_state_dict(state[f"agent_{idx}"])
