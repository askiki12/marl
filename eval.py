"""Evaluation entrypoint for the Switch4-v0 MARL experiment.

This script loads saved checkpoints from ``results/models/<algorithm>/seed_<seed>``
and reports test metrics aggregated across seeds:
- episodic return mean
- task completion rate
- policy stability (return std)
- average task completion steps
- agent collision rate
- path planning efficiency

The default task is Switch4-v0, but the checkpoint loading path, seeds, and
episode count can be configured from the command line.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

try:
	from tqdm.auto import trange
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
	def trange(*args, **kwargs):
		return range(*args)

try:
	import gym
except ImportError as exc:  # pragma: no cover
	raise ImportError("Gym is required to run evaluation") from exc

try:
	import torch
except ImportError as exc:  # pragma: no cover
	raise ImportError("PyTorch is required to run evaluation") from exc

from algorithms import IQLConfig, IQLTrainer, VDNConfig, VDNTrainer
from utils import ensure_directory, save_json


FIXED_ENV_NAME = "Switch4-v0"
FIXED_ALGORITHMS = ("iql", "vdn")
DEFAULT_EVAL_EPISODES = 100
DEFAULT_CHECKPOINT_NAME = "best.pt"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Switch4-v0 MARL evaluation entrypoint")
	parser.add_argument("--algorithm", choices=("iql", "vdn", "all"), default="all", help="Algorithm to evaluate")
	parser.add_argument("--env-name", default=FIXED_ENV_NAME, help="Environment name")
	parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4], help="Random seed list")
	parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES, help="Evaluation episodes per seed")
	parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME, help="Checkpoint filename to load")
	parser.add_argument("--results-root", default=None, help="Results root directory containing models/")
	parser.add_argument("--output-dir", default=None, help="Directory to write evaluation summaries")
	parser.add_argument("--device", default="cpu", help="Torch device")
	return parser.parse_args()


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def make_env(env_name: str):
	return gym.make(f"ma_gym:{env_name}")


def _unpack_reset_result(reset_result: Any) -> List[np.ndarray]:
	return [np.asarray(obs, dtype=np.float32).reshape(-1) for obs in reset_result]


def _unpack_step_result(step_result: Any) -> tuple[list[np.ndarray], list[float], list[bool], dict]:
	next_observations, rewards, dones, info = step_result
	return (
		[np.asarray(obs, dtype=np.float32).reshape(-1) for obs in next_observations],
		[float(reward) for reward in rewards],
		[bool(done) for done in dones],
		dict(info),
	)


def infer_obs_action_dims(env) -> Dict[str, int]:
	obs_space = env.observation_space
	action_space = env.action_space

	if hasattr(obs_space, "shape") and obs_space.shape is not None:
		obs_dim = int(np.prod(obs_space.shape))
	else:
		sample_obs = env.reset()
		obs_dim = int(np.asarray(sample_obs).shape[-1])

	if hasattr(action_space, "n"):
		action_dim = int(action_space.n)
	else:
		action_dim = int(action_space[0].n)

	return {"obs_dim": obs_dim, "action_dim": action_dim}


def build_trainer(algorithm: str, env, device: str) -> object:
	dims = infer_obs_action_dims(env)
	agent_count = int(env.n_agents)
	if algorithm == "iql":
		agent_configs = [
			IQLConfig(obs_dim=dims["obs_dim"], action_dim=dims["action_dim"], device=device)
			for _ in range(agent_count)
		]
		return IQLTrainer(agent_configs)
	if algorithm == "vdn":
		agent_configs = [
			VDNConfig(obs_dim=dims["obs_dim"], action_dim=dims["action_dim"], device=device)
			for _ in range(agent_count)
		]
		return VDNTrainer(agent_configs)
	raise ValueError(f"Unsupported algorithm: {algorithm}")


def resolve_results_root(results_root: str | None) -> Path:
	if results_root is not None:
		return Path(results_root)

	candidates = [
		Path.cwd() / "results",
		Path(__file__).resolve().parent / "results",
		Path(__file__).resolve().parent.parent / "results",
		Path(__file__).resolve().parent.parent.parent / "results",
	]
	for candidate in candidates:
		if (candidate / "models").exists():
			return candidate
	return candidates[0]


def load_checkpoint(trainer: object, checkpoint_path: Path, device: str) -> None:
	state = torch.load(checkpoint_path, map_location=device)
	if not hasattr(trainer, "load_state_dict"):
		raise TypeError("Trainer does not support load_state_dict")
	trainer.load_state_dict(state)


def _switch_shortest_path_length(base_grid: np.ndarray, start: Sequence[int], goal: Sequence[int]) -> int:
	start_node = (int(start[0]), int(start[1]))
	goal_node = (int(goal[0]), int(goal[1]))
	if start_node == goal_node:
		return 0

	rows, cols = base_grid.shape
	queue = deque([(start_node, 0)])
	visited = {start_node}
	directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

	while queue:
		(node_row, node_col), distance = queue.popleft()
		for delta_row, delta_col in directions:
			next_row = node_row + delta_row
			next_col = node_col + delta_col
			next_node = (next_row, next_col)
			if not (0 <= next_row < rows and 0 <= next_col < cols):
				continue
			if base_grid[next_row, next_col] == -1:
				continue
			if next_node in visited:
				continue
			if next_node == goal_node:
				return distance + 1
			visited.add(next_node)
			queue.append((next_node, distance + 1))

	raise RuntimeError(f"No path found between {start_node} and {goal_node}")


def _episode_success(env) -> bool:
	if hasattr(env, "agent_pos") and hasattr(env, "final_agent_pos"):
		for agent_index in range(env.n_agents):
			if list(env.agent_pos[agent_index]) != list(env.final_agent_pos[agent_index]):
				return False
		return True
	return False


def _episode_optimal_makespan(env) -> float:
	if not hasattr(env, "_base_grid") or not hasattr(env, "init_agent_pos") or not hasattr(env, "final_agent_pos"):
		return float("nan")
	shortest_paths = []
	for agent_index in range(env.n_agents):
		shortest_paths.append(
			_switch_shortest_path_length(env._base_grid, env.init_agent_pos[agent_index], env.final_agent_pos[agent_index])
		)
	return float(max(shortest_paths)) if shortest_paths else float("nan")


def _evaluate_single_episode(env, trainer: object) -> Dict[str, float]:
	observations = _unpack_reset_result(env.reset())
	done_n = [False] * env.n_agents
	episode_return = 0.0
	episode_length = 0
	blocked_moves = 0
	active_move_attempts = 0

	while not all(done_n):
		prev_positions = {agent_index: list(position) for agent_index, position in getattr(env, "agent_pos", {}).items()}
		actions = trainer.act(observations, greedy=True)
		for agent_index, action in enumerate(actions):
			if not done_n[agent_index] and action != 4:
				active_move_attempts += 1
		step_result = env.step(actions)
		next_observations, rewards, done_n, _ = _unpack_step_result(step_result)

		for agent_index, action in enumerate(actions):
			if done_n[agent_index]:
				continue
			if action == 4:
				continue
			if agent_index in prev_positions and list(env.agent_pos[agent_index]) == prev_positions[agent_index]:
				blocked_moves += 1

		episode_return += float(np.sum(rewards))
		episode_length += 1
		observations = next_observations

	success = _episode_success(env)
	completion_steps = float(episode_length if success else np.nan)
	optimal_makespan = _episode_optimal_makespan(env)
	if np.isfinite(optimal_makespan) and episode_length > 0:
		path_efficiency = float(optimal_makespan / episode_length)
	else:
		path_efficiency = float(np.nan)

	return {
		"episode_return": float(episode_return),
		"episode_length": float(episode_length),
		"success": float(success),
		"collision_rate": float(blocked_moves / active_move_attempts) if active_move_attempts else 0.0,
		"completion_steps": completion_steps,
		"path_efficiency": path_efficiency,
	}


def _nanmean(values: Sequence[float]) -> float:
	filtered = [value for value in values if np.isfinite(value)]
	return float(np.mean(filtered)) if filtered else float("nan")


def _summarize_seed_metrics(episode_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
	returns = [metric["episode_return"] for metric in episode_metrics]
	successes = [metric["success"] for metric in episode_metrics]
	completion_steps = [metric["completion_steps"] for metric in episode_metrics]
	collision_rates = [metric["collision_rate"] for metric in episode_metrics]
	path_efficiencies = [metric["path_efficiency"] for metric in episode_metrics]

	return {
		"eval_return_mean": float(np.mean(returns)) if returns else 0.0,
		"eval_return_std": float(np.std(returns)) if returns else 0.0,
		"task_completion_rate": float(np.mean(successes)) if successes else 0.0,
		"policy_stability": float(np.std(returns)) if returns else 0.0,
		"avg_completion_steps": _nanmean(completion_steps),
		"collision_rate": float(np.mean(collision_rates)) if collision_rates else 0.0,
		"path_planning_efficiency": _nanmean(path_efficiencies),
		"successful_episodes": float(np.sum(successes)) if successes else 0.0,
		"total_episodes": float(len(episode_metrics)),
	}


def _to_serializable(value: Any) -> Any:
	if isinstance(value, float) and np.isnan(value):
		return None
	if isinstance(value, np.floating) and np.isnan(value):
		return None
	if isinstance(value, dict):
		return {key: _to_serializable(subvalue) for key, subvalue in value.items()}
	if isinstance(value, list):
		return [_to_serializable(item) for item in value]
	return value


def evaluate_seed(algorithm: str, env_name: str, seed: int, episodes: int, checkpoint_path: Path, device: str) -> Dict[str, Any]:
	set_seed(seed)
	env = make_env(env_name)
	trainer = build_trainer(algorithm, env, device)
	load_checkpoint(trainer, checkpoint_path, device)

	episode_metrics = []
	for _ in trange(episodes, desc=f"{algorithm} seed={seed}", leave=False, unit="ep"):
		episode_metrics.append(_evaluate_single_episode(env, trainer))
	env.close()

	seed_summary = _summarize_seed_metrics(episode_metrics)
	seed_summary.update(
		{
			"seed": seed,
			"algorithm": algorithm,
			"env_name": env_name,
			"checkpoint_path": str(checkpoint_path),
		}
	)
	return {"seed_summary": seed_summary, "episode_metrics": episode_metrics}


def _aggregate_across_seeds(seed_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
	metric_keys = [
		"eval_return_mean",
		"eval_return_std",
		"task_completion_rate",
		"policy_stability",
		"avg_completion_steps",
		"collision_rate",
		"path_planning_efficiency",
	]
	aggregated: Dict[str, Dict[str, float]] = {}
	for metric_key in metric_keys:
		values = [summary[metric_key] for summary in seed_summaries if metric_key in summary and np.isfinite(summary[metric_key])]
		aggregated[metric_key] = {
			"mean": float(np.mean(values)) if values else float("nan"),
			"std": float(np.std(values)) if values else float("nan"),
		}
	return aggregated


def _discover_checkpoint_path(results_root: Path, algorithm: str, seed: int, checkpoint_name: str) -> Path | None:
	checkpoint_path = results_root / "models" / algorithm / f"seed_{seed}" / checkpoint_name
	if checkpoint_path.exists():
		return checkpoint_path
	return None


def _write_seed_csv(output_path: Path, seed_summaries: Sequence[Dict[str, Any]]) -> None:
	fieldnames = [
		"seed",
		"algorithm",
		"env_name",
		"checkpoint_path",
		"eval_return_mean",
		"eval_return_std",
		"task_completion_rate",
		"policy_stability",
		"avg_completion_steps",
		"collision_rate",
		"path_planning_efficiency",
		"successful_episodes",
		"total_episodes",
	]
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8", newline="") as file_handle:
		writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
		writer.writeheader()
		for summary in seed_summaries:
			writer.writerow({field: _to_serializable(summary.get(field)) for field in fieldnames})


def run_evaluation(algorithm: str, env_name: str, seeds: Sequence[int], episodes: int, checkpoint_name: str, results_root: str | None, output_dir: str | None, device: str) -> Dict[str, Any]:
	root = resolve_results_root(results_root)
	output_root = Path(output_dir) if output_dir is not None else root / "eval"
	ensure_directory(output_root)

	algorithms = FIXED_ALGORITHMS if algorithm == "all" else (algorithm,)
	final_report: Dict[str, Any] = {
		"env_name": env_name,
		"episodes_per_seed": episodes,
		"checkpoint_name": checkpoint_name,
		"results_root": str(root),
		"output_root": str(output_root),
		"algorithms": {},
	}

	for algo in algorithms:
		seed_summaries: List[Dict[str, Any]] = []
		episode_metrics_by_seed: Dict[str, List[Dict[str, float]]] = {}
		missing_seeds: List[int] = []

		for seed in trange(len(seeds), desc=f"{algo} seeds", unit="seed"):
			seed = seeds[seed]
			checkpoint_path = _discover_checkpoint_path(root, algo, seed, checkpoint_name)
			if checkpoint_path is None:
				missing_seeds.append(seed)
				continue
			seed_result = evaluate_seed(algo, env_name, seed, episodes, checkpoint_path, device)
			seed_summary = seed_result["seed_summary"]
			seed_summaries.append(seed_summary)
			episode_metrics_by_seed[str(seed)] = seed_result["episode_metrics"]

		aggregated = _aggregate_across_seeds(seed_summaries)
		algo_report = {
			"seed_summaries": seed_summaries,
			"aggregate": aggregated,
			"missing_seeds": missing_seeds,
		}
		final_report["algorithms"][algo] = algo_report

		algo_output_dir = ensure_directory(output_root / algo)
		save_json(algo_output_dir / "summary.json", _to_serializable(algo_report))
		save_json(algo_output_dir / "episodes.json", _to_serializable(episode_metrics_by_seed))
		_write_seed_csv(algo_output_dir / "seed_summary.csv", seed_summaries)

	return final_report


def main() -> None:
	args = parse_args()
	report = run_evaluation(
		algorithm=args.algorithm,
		env_name=args.env_name,
		seeds=args.seeds,
		episodes=args.episodes,
		checkpoint_name=args.checkpoint_name,
		results_root=args.results_root,
		output_dir=args.output_dir,
		device=args.device,
	)
	print(json.dumps(_to_serializable(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
