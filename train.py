"""Training entrypoint for the Switch4-v0 MARL homework experiment.

Fixed experiment settings live here, not inside algorithm packages:
- environment: Switch4-v0
- algorithms: IQL, VDN
- training episodes: fixed below
- evaluation episodes: fixed below
- random seed list: fixed below
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import shutil

try:
	from tqdm.auto import trange
except Exception:  # pragma: no cover - fallback when tqdm isn't installed
	def trange(*args, **kwargs):
		# Fallback to built-in range when tqdm is unavailable.
		return range(*args)

try:
	import gym
except ImportError as exc:  # pragma: no cover
	raise ImportError("Gym is required to run the experiment entrypoint") from exc

try:
	import torch
except ImportError as exc:  # pragma: no cover
	raise ImportError("PyTorch is required to run the experiment entrypoint") from exc

from algorithms import IQLConfig, IQLTrainer, VDNConfig, VDNTrainer
from utils import append_jsonl, ensure_directory, plot_learning_curves, save_checkpoint, save_json


FIXED_ENV_NAME = "Switch4-v0"
FIXED_ALGORITHMS = ("iql", "vdn")
FIXED_TRAIN_EPISODES = 3000
FIXED_EVAL_EPISODES = 5
FIXED_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_OUTPUT_DIR = Path("results")


@dataclass(frozen=True)
class ExperimentConfig:
	env_name: str = FIXED_ENV_NAME
	train_episodes: int = FIXED_TRAIN_EPISODES
	eval_episodes: int = FIXED_EVAL_EPISODES
	seeds: Sequence[int] = FIXED_SEEDS
	algorithms: Sequence[str] = FIXED_ALGORITHMS
	output_dir: str = str(DEFAULT_OUTPUT_DIR)
	device: str = "cpu"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Switch4-v0 MARL experiment entrypoint")
	parser.add_argument("--algorithm", choices=FIXED_ALGORITHMS, default="iql", help="Training algorithm")
	parser.add_argument("--env-name", default=FIXED_ENV_NAME, help="Environment name")
	parser.add_argument("--train-episodes", type=int, default=FIXED_TRAIN_EPISODES, help="Training episodes")
	parser.add_argument("--eval-episodes", type=int, default=FIXED_EVAL_EPISODES, help="Evaluation episodes")
	parser.add_argument("--seeds", type=int, nargs="*", default=list(FIXED_SEEDS), help="Random seed list")
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
	parser.add_argument("--device", default="cpu", help="Torch device")
	return parser.parse_args()


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _unpack_reset_result(reset_result: Any) -> List[np.ndarray]:
	observations = reset_result
	return [np.asarray(obs, dtype=np.float32).reshape(-1) for obs in observations]


def _unpack_step_result(step_result: Any) -> tuple[list[np.ndarray], list[float], list[bool], dict]:
	next_observations, rewards, dones, info = step_result
	return (
		[np.asarray(obs, dtype=np.float32).reshape(-1) for obs in next_observations],
		[float(reward) for reward in rewards],
		[bool(done) for done in dones],
		dict(info),
	)


def _collect_update_stats(update_result: Any) -> Dict[str, float]:
	losses: List[float] = []
	epsilons: List[float] = []
	if isinstance(update_result, list):
		for item in update_result:
			if isinstance(item, dict):
				if "loss" in item:
					losses.append(float(item["loss"]))
				if "epsilon" in item:
					epsilons.append(float(item["epsilon"]))
	return {
		"loss": float(np.mean(losses)) if losses else 0.0,
		"epsilon": float(np.mean(epsilons)) if epsilons else 0.0,
	}


def make_env(env_name: str):
	return gym.make(f"ma_gym:{env_name}")


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


def build_artifact_dirs(output_root: Path, algorithm: str, seed: int) -> Dict[str, Path]:
	"""Create the standard artifact directories for one run.

	The layout is:
	- results/figures/<algorithm>/seed_<seed>/
	- results/logs/<algorithm>/seed_<seed>/
	- results/models/<algorithm>/seed_<seed>/
	"""

	figures_dir = ensure_directory(output_root / "figures" / algorithm / f"seed_{seed}")
	logs_dir = ensure_directory(output_root / "logs" / algorithm / f"seed_{seed}")
	models_dir = ensure_directory(output_root / "models" / algorithm / f"seed_{seed}")
	return {"figures": figures_dir, "logs": logs_dir, "models": models_dir}


def evaluate_policy(env_name: str, trainer: object, eval_episodes: int, seed: int) -> Dict[str, float]:
	set_seed(seed)
	env = make_env(env_name)
	total_returns: List[float] = []
	total_lengths: List[int] = []
	total_success: List[float] = []

	for _ in trange(eval_episodes, desc=f"eval_seed={seed}", unit="ep"):
		observations = _unpack_reset_result(env.reset())
		done_n = [False] * env.n_agents
		has_negative_reward = True
		episode_return = 0.0
		episode_length = 0

		while not all(done_n):
			actions = trainer.act(observations, greedy=True)
			step_result = env.step(actions)
			next_observations, rewards, done_n, _ = _unpack_step_result(step_result)
			if not any(reward < 0.0 for reward in rewards):
				has_negative_reward = False
			episode_return += float(np.sum(rewards))
			episode_length += 1
			observations = next_observations

		total_returns.append(episode_return)
		print(f"Eval Episode {_}: Return={episode_return:.2f}")
		total_lengths.append(episode_length)
		total_success.append(float(all(done_n) and not has_negative_reward))

	env.close()
	return {
		"eval_return_mean": float(np.mean(total_returns)) if total_returns else 0.0,
		"eval_return_std": float(np.std(total_returns)) if total_returns else 0.0,
		"eval_length_mean": float(np.mean(total_lengths)) if total_lengths else 0.0,
		"eval_success_rate": float(np.mean(total_success)) if total_success else 0.0,
	}


def run_training_loop(algorithm: str, env_name: str, train_episodes: int, eval_episodes: int, seeds: Sequence[int], output_dir: str, device: str) -> None:
	output_root = ensure_directory(output_dir)

	experiment_config = ExperimentConfig(
		env_name=env_name,
		train_episodes=train_episodes,
		eval_episodes=eval_episodes,
		seeds=tuple(seeds),
		algorithms=FIXED_ALGORITHMS,
		output_dir=output_dir,
		device=device,
	)
	save_json(output_root / "experiment_config.json", asdict(experiment_config))

	for seed in seeds:
		set_seed(seed)
		env = make_env(env_name)
		trainer = build_trainer(algorithm, env, device)
		artifact_dirs = build_artifact_dirs(output_root, algorithm, seed)
		metrics_path = artifact_dirs["logs"] / "metrics.jsonl"
		# Clear previous run artifacts so each run starts fresh (overwrite semantics)
		# Remove old metrics file
		try:
			if metrics_path.exists():
				metrics_path.unlink()
		except Exception:
			pass
		# Clear models directory contents
		checkpoints_dir = artifact_dirs["models"]
		try:
			for p in list(checkpoints_dir.iterdir()):
				if p.is_file() or p.is_symlink():
					p.unlink()
				elif p.is_dir():
					shutil.rmtree(p)
		except Exception:
			pass
		# Clear figures directory contents
		figures_dir = artifact_dirs["figures"]
		try:
			for p in list(figures_dir.iterdir()):
				if p.is_file() or p.is_symlink():
					p.unlink()
				elif p.is_dir():
					shutil.rmtree(p)
		except Exception:
			pass
		# Ensure directories exist after clearing
		ensure_directory(checkpoints_dir)
		ensure_directory(figures_dir)
		checkpoints_dir = artifact_dirs["models"]
		best_eval_return = float("-inf")
		train_returns: List[float] = []
		eval_returns: List[float] = []
		eval_success_rates: List[float] = []
		eval_interval = max(1, train_episodes // 10)

		episode_iter = trange(1, train_episodes + 1, desc=f"seed={seed}", unit="ep")
		for episode_index in episode_iter:
			observations = _unpack_reset_result(env.reset())
			done_n = [False] * env.n_agents
			has_negative_reward = True
			episode_return = 0.0
			episode_length = 0

			while not all(done_n):
				actions = trainer.act(observations, greedy=False)
				step_result = env.step(actions)
				next_observations, rewards, done_n, _ = _unpack_step_result(step_result)
				if not any(reward < 0.0 for reward in rewards):
					has_negative_reward = False
				trainer.observe(observations, actions, rewards, next_observations, done_n)
				update_stats = _collect_update_stats(trainer.update())

				episode_return += float(np.sum(rewards))
				episode_length += 1
				observations = next_observations

			episode_success = float(all(done_n) and not has_negative_reward)

			train_returns.append(episode_return)
			train_record = {
				"phase": "train",
				"seed": seed,
				"episode": episode_index,
				"train_return": episode_return,
				"episode_length": episode_length,
				"success": episode_success,
				"loss": update_stats["loss"],
				"epsilon": update_stats["epsilon"],
			}
			append_jsonl(metrics_path, train_record)
			# update tqdm postfix with latest metrics
			try:
				episode_iter.set_postfix(
					{
						"ret": f"{episode_return:.2f}",
						"loss": f"{update_stats['loss']:.4f}",
						"eps": f"{update_stats['epsilon']:.3f}",
					}
				)
			except Exception:
				pass

			if episode_index % eval_interval == 0 or episode_index == train_episodes:
				eval_stats = evaluate_policy(env_name, trainer, eval_episodes, seed)
				eval_returns.append(eval_stats["eval_return_mean"])
				eval_success_rates.append(eval_stats["eval_success_rate"])
				eval_record = {
					"phase": "eval",
					"seed": seed,
					"episode": episode_index,
					**eval_stats,
				}
				append_jsonl(metrics_path, eval_record)
				if eval_stats["eval_return_mean"] >= best_eval_return:
					best_eval_return = eval_stats["eval_return_mean"]
					save_checkpoint(checkpoints_dir / "best.pt", trainer.state_dict())

		save_checkpoint(checkpoints_dir / "final.pt", trainer.state_dict())
		save_json(
			artifact_dirs["logs"] / "summary.json",
			{
				"seed": seed,
				"algorithm": algorithm,
				"env_name": env_name,
				"train_episodes": train_episodes,
				"eval_episodes": eval_episodes,
				"best_eval_return": best_eval_return,
				"final_train_return": train_returns[-1] if train_returns else 0.0,
				"final_eval_return": eval_returns[-1] if eval_returns else 0.0,
				"final_eval_success_rate": eval_success_rates[-1] if eval_success_rates else 0.0,
			},
		)
		plot_learning_curves(
			{"train_return": train_returns},
			artifact_dirs["figures"] / "train_return.png",
			title="Training Return",
			ylabel="Return",
		)
		plot_learning_curves(
			{"eval_return": eval_returns},
			artifact_dirs["figures"] / "eval_return.png",
			title="Evaluation Return",
			ylabel="Return",
		)
		plot_learning_curves(
			{"eval_success_rate": eval_success_rates},
			artifact_dirs["figures"] / "eval_success_rate.png",
			title="Evaluation Success Rate",
			ylabel="Success Rate",
		)
		env.close()


def main() -> None:
	args = parse_args()
	run_training_loop(
		algorithm=args.algorithm,
		env_name=args.env_name,
		train_episodes=args.train_episodes,
		eval_episodes=args.eval_episodes,
		seeds=args.seeds,
		output_dir=args.output_dir,
		device=args.device,
	)


if __name__ == "__main__":
	main()
