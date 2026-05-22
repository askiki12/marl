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
from typing import Dict, List, Sequence

import numpy as np

try:
	import gym
except ImportError as exc:  # pragma: no cover
	raise ImportError("Gym is required to run the experiment entrypoint") from exc

try:
	import torch
except ImportError as exc:  # pragma: no cover
	raise ImportError("PyTorch is required to run the experiment entrypoint") from exc

from marl.algorithms import IQLConfig, IQLTrainer, VDNConfig, VDNTrainer


FIXED_ENV_NAME = "Switch4-v0"
FIXED_ALGORITHMS = ("iql", "vdn")
FIXED_TRAIN_EPISODES = 3000
FIXED_EVAL_EPISODES = 100
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


def make_env(env_name: str):
	return gym.make(f"ma_gym:{env_name}")


def infer_obs_action_dims(env) -> Dict[str, int]:
	obs_space = env.observation_space
	action_space = env.action_space

	if hasattr(obs_space, "shape") and obs_space.shape is not None:
		obs_dim = int(np.prod(obs_space.shape))
	else:
		sample_obs = env.reset()[0]
		obs_dim = int(np.asarray(sample_obs).shape[-1])

	if hasattr(action_space, "n"):
		action_dim = int(action_space.n)
	else:
		action_dim = int(action_space[0].n)

	return {"obs_dim": obs_dim, "action_dim": action_dim}


def build_trainer(algorithm: str, env) -> object:
	dims = infer_obs_action_dims(env)
	agent_count = int(env.n_agents)
	if algorithm == "iql":
		agent_configs = [IQLConfig(obs_dim=dims["obs_dim"], action_dim=dims["action_dim"])] * agent_count
		return IQLTrainer(agent_configs)
	if algorithm == "vdn":
		agent_configs = [VDNConfig(obs_dim=dims["obs_dim"], action_dim=dims["action_dim"])] * agent_count
		return VDNTrainer(agent_configs)
	raise ValueError(f"Unsupported algorithm: {algorithm}")


def run_training_loop(algorithm: str, env_name: str, train_episodes: int, eval_episodes: int, seeds: Sequence[int], output_dir: str, device: str) -> None:
	output_root = Path(output_dir)
	output_root.mkdir(parents=True, exist_ok=True)

	experiment_config = ExperimentConfig(
		env_name=env_name,
		train_episodes=train_episodes,
		eval_episodes=eval_episodes,
		seeds=tuple(seeds),
		algorithms=FIXED_ALGORITHMS,
		output_dir=output_dir,
		device=device,
	)
	with open(output_root / "experiment_config.json", "w", encoding="utf-8") as config_file:
		json.dump(asdict(experiment_config), config_file, ensure_ascii=False, indent=2)

	for seed in seeds:
		set_seed(seed)
		env = make_env(env_name)
		trainer = build_trainer(algorithm, env)

		# TODO: implement the complete episode loop.
		# Required pieces:
		# - reset env and obtain per-agent observations
		# - select actions with trainer.act(...)
		# - step env and collect transitions
		# - call trainer.observe(...) and trainer.update()
		# - log episode return, success rate, steps, collision metrics
		# - evaluate every fixed interval for eval_episodes
		# - save checkpoints and plots under output_dir / algorithm / seed_xxx
		_ = trainer
		_ = eval_episodes
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
