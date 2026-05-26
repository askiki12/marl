"""Render a trained Switch4-v0 policy from a random saved seed.

The script picks one available checkpoint under ``results/models/<algorithm>/seed_<seed>``
and runs a greedy rollout while calling ``env.render``.

By default it tries to open the gym human viewer when a display is available.
It also saves a GIF automatically under ``results/gif/<algorithm>/`` unless an
explicit ``--save-gif`` path is provided.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image

try:
	import gym
except ImportError as exc:  # pragma: no cover
	raise ImportError("Gym is required to render the environment") from exc

try:
	import torch
except ImportError as exc:  # pragma: no cover
	raise ImportError("PyTorch is required to load checkpoints") from exc

from algorithms import IQLConfig, IQLTrainer, VDNConfig, VDNTrainer
from utils import ensure_directory


FIXED_ENV_NAME = "Switch4-v0"
FIXED_ALGORITHMS = ("iql", "vdn")
DEFAULT_CHECKPOINT_NAME = "best.pt"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Render a random Switch4-v0 checkpoint")
	parser.add_argument("--algorithm", choices=FIXED_ALGORITHMS, default="iql", help="Algorithm to render")
	parser.add_argument("--env-name", default=FIXED_ENV_NAME, help="Environment name")
	parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME, help="Checkpoint filename to load")
	parser.add_argument("--results-root", default=None, help="Results root directory containing models/")
	parser.add_argument("--seed", type=int, default=None, help="Specific seed to render; random seed is chosen when omitted")
	parser.add_argument("--render-mode", choices=("auto", "human", "rgb_array"), default="auto", help="Rendering mode")
	parser.add_argument("--save-gif", default=None, help="Optional output GIF path")
	parser.add_argument("--fps", type=int, default=8, help="GIF frame rate when saving")
	parser.add_argument("--step-delay", type=float, default=0.5, help="Seconds to wait between environment steps")
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
		obs_dim = int(np.asarray(env.reset()).shape[-1])

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


def _available_seeds(results_root: Path, algorithm: str, checkpoint_name: str) -> List[int]:
	model_root = results_root / "models" / algorithm
	if not model_root.exists():
		return []
	seed_values: List[int] = []
	for seed_dir in model_root.glob("seed_*"):
		if not seed_dir.is_dir():
			continue
		seed_text = seed_dir.name.replace("seed_", "", 1)
		if not seed_text.isdigit():
			continue
		checkpoint_path = seed_dir / checkpoint_name
		if checkpoint_path.exists():
			seed_values.append(int(seed_text))
	return sorted(seed_values)


def _load_checkpoint(trainer: object, checkpoint_path: Path, device: str) -> None:
	state = torch.load(checkpoint_path, map_location=device)
	trainer.load_state_dict(state)


def _select_seed(results_root: Path, algorithm: str, checkpoint_name: str, requested_seed: int | None) -> int:
	seed_values = _available_seeds(results_root, algorithm, checkpoint_name)
	if not seed_values:
		raise FileNotFoundError(f"No checkpoints found under {results_root / 'models' / algorithm}")
	if requested_seed is not None:
		if requested_seed not in seed_values:
			raise FileNotFoundError(f"Checkpoint not found for seed {requested_seed} and algorithm {algorithm}")
		return requested_seed
	return random.choice(seed_values)


def _maybe_init_gif_writer(save_gif: str | None):
	if save_gif is None:
		return None, None
	output_path = Path(save_gif)
	ensure_directory(output_path.parent)
	return output_path, []


def _default_gif_path(results_root: Path, algorithm: str, env_name: str, seed: int) -> Path:
	gif_dir = ensure_directory(results_root / "gif" / algorithm)
	return gif_dir / f"{env_name.lower()}_seed_{seed}.gif"


def _capture_frame(env, render_mode: str) -> np.ndarray | None:
	frame = env.render(mode=render_mode)
	if frame is None:
		return None
	return np.asarray(frame)


def _write_gif(frames: Sequence[np.ndarray], output_path: Path, fps: int | None = None, step_delay: float | None = None) -> None:
	if not frames:
		return
	images = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
	if step_delay is not None:
		frame_duration_ms = 100
		repeat_count = max(1, int(round((step_delay * 1000) / frame_duration_ms)))
		images = [image for image in images for _ in range(repeat_count)]
	else:
		frame_duration_ms = max(1, int(1000 / max(1, fps or 8)))
	images[0].save(
		output_path,
		save_all=True,
		append_images=images[1:],
		duration=frame_duration_ms,
		loop=0,
	)


def render_episode(algorithm: str, env_name: str, checkpoint_name: str, results_root: str | None, seed: int | None, render_mode: str, save_gif: str | None, fps: int, step_delay: float, device: str) -> Dict[str, Any]:
	root = resolve_results_root(results_root)
	selected_seed = _select_seed(root, algorithm, checkpoint_name, seed)
	checkpoint_path = root / "models" / algorithm / f"seed_{selected_seed}" / checkpoint_name
	set_seed(selected_seed)
	gif_output_path = Path(save_gif) if save_gif is not None else _default_gif_path(root, algorithm, env_name, selected_seed)

	env = make_env(env_name)
	trainer = build_trainer(algorithm, env, device)
	_load_checkpoint(trainer, checkpoint_path, device)

	frames: List[np.ndarray] = []
	should_show_window = render_mode in ("auto", "human") and os.environ.get("DISPLAY")
	should_capture_frames = True

	observations = _unpack_reset_result(env.reset())
	done_n = [False] * env.n_agents
	episode_return = 0.0
	episode_length = 0

	if should_show_window:
		try:
			env.render(mode="human")
		except Exception:
			should_show_window = False

	while not all(done_n):
		actions = trainer.act(observations, greedy=True)
		step_result = env.step(actions)
		next_observations, rewards, done_n, _ = _unpack_step_result(step_result)
		print(f"step={episode_length + 1} actions={actions} rewards={rewards}")
		episode_return += float(np.sum(rewards))
		episode_length += 1
		observations = next_observations

		if should_capture_frames:
			frame = _capture_frame(env, "rgb_array")
			if frame is not None:
				frames.append(frame)
		if should_show_window:
			try:
				env.render(mode="human")
			except Exception:
				should_show_window = False
		time.sleep(max(0.0, step_delay))

	env.close()

	_write_gif(frames, gif_output_path, fps=fps, step_delay=step_delay)

	return {
		"algorithm": algorithm,
		"env_name": env_name,
		"seed": selected_seed,
		"checkpoint_path": str(checkpoint_path),
		"episode_return": float(episode_return),
		"episode_length": int(episode_length),
		"render_mode": render_mode,
		"gif_path": str(gif_output_path),
	}


def main() -> None:
	args = parse_args()
	result = render_episode(
		algorithm=args.algorithm,
		env_name=args.env_name,
		checkpoint_name=args.checkpoint_name,
		results_root=args.results_root,
		seed=args.seed,
		render_mode=args.render_mode,
		save_gif=args.save_gif,
		fps=args.fps,
		step_delay=args.step_delay,
		device=args.device,
	)
	print(result)


if __name__ == "__main__":
	main()
