"""Closed-loop episodes in a LeRobot simulator (PushT or ALOHA), shared by rollout.py and training."""

import random
from contextlib import contextmanager
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from accelerate.utils import set_seed
from hydra.utils import instantiate

from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.utils.constants import ACTION


def check_compatibility(policy_config, batch, env):
    """Fail early if the policy was trained on other observation or action features than the env provides."""
    features = {**policy_config.input_features, **policy_config.output_features}
    expected = {key: tuple(feature.shape) for key, feature in features.items()}
    actual = {key: tuple(value.shape[1:]) for key, value in batch.items() if torch.is_tensor(value)}
    actual[ACTION] = tuple(env.single_action_space.shape)
    mismatched = {key: (shape, actual.get(key)) for key, shape in expected.items() if actual.get(key) != shape}
    if mismatched:
        raise ValueError(f"Policy and env features differ as {{key: (policy, env)}}: {mismatched}")


def make_rollout_env(env_config, policy_config):
    """Build a single-env LeRobot simulator and its processors from an env config node."""
    env_cfg = instantiate(env_config)
    env = make_env(env_cfg, n_envs=1, use_async_envs=False)[env_cfg.type][0]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_config)
    return env_cfg, env, env_preprocessor, env_postprocessor


@contextmanager
def preserved_rng():
    """Restore the Python, NumPy and torch generators afterwards, so seeded episodes leave training RNG intact."""
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])


def run_episodes(policy, preprocessor, postprocessor, env, env_preprocessor, env_postprocessor, episodes,
                 seed, max_steps, render=None, video_dir=None, fps=None, overlay=None, log=None):
    """Run the given episode indices (episode i uses seed + i) and return one result per episode.

    Videos episode_<i>.mp4 are written when video_dir is set.
    """
    observation, _ = env.reset(seed=seed)
    check_compatibility(policy.config, env_preprocessor(preprocess_observation(observation)), env)
    results = []
    for episode in episodes:
        seed_i = seed + episode
        # Seed the env and the diffusion noise per episode, so each episode is reproducible on its own.
        set_seed(seed_i)
        observation, _ = env.reset(seed=seed_i)
        policy.reset()
        frame = render() if video_dir is not None else None
        frames = [frame]
        rewards = []
        success = False

        for _ in range(max_steps):
            batch = env_preprocessor(preprocess_observation(observation))
            with torch.inference_mode():
                action = policy.select_action(preprocessor(batch))
            if overlay is not None:
                frames.extend(overlay.denoising_frames(frame))
            action = postprocessor(action)
            action = env_postprocessor({"action": action})["action"]
            observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            rewards.append(float(np.asarray(reward)[0]))
            # The env's own success flag (PushT: goal coverage > 95%; ALOHA: reward 4, the cube touches the
            # left gripper and not the table). The vector env resets at the end of an episode, so the
            # finished episode's info is in final_info and the top-level info belongs to the next one.
            done = bool(np.asarray(terminated)[0] or np.asarray(truncated)[0])
            step_info = info["final_info"] if done else info
            success = success or bool(step_info["is_success"][0])
            if done:
                break
            if video_dir is not None:
                frame = render()
                frames.append(frame if overlay is None else overlay.draw_plan(frame))

        result = {"episode": episode, "seed": seed_i, "success": success, "length": len(rewards),
                  "sum_reward": sum(rewards), "max_reward": max(rewards)}
        if video_dir is not None:
            path = Path(video_dir) / f"episode_{episode:03d}.mp4"
            imageio.mimsave(path, frames, fps=fps)
            result["video"] = path.name
        results.append(result)
        if log is not None:
            log(result, path if video_dir is not None else None)
    return results


def summarize(episodes):
    """Success rate and reward statistics over a list of run_episodes results."""
    n_successes = sum(result["success"] for result in episodes)
    return {
        "n_episodes": len(episodes),
        "n_successes": n_successes,
        "success_rate": n_successes / len(episodes),
        "mean_sum_reward": float(np.mean([result["sum_reward"] for result in episodes])),
        "mean_max_reward": float(np.mean([result["max_reward"] for result in episodes])),
        "mean_length": float(np.mean([result["length"] for result in episodes])),
    }
