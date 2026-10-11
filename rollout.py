"""Generate genuine closed-loop videos and success metrics in a LeRobot simulator (PushT or ALOHA)."""

import json
from pathlib import Path

import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from accelerate.utils import set_seed
from hydra.utils import instantiate
from omegaconf import DictConfig

from lerobot.envs import PushtEnv, make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION

from src.trainer.checkpoint_manager import CheckpointManager
from src.utils.diffusion_overlay import DiffusionOverlay


def check_compatibility(policy_config, batch, env):
    """Fail early if the policy was trained on other observation or action features than the env provides."""
    features = {**policy_config.input_features, **policy_config.output_features}
    expected = {key: tuple(feature.shape) for key, feature in features.items()}
    actual = {key: tuple(value.shape[1:]) for key, value in batch.items() if torch.is_tensor(value)}
    actual[ACTION] = tuple(env.single_action_space.shape)
    mismatched = {key: (shape, actual.get(key)) for key, shape in expected.items() if actual.get(key) != shape}
    if mismatched:
        raise ValueError(f"Policy and env features differ as {{key: (policy, env)}}: {mismatched}")


@hydra.main(version_base=None, config_path="config", config_name="rollout")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.accelerator.cpu else "cpu")

    if cfg.checkpoint_path is not None:
        checkpoint = CheckpointManager.read(cfg.checkpoint_path)
        model = instantiate(checkpoint["config"]["model"]).to(device)
        CheckpointManager.restore_model(checkpoint, model, apply_ema=cfg.apply_ema)
        policy = model.policy
        preprocessor, postprocessor = model.get_processors()
    else:
        policy = DiffusionPolicy.from_pretrained(cfg.pretrained_path).to(device).eval()
        policy.config.device = str(device)
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=cfg.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

    env_cfg = instantiate(cfg.env)
    env = make_env(env_cfg, n_envs=1, use_async_envs=False)[env_cfg.type][0]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy.config)
    overlay = None
    # The overlay draws 2D PushT positions; ALOHA actions are 14 joint targets.
    if cfg.diffusion_overlay and isinstance(env_cfg, PushtEnv):
        overlay = DiffusionOverlay(
            policy,
            to_env_action=lambda actions: env_postprocessor({"action": postprocessor(actions)})["action"],
            denoising_frames=cfg.denoising_frames,
        )
    video_dir = Path(cfg.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    try:
        observation, _ = env.reset(seed=cfg.seed)
        check_compatibility(policy.config, env_preprocessor(preprocess_observation(observation)), env)
        for episode in range(cfg.n_episodes):
            # Seed the env and the diffusion noise per episode, so each episode is reproducible on its own.
            seed = cfg.seed + episode
            set_seed(seed)
            observation, _ = env.reset(seed=seed)
            policy.reset()
            frame = env.call("render")[0]
            frames = [frame]
            rewards = []
            success = False

            for _ in range(cfg.max_steps):
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
                frame = env.call("render")[0]
                frames.append(frame if overlay is None else overlay.draw_plan(frame))

            path = video_dir / f"episode_{episode:03d}.mp4"
            imageio.mimsave(path, frames, fps=env_cfg.fps)
            episodes.append({
                "episode": episode, "seed": seed, "success": success, "length": len(rewards),
                "sum_reward": sum(rewards), "max_reward": max(rewards), "video": path.name,
            })
            print(f"Episode {episode}: success={success}, reward={sum(rewards):.3f}, "
                  f"length={len(rewards)}, video={path}")
    finally:
        env.close()

    n_successes = sum(result["success"] for result in episodes)
    metrics = {
        "env": env_cfg.gym_id,
        "policy": cfg.checkpoint_path or cfg.pretrained_path,
        "apply_ema": cfg.apply_ema,
        "n_episodes": len(episodes),
        "n_successes": n_successes,
        "success_rate": n_successes / len(episodes),
        "mean_sum_reward": float(np.mean([result["sum_reward"] for result in episodes])),
        "mean_max_reward": float(np.mean([result["max_reward"] for result in episodes])),
        "mean_length": float(np.mean([result["length"] for result in episodes])),
        "episodes": episodes,
    }
    (video_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Success rate: {n_successes}/{len(episodes)} = {metrics['success_rate']:.3f}, "
          f"metrics={video_dir / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    main()
