"""Generate genuine closed-loop videos and success metrics in a LeRobot simulator (PushT or ALOHA)."""

import json
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from lerobot.envs import PushtEnv
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy

from src.inference.rollout import make_rollout_env, run_episodes, summarize
from src.trainer.checkpoint_manager import CheckpointManager
from src.utils.diffusion_overlay import DiffusionOverlay
from src.utils.viz import make_video_renderer


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

    env_cfg, env, env_preprocessor, env_postprocessor = make_rollout_env(cfg.env, policy.config)
    render = make_video_renderer(env, cfg.video_camera)
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

    def log(result, path):
        print(f"Episode {result['episode']}: success={result['success']}, reward={result['sum_reward']:.3f}, "
              f"length={result['length']}, video={path}")

    try:
        episodes = run_episodes(
            policy, preprocessor, postprocessor, env, env_preprocessor, env_postprocessor,
            episodes=range(cfg.n_episodes), seed=cfg.seed, max_steps=cfg.max_steps, render=render,
            video_dir=video_dir, fps=env_cfg.fps, overlay=overlay, log=log,
        )
    finally:
        env.close()

    metrics = {
        "env": env_cfg.gym_id,
        "policy": cfg.checkpoint_path or cfg.pretrained_path,
        "apply_ema": cfg.apply_ema,
        "video_camera": cfg.video_camera,
        **summarize(episodes),
        "episodes": episodes,
    }
    (video_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Success rate: {metrics['n_successes']}/{len(episodes)} = {metrics['success_rate']:.3f}, "
          f"metrics={video_dir / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    main()
