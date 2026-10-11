"""Generate genuine closed-loop PushT videos from LeRobot or skeleton checkpoints."""

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

from src.trainer.checkpoint_manager import CheckpointManager
from src.utils.diffusion_overlay import DiffusionOverlay


@hydra.main(version_base=None, config_path="config", config_name="rollout")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
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

    env_cfg = PushtEnv(render_mode="rgb_array")
    env = make_env(env_cfg, n_envs=1, use_async_envs=False)["pusht"][0]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy.config)
    overlay = None
    if cfg.diffusion_overlay:
        overlay = DiffusionOverlay(
            policy,
            to_env_action=lambda actions: env_postprocessor({"action": postprocessor(actions)})["action"],
            denoising_frames=cfg.denoising_frames,
        )
    video_dir = Path(cfg.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    try:
        for episode in range(cfg.n_episodes):
            observation, _ = env.reset(seed=cfg.seed + episode)
            policy.reset()
            frame = env.call("render")[0]
            frames = [frame]
            reward_sum = 0.0

            for _ in range(cfg.max_steps):
                batch = env_preprocessor(preprocess_observation(observation))
                with torch.inference_mode():
                    action = policy.select_action(preprocessor(batch))
                if overlay is not None:
                    frames.extend(overlay.denoising_frames(frame))
                action = postprocessor(action)
                action = env_postprocessor({"action": action})["action"]
                observation, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                reward_sum += float(np.asarray(reward)[0])
                frame = env.call("render")[0]
                frames.append(frame if overlay is None else overlay.draw_plan(frame))
                if bool(np.asarray(terminated)[0] or np.asarray(truncated)[0]):
                    break

            path = video_dir / f"episode_{episode:03d}.mp4"
            imageio.mimsave(path, frames, fps=env_cfg.fps)
            print(f"Episode {episode}: reward={reward_sum:.3f}, video={path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
