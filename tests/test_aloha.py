"""ALOHA Transfer Cube experiment: config selection, policy/env compatibility and rollout metrics.

The model and rollout tests download lerobot/aloha_sim_transfer_cube_human (about 75 MB) on first use.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lerobot.configs import FeatureType, PolicyFeature

from rollout import check_compatibility
from rollout import main as rollout_main

ROOT = Path(__file__).resolve().parents[1]
ALOHA = "lerobot/aloha_sim_transfer_cube_human"


def compose_config(config_name, overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        return compose(config_name=config_name, overrides=list(overrides))


def test_experiment_selects_aloha_everywhere_and_keeps_pusht_default():
    pusht = compose_config("rollout")
    assert pusht.train_dataloader.dataset.repo_id == "lerobot/pusht"
    assert pusht.model.resize_shape is None and pusht.model.crop_shape is None
    assert pusht.env._target_ == "lerobot.envs.PushtEnv"
    assert pusht.max_steps == 300

    aloha = compose_config("eval", ["experiment=aloha_transfer_cube"])
    for split in ("train", "val", "test"):
        assert aloha[f"{split}_dataloader"].dataset.repo_id == ALOHA
    assert list(aloha.model.resize_shape) == [96, 128]
    assert aloha.model.crop_ratio == 1.0
    assert aloha.env.task == "AlohaTransferCube-v0"
    aloha_rollout = compose_config("rollout", ["experiment=aloha_transfer_cube"])
    assert aloha_rollout.max_steps == 400
    # There is no pretrained ALOHA policy; never fall back to the PushT one.
    assert aloha_rollout.pretrained_path is None


def test_compatibility_check_rejects_mismatched_dimensions():
    config = SimpleNamespace(
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
    )
    aloha_env = SimpleNamespace(single_action_space=Box(-1, 1, shape=(14,)))
    check_compatibility(config, {"observation.state": torch.zeros(1, 14), "task": None}, aloha_env)
    pusht_env = SimpleNamespace(single_action_space=Box(0, 512, shape=(2,)))
    with pytest.raises(ValueError, match="observation.state"):
        check_compatibility(config, {"observation.state": torch.zeros(1, 2)}, pusht_env)


def test_aloha_model_sees_the_whole_resized_frame():
    from src.dataset.dataset import Dataset
    from src.models.model import DiffusionPolicyWrapper

    dataset = Dataset("", "val", ALOHA, n_obs_steps=2, horizon=16)
    batch = torch.utils.data.default_collate([dataset[0], dataset[200]])
    assert batch["observation.images.top"].shape == (2, 2, 3, 480, 640)
    assert batch["observation.state"].shape == (2, 2, 14)
    assert batch["action"].shape == (2, 16, 14)

    model = DiffusionPolicyWrapper(ALOHA, resize_shape=[96, 128])
    shapes = []
    encoder = model.policy.diffusion.rgb_encoder
    encoder = encoder[0] if isinstance(encoder, torch.nn.ModuleList) else encoder
    encoder.backbone.register_forward_pre_hook(lambda module, args: shapes.append(tuple(args[0].shape)))
    loss = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert shapes == [(4, 3, 96, 128)]


def test_backbone_normalization_follows_use_group_norm():
    from src.models.model import DiffusionPolicyWrapper

    def norm_layers(model):
        encoder = model.policy.diffusion.rgb_encoder
        encoder = encoder[0] if isinstance(encoder, torch.nn.ModuleList) else encoder
        return {type(module) for module in encoder.backbone.modules()
                if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.GroupNorm))}

    assert norm_layers(DiffusionPolicyWrapper(ALOHA, resize_shape=[96, 128])) == {torch.nn.BatchNorm2d}
    assert norm_layers(DiffusionPolicyWrapper(ALOHA, resize_shape=[96, 128], use_group_norm=True)) == {torch.nn.GroupNorm}

    cfg = compose_config("train", ["experiment=aloha_transfer_cube", "model.use_group_norm=true"])
    assert cfg.model.use_group_norm is True and cfg.model.pretrained_backbone_weights is None


@pytest.mark.parametrize("forced_success", [False, True])
def test_aloha_rollout_reports_env_success_and_records_video(tmp_path, monkeypatch, forced_success):
    pytest.importorskip("gym_aloha")
    from gym_aloha.tasks.sim import TransferCubeTask
    from src.models.model import DiffusionPolicyWrapper

    model_config = {"_target_": "src.models.model.DiffusionPolicyWrapper", "repo_id": ALOHA, "resize_shape": [96, 128]}
    model = DiffusionPolicyWrapper(ALOHA, resize_shape=[96, 128])
    checkpoint_path = tmp_path / "best.pth"
    torch.save({"model_state_dict": model.state_dict(), "ema_model_state_dict": None,
                "config": {"model": model_config}}, checkpoint_path)
    if forced_success:
        # Reward 4 is gym-aloha's success condition: the left gripper holds the cube above the table.
        monkeypatch.setattr(TransferCubeTask, "get_reward", lambda self, physics: 4)

    cfg = compose_config("rollout", ["experiment=aloha_transfer_cube"])
    cfg.original_work_dir = str(tmp_path)
    cfg.video_dir = str(tmp_path / "videos")
    cfg.checkpoint_path = str(checkpoint_path)
    cfg.accelerator.cpu = True
    cfg.n_episodes = 2
    cfg.max_steps = 3
    metrics = rollout_main.__wrapped__(cfg)

    assert metrics["env"] == "gym_aloha/AlohaTransferCube-v0"
    assert metrics["video_camera"] == "angle"
    assert metrics["n_episodes"] == 2
    assert metrics["n_successes"] == (2 if forced_success else 0)
    assert metrics["success_rate"] == (1.0 if forced_success else 0.0)
    # A success terminates the episode after its first step.
    assert metrics["mean_length"] == (1 if forced_success else 3)
    assert [episode["seed"] for episode in metrics["episodes"]] == [cfg.seed, cfg.seed + 1]
    assert json.loads((tmp_path / "videos/metrics.json").read_text()) == metrics
    frames = imageio.mimread(tmp_path / "videos/episode_000.mp4")
    assert len(frames) == (1 if forced_success else 4)
    assert frames[0].shape == (480, 640, 3)


def test_video_camera_renders_the_selected_scene_camera():
    pytest.importorskip("gym_aloha")
    from lerobot.envs import make_env

    from src.utils.viz import make_video_renderer

    cfg = compose_config("rollout", ["experiment=aloha_transfer_cube"])
    assert cfg.video_camera == "angle"
    assert compose_config("rollout").video_camera is None
    env_cfg = instantiate(cfg.env)
    env = make_env(env_cfg, n_envs=1, use_async_envs=False)[env_cfg.type][0]
    try:
        env.reset(seed=0)
        default = make_video_renderer(env, None)()
        top = make_video_renderer(env, "top")()
        angle = make_video_renderer(env, "angle")()
        # The env's own render is the "top" camera; "angle" is a different view of the same state.
        assert default.shape == angle.shape == (480, 640, 3)
        assert np.array_equal(default, top)
        assert not np.array_equal(default, angle)
        with pytest.raises(ValueError, match="Unknown video_camera"):
            make_video_renderer(env, "nonexistent")
    finally:
        env.close()
