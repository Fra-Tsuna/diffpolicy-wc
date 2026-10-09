import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from omegaconf import OmegaConf

from src.trainer.checkpoint_manager import CheckpointManager
from tests.fixtures import TestModel


def runtime(tmp_path, use_ema=False, use_scaler=False):
    accelerator = Accelerator(cpu=True, step_scheduler_with_optimizer=False)
    if use_scaler:
        accelerator.scaler = torch.amp.GradScaler("cpu", init_scale=64)
    model = TestModel(1)
    # Match Hydra's default container behavior without requiring _convert_.
    optimizer = torch.optim.AdamW(model.parameters(), betas=OmegaConf.create([0.9, 0.999]))
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=0, num_training_steps=10)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    ema = EMAModel(model.parameters(), decay=0.5) if use_ema else None
    manager = CheckpointManager(tmp_path, accelerator, patience=3, ema=ema)
    return accelerator, model, optimizer, scheduler, ema, manager


@pytest.mark.parametrize("use_ema", [False, True])
@pytest.mark.parametrize("use_scaler", [False, True])
def test_full_state_restores_training_objects_counters_and_rng(tmp_path, use_ema, use_scaler):
    accelerator, model, optimizer, scheduler, ema, manager = runtime(tmp_path, use_ema, use_scaler)
    for _ in range(2):
        optimizer.zero_grad()
        accelerator.backward(model({"x": torch.ones(2, 1)}).square().mean())
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.step(model.parameters())

    manager.best_val_loss = 0.25
    manager.counter = 3
    manager.early_stop = True
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    expected_ema = copy.deepcopy(ema.state_dict()) if ema is not None else None
    expected_scaler = copy.deepcopy(accelerator.scaler.state_dict()) if use_scaler else None
    path = manager.save_training(7)
    expected_random = (random.random(), np.random.random(), torch.rand(3))

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    optimizer.state.clear()
    scheduler.step()
    manager.iteration = 0
    manager.counter = 0
    manager.early_stop = False
    manager.best_val_loss = float("inf")
    if ema is not None:
        ema.step(model.parameters())
    if use_scaler:
        accelerator.scaler.load_state_dict({**expected_scaler, "scale": 1024})
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    manager.load_training(path)
    assert manager.state_dict() == {
        "iteration": 7, "best_val_loss": 0.25, "counter": 3, "early_stop": True, "num_processes": 1,
    }
    for key, value in expected_model.items():
        assert torch.equal(model.state_dict()[key], value)
    for key, state in expected_optimizer["state"].items():
        for name, value in state.items():
            assert torch.equal(optimizer.state_dict()["state"][key][name], value)
    assert scheduler.state_dict() == expected_scheduler
    if ema is not None:
        assert ema.optimization_step == expected_ema["optimization_step"]
        for actual, expected in zip(ema.shadow_params, expected_ema["shadow_params"]):
            assert torch.equal(actual, expected)
    if use_scaler:
        assert accelerator.scaler.state_dict() == expected_scaler
    assert random.random() == expected_random[0]
    assert np.random.random() == expected_random[1]
    assert torch.equal(torch.rand(3), expected_random[2])


@pytest.mark.parametrize("failure", ["write", "publish"])
def test_failed_training_save_preserves_previous_checkpoint(tmp_path, monkeypatch, failure):
    accelerator, model, optimizer, scheduler, ema, manager = runtime(tmp_path)
    last = manager.save_training(1)
    previous = last.resolve()
    if failure == "write":
        def partial_save(path):
            Path(path).mkdir(exist_ok=True)
            (Path(path) / "partial.bin").write_bytes(b"incomplete")
            raise OSError("interrupted write")

        monkeypatch.setattr(accelerator, "save_state", partial_save)
    else:
        import src.trainer.checkpoint_manager as module
        original_replace = module.os.replace

        def interrupted_replace(source, destination):
            if Path(destination) == last:
                raise OSError("interrupted publication")
            return original_replace(source, destination)

        monkeypatch.setattr(module.os, "replace", interrupted_replace)

    with pytest.raises((OSError, RuntimeError), match="interrupted"):
        manager.save_training(2)
    assert last.resolve() == previous
    manager.load_training(last)
    assert manager.iteration == 1
    assert not list(tmp_path.glob(".state-*"))


@pytest.mark.parametrize("inside_root", [False, True])
def test_cleanup_does_not_delete_unowned_previous_checkpoint(tmp_path, inside_root):
    root = tmp_path / "checkpoints"
    accelerator, model, optimizer, scheduler, ema, manager = runtime(root)
    foreign = (root if inside_root else tmp_path) / ("state-" + "a" * 32)
    foreign.mkdir()
    (foreign / "important.txt").write_text("keep")
    if not inside_root:
        (foreign / ".checkpoint-owner").write_text(manager.owner)
    (root / "last").symlink_to(foreign, target_is_directory=True)
    manager.save_training(1)
    assert (foreign / "important.txt").read_text() == "keep"


def test_stale_staging_is_never_published_or_automatically_pruned(tmp_path):
    accelerator, model, optimizer, scheduler, ema, manager = runtime(tmp_path)
    abandoned = manager._new_path("state")
    (abandoned / "partial.bin").write_bytes(b"partial")
    last = manager.save_training(1)
    assert last.resolve() != abandoned
    assert abandoned.is_dir()
    previous = last.resolve()
    manager.save_training(2)
    assert not previous.exists()
    assert abandoned.is_dir()


def test_best_selection_is_independent_of_full_checkpointing_and_config_is_plain(tmp_path):
    accelerator, model, optimizer, scheduler, ema, manager = runtime(tmp_path)
    config = OmegaConf.create({"model": {"dim": 1}, "optimizer": {"betas": [0.9, 0.999]}, "dim": "${model.dim}"})
    assert manager.save_if_best(0.5, model, config, 1)
    assert not (tmp_path / "last").exists()
    checkpoint = manager.read(tmp_path / "best.pth")
    assert type(checkpoint["config"]) is dict
    assert type(checkpoint["config"]["optimizer"]["betas"]) is list
    assert checkpoint["config"]["dim"] == 1
    assert not manager.save_if_best(0.6, model, config, 2)
    assert manager.counter == 1


def test_ema_restore_keeps_its_device_without_calling_to(tmp_path, monkeypatch):
    accelerator, model, optimizer, scheduler, ema, manager = runtime(tmp_path, use_ema=True)
    expected_devices = [parameter.device for parameter in ema.shadow_params]
    last = manager.save_training(1)

    def forbidden_to(*args, **kwargs):
        raise AssertionError("EMA.to should not be called during restore")

    monkeypatch.setattr(ema, "to", forbidden_to)
    manager.load_training(last)
    assert [parameter.device for parameter in ema.shadow_params] == expected_devices


def test_resume_restores_counters_even_when_best_model_is_newer(tmp_path):
    accelerator, model, optimizer, scheduler, ema, manager = runtime(tmp_path)
    manager.best_val_loss = 0.5
    manager.counter = 2
    last = manager.save_training(3)
    manager.save_if_best(0.25, model, {}, 4)
    manager.load_training(last)
    assert manager.iteration == 3
    assert manager.best_val_loss == 0.5
    assert manager.counter == 2
    assert not manager.early_stop


def test_automatic_checkpoint_naming_is_rejected(tmp_path):
    from accelerate.utils import ProjectConfiguration
    accelerator = Accelerator(cpu=True, project_config=ProjectConfiguration(automatic_checkpoint_naming=True))
    with pytest.raises(ValueError, match="automatic_checkpoint_naming"):
        CheckpointManager(tmp_path, accelerator)


def test_lightweight_best_exports_ema_without_changing_live_weights(tmp_path):
    accelerator, model, optimizer, scheduler, ema, manager = runtime(tmp_path, use_ema=True)
    expected_ema = [parameter.clone() for parameter in ema.shadow_params]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    expected_model = copy.deepcopy(model.state_dict())
    manager.best_val_loss = 0.5
    best = manager.save_model(model, {"model": {}}, 2)
    checkpoint = manager.read(best)
    assert set(checkpoint) == {
        "iteration", "model_state_dict", "ema_model_state_dict", "config", "best_val_loss",
    }
    for key, value in expected_model.items():
        assert torch.equal(model.state_dict()[key], value)
        assert torch.equal(checkpoint["model_state_dict"][key], value)
    manager.restore_model(checkpoint, model, apply_ema=True)
    for actual, expected in zip(model.parameters(), expected_ema):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("failure", ["write", "publish"])
def test_failed_best_save_preserves_previous_model(tmp_path, monkeypatch, failure):
    accelerator, model, optimizer, scheduler, ema, manager = runtime(tmp_path)
    best = manager.save_model(model, {}, 1)
    previous = best.read_bytes()
    if failure == "write":
        def partial_save(obj, path):
            Path(path).write_bytes(b"incomplete")
            raise OSError("interrupted write")

        monkeypatch.setattr(accelerator, "save", partial_save)
    else:
        import src.trainer.checkpoint_manager as module
        original_replace = module.os.replace

        def interrupted_replace(source, destination):
            if Path(destination) == best:
                raise OSError("interrupted publication")
            return original_replace(source, destination)

        monkeypatch.setattr(module.os, "replace", interrupted_replace)

    with pytest.raises((OSError, RuntimeError), match="interrupted"):
        manager.save_model(model, {}, 2)
    assert best.read_bytes() == previous
    assert manager.read(best)["iteration"] == 1
