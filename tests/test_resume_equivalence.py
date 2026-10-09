from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from main import main as training_main
from tests.fixtures import TestTrainer
from tests.test_template import configuration


def assert_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("interrupt_step", [3, 4, 6])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_resumed_training_matches_uninterrupted(tmp_path, monkeypatch, capfd, interrupt_step, num_workers):
    cfg = configuration(tmp_path)
    cfg.model._target_ = "tests.fixtures.StochasticTestModel"
    cfg.train_dataloader.dataset._target_ = "tests.fixtures.StatefulTestDataset"
    cfg.train_dataloader.num_workers = num_workers
    if num_workers:
        OmegaConf.update(cfg.train_dataloader, "multiprocessing_context", "spawn", force_add=True)
    cfg.trainer.iterations = 8
    cfg.trainer.validate_every = 3
    cfg.trainer.checkpoint_every = interrupt_step
    cfg.trainer.use_ema = True
    cfg.lr_scheduler.num_training_steps = 8
    cfg.optimizer.lr = 0.01
    seen = []
    original_loss = TestTrainer.compute_loss

    def record_loss(self, batch):
        if self.model.training:
            seen.append(batch["x"].detach().cpu().clone())
        return original_loss(self, batch)

    monkeypatch.setattr(TestTrainer, "compute_loss", record_loss)
    cfg.result_dir = str(tmp_path / "uninterrupted")
    try:
        training_main.__wrapped__(cfg)
    except RuntimeError as exc:
        if num_workers and "exited unexpectedly" in str(exc):
            captured = capfd.readouterr()
            if "unable to resize file" in captured.err and "Invalid argument (22)" in captured.err:
                pytest.skip("This runtime blocks shared tensor storage for spawned DataLoader workers.")
        raise
    expected_batches = list(seen)
    baseline = Path(cfg.result_dir) / "checkpoints/last"

    seen.clear()
    cfg.result_dir = str(tmp_path / "interrupted")

    def interrupt_loss(self, batch):
        if self.model.training and len(seen) == interrupt_step:
            raise RuntimeError("planned interruption")
        return record_loss(self, batch)

    monkeypatch.setattr(TestTrainer, "compute_loss", interrupt_loss)
    with pytest.raises(RuntimeError, match="planned interruption"):
        training_main.__wrapped__(cfg)
    first_batches = list(seen)
    assert len(first_batches) == interrupt_step
    cfg.resume_from = str(Path(cfg.result_dir) / "checkpoints/last")
    cfg.result_dir = str(tmp_path / "resumed")
    seen.clear()
    monkeypatch.setattr(TestTrainer, "compute_loss", record_loss)
    training_main.__wrapped__(cfg)
    for actual, expected in zip(first_batches + seen, expected_batches):
        assert torch.equal(actual, expected)
    assert len(first_batches + seen) == len(expected_batches) == 8
    resumed = Path(cfg.result_dir) / "checkpoints/last"
    for name in ["optimizer.bin", "scheduler.bin", "custom_checkpoint_0.pkl", "custom_checkpoint_1.pkl"]:
        assert_equal(torch.load(resumed / name, weights_only=False), torch.load(baseline / name, weights_only=False))
    from safetensors.torch import load_file
    assert_equal(load_file(str(resumed / "model.safetensors")), load_file(str(baseline / "model.safetensors")))
