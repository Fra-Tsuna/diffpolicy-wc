"""Executed by torchrun from test_distributed_checkpoint.py."""
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from src.trainer.checkpoint_manager import CheckpointManager
from src.utils.torch_utils import aggregate_metrics, cycle, prepare_eval_dataloader
from tests.fixtures import StatefulTestDataset, StochasticTestModel, TestDataset, TestTrainer, TestInference
from tests.test_resume_equivalence import assert_equal


def main():
    try:
        accelerator = Accelerator(cpu=True, step_scheduler_with_optimizer=False,
                                  dataloader_config=DataLoaderConfiguration(use_stateful_dataloader=True,
                                                                           use_seedable_sampler=True, data_seed=42))
    except RuntimeError as exc:
        if "Operation not permitted" in str(exc):
            print("DDP_UNAVAILABLE: Gloo socket operation not permitted", flush=True)
            return 77
        raise
    assert accelerator.num_processes == 2
    rank = accelerator.process_index
    set_seed(42)
    model = StochasticTestModel(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=0, num_training_steps=6)
    loader = torch.utils.data.DataLoader(StatefulTestDataset("", "train"), batch_size=2, shuffle=True,
                                        generator=torch.Generator().manual_seed(42))
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    ema = EMAModel(accelerator.unwrap_model(model).parameters(), decay=0.5)
    root = Path(sys.argv[1])
    manager = CheckpointManager(root, accelerator, patience=2, ema=ema, dataloader=loader)

    # Unequal local counts must produce the same sample-weighted global metric.
    metrics = aggregate_metrics({"loss": float((rank + 1) ** 2)}, rank + 1)
    assert metrics["loss"] == 5 / 3
    manager.counter = rank * 4
    assert manager.save_if_best(metrics["loss"], model, {"model": {}}, 0)
    assert manager.counter == 0
    assert not (root / "last").exists()
    assert not manager.save_if_best(metrics["loss"], model, {}, 0)
    assert manager.counter == 1 and not manager.early_stop
    assert not manager.save_if_best(metrics["loss"], model, {}, 0)
    assert manager.counter == 2 and manager.early_stop
    try:
        manager.save_if_best(float(rank), model, {}, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Rank-local validation losses must be rejected collectively")

    def train_steps(iterator, count):
        batches = []
        for _ in range(count):
            batch = next(iterator)
            batches.append(batch["x"].clone())
            optimizer.zero_grad()
            accelerator.backward((model(batch) - batch["y"]).square().mean())
            optimizer.step()
            scheduler.step()
            ema.step(accelerator.unwrap_model(model).parameters())
            manager.iteration += 1
        return batches

    iterator = cycle(loader)
    train_steps(iterator, 1)
    last = manager.save_training(1)
    expected_batches = train_steps(iterator, 5)
    expected_model = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    expected_ema = copy.deepcopy(ema.state_dict())
    expected_rng = (random.random(), np.random.random(), torch.rand(3))
    del iterator
    manager.load_training(last)
    assert len(list(last.glob("random_states_*.pkl"))) == 2
    assert len(list(last.glob("dataloader_*.pth"))) == 2
    resumed_batches = train_steps(cycle(loader), 5)
    assert_equal(resumed_batches, expected_batches)
    assert_equal(accelerator.unwrap_model(model).state_dict(), expected_model)
    assert_equal(optimizer.state_dict(), expected_optimizer)
    assert_equal(scheduler.state_dict(), expected_scheduler)
    assert_equal(ema.state_dict(), expected_ema)
    assert manager.iteration == 6
    assert random.random() == expected_rng[0]
    assert np.random.random() == expected_rng[1]
    assert torch.equal(torch.rand(3), expected_rng[2])

    # A single rank's write failure must reach every rank without replacing last.
    previous = last.resolve()
    original_save = accelerator.save_state
    if rank == 1:
        def fail_save(*args, **kwargs):
            raise OSError("simulated rank write failure")
        accelerator.save_state = fail_save
    try:
        manager.save_training(7)
    except RuntimeError as exc:
        assert "simulated rank write failure" in str(exc)
    else:
        raise AssertionError("Expected distributed write failure")
    accelerator.save_state = original_save
    assert last.resolve() == previous
    manager.load_training(last)
    assert manager.iteration == 1

    # Unwrapped validation must synchronize DDP buffers before processing uneven shards.
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.register_buffer("evaluation_scale", torch.tensor(float(rank + 1)))
    val_loader = prepare_eval_dataloader(torch.utils.data.DataLoader(TestDataset("", "val"), batch_size=2), accelerator)
    trainer = TestTrainer(model=model, iterations=1, dataloaders={"train": loader, "val": val_loader},
                          optimizer=optimizer, lr_scheduler=scheduler, accelerator=accelerator,
                          result_dir=root / "validation", run_config={})
    trainer.validate()
    assert unwrapped.evaluation_scale.item() == 1
    unwrapped.evaluation_scale.fill_(rank + 1)
    TestInference(model, val_loader, accelerator).run()
    assert unwrapped.evaluation_scale.item() == 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("DDP_CHECKPOINT_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
