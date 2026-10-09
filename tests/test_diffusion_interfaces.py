"""Small interface contracts requiring no downloaded robot demonstrations.

The integration that needs LeRobot / PushT runs in separate experiments.
"""

import torch
from src.trainer.trainer import TrainingPipeline
from src.inference.inference import InferencePipeline


def test_training_pipeline_has_eval_and_preserves_validate():
    assert callable(TrainingPipeline.train)
    assert callable(TrainingPipeline.eval)
    assert callable(TrainingPipeline.validate)
    assert callable(TrainingPipeline.resume)


def test_training_loss_delegates_to_model():
    obj = TrainingPipeline.__new__(TrainingPipeline)
    obj.model = lambda batch: (batch["a"] ** 2).mean()
    batch = {"a": torch.tensor([1.0, 3.0], requires_grad=True)}
    loss = obj.compute_loss(batch)["loss"]
    loss.backward()
    assert batch["a"].grad is not None


def test_inference_reports_offline_loss():
    obj = InferencePipeline.__new__(InferencePipeline)
    pred = torch.tensor(0.25)
    metrics = obj.compute_metrics(pred, {})
    assert metrics["loss"].item() == 0.25
