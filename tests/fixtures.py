"""Small CPU fixtures for infrastructure tests; never used by default configs."""

import torch
from torch import nn
from torch.utils.data import Dataset

from src.inference.inference import InferencePipeline
from src.trainer.trainer import TrainingPipeline


class TestDataset(Dataset):
    __test__ = False

    def __init__(self, data_root, split, **kwargs):
        self.x = torch.arange(1, 6, dtype=torch.float32).unsqueeze(1) / 5
        self.y = 2 * self.x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"x": self.x[index], "y": self.y[index], "metadata": "sample"}

    def model_kwargs(self):
        return {"input_dim": 1}


class TestModel(nn.Module):
    __test__ = False

    def __init__(self, input_dim, **kwargs):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, batch):
        return self.linear(batch["x"])

    def trainable_parameters(self):
        return self.parameters()


class TestTrainer(TrainingPipeline):
    __test__ = False

    def compute_loss(self, batch):
        loss = ((self.model(batch) - batch["y"]) ** 2).mean()
        return {"loss": loss}


class TestInference(InferencePipeline):
    __test__ = False

    def compute_metrics(self, predictions, batch):
        return {"mse": ((predictions - batch["y"]) ** 2).mean()}


class StochasticTestModel(TestModel):
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, batch):
        return self.linear(self.dropout(batch["x"]))


class StatefulTestDataset(TestDataset):
    """A stochastic dataset whose transform RNG is independent of the model RNG."""
    def __init__(self, data_root, split, **kwargs):
        super().__init__(data_root, split)
        self.generator = torch.Generator().manual_seed(123)

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        batch["x"] = batch["x"] + torch.rand(1, generator=self.generator) * 0.1
        return batch

    def state_dict(self):
        return {"rng": self.generator.get_state()}

    def load_state_dict(self, state):
        self.generator.set_state(state["rng"])
