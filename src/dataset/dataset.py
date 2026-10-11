"""Episode-disjoint LeRobot demonstrations (PushT, ALOHA) with Diffusion Policy temporal windows."""

import numpy as np
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        repo_id: str,
        n_obs_steps: int,
        horizon: int,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.repo_id = repo_id
        metadata = LeRobotDatasetMetadata(repo_id)

        # Split by entire demonstration episodes, never by adjacent frames.
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        episodes = np.arange(metadata.total_episodes)
        np.random.default_rng(seed).shuffle(episodes)
        n_val = max(1, round(len(episodes) * val_fraction))
        n_test = max(1, round(len(episodes) * test_fraction))
        if n_val + n_test >= len(episodes):
            raise ValueError("Need at least three episodes for a train/val/test split.")
        selected = {
            "val": episodes[:n_val],
            "test": episodes[n_val:n_val + n_test],
            "train": episodes[n_val + n_test:],
        }[split]

        # LeRobot handles frames near episode boundaries and video decoding.
        obs_offsets = [i / metadata.fps for i in range(1 - n_obs_steps, 1)]
        action_offsets = [
            i / metadata.fps for i in range(1 - n_obs_steps, 1 - n_obs_steps + horizon)
        ]
        delta_timestamps = {"observation.state": obs_offsets, "action": action_offsets}
        for name, feature in metadata.features.items():
            if name.startswith("observation.images") or name.startswith(
                "observation.image"
            ):
                delta_timestamps[name] = obs_offsets

        self.dataset = LeRobotDataset(
            repo_id, episodes=selected.tolist(), delta_timestamps=delta_timestamps
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def model_kwargs(self) -> dict:
        # Skeleton stores this argument in the model checkpoint configuration.
        return {"repo_id": self.repo_id}
