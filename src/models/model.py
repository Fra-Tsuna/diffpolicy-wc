"""LeRobot Diffusion Policy behind the skeleton's ordinary nn.Module interface."""

import torch
from torch import nn

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


class DiffusionPolicyWrapper(nn.Module):
    def __init__(
        self,
        repo_id: str,
        n_obs_steps: int = 2,
        horizon: int = 16,
        n_action_steps: int = 8,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 20,
        down_dims: tuple[int, ...] = (128, 256, 512),
        pretrained_backbone_weights: str | None = None,
        resize_shape: tuple[int, int] | None = None,
        crop_ratio: float = 1.0,
        crop_shape: tuple[int, int] | None = None,
        crop_is_random: bool = True,
    ):
        super().__init__()
        metadata = LeRobotDatasetMetadata(repo_id)
        features = dataset_to_policy_features(metadata.features)
        outputs = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}
        inputs = {k: v for k, v in features.items() if k not in outputs}

        config = DiffusionConfig(
            input_features=inputs,
            output_features=outputs,
            n_obs_steps=n_obs_steps,
            horizon=horizon,
            n_action_steps=n_action_steps,
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            down_dims=tuple(down_dims),
            pretrained_backbone_weights=pretrained_backbone_weights,
            # The policy resizes and crops images itself, so training and rollouts preprocess identically.
            resize_shape=tuple(resize_shape) if resize_shape is not None else None,
            crop_ratio=crop_ratio,
            crop_shape=tuple(crop_shape) if crop_shape is not None else None,
            crop_is_random=crop_is_random,
            device="cpu",
        )
        self.policy = DiffusionPolicy(config)
        self.dataset_stats = metadata.stats
        self.preprocessor = None
        self.postprocessor = None
        self._processor_device = None

    def get_processors(self):
        # Accelerate may move the module after construction. Match that device lazily.
        device = str(next(self.policy.parameters()).device)
        if self._processor_device != device:
            self.policy.config.device = device
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                dataset_stats=self.dataset_stats,
            )
            self._processor_device = device
        return self.preprocessor, self.postprocessor

    def forward(self, batch):
        preprocessor, _ = self.get_processors()
        loss, _ = self.policy(preprocessor(batch))
        return loss

    @torch.no_grad()
    def select_action(self, observation):
        preprocessor, postprocessor = self.get_processors()
        action = self.policy.select_action(preprocessor(observation))
        return postprocessor(action)

    def reset(self):
        self.policy.reset()

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)
