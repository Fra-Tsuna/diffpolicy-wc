# Diffusion Policy on PushT and ALOHA

LeRobot's Diffusion Policy trained with Accelerate, Hydra and W&B on two experiments:

| Experiment | Dataset | Simulator | Observations | Actions |
| --- | --- | --- | --- | --- |
| PushT (default) | [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht), 206 episodes at 10 Hz | `PushT-v0` | 96x96 image, 2-D agent position | 2-D target position |
| `experiment=aloha_transfer_cube` | [`lerobot/aloha_sim_transfer_cube_human`](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human), 50 episodes at 50 Hz | `AlohaTransferCube-v0` | 480x640 top camera, 14-D joint positions | 14-D joint targets |

## Layout

| Path | Contents |
| --- | --- |
| `main.py` | Training entry point. |
| `eval.py` | Offline evaluation of a checkpoint (denoising loss) on the test or val split. |
| `rollout.py` | Closed-loop rollouts in the PushT or ALOHA simulator, saved as MP4 videos with success metrics. |
| `src/dataset/dataset.py` | LeRobot dataset with observation/action windows and an episode-disjoint train/val/test split. |
| `src/models/model.py` | `DiffusionPolicyWrapper`: LeRobot's `DiffusionPolicy` and its pre/postprocessors as an `nn.Module`. |
| `src/trainer/` | Training loop, validation, EMA, checkpointing and resume. |
| `src/inference/inference.py` | Evaluation loop used by `eval.py`. |
| `config/` | Hydra configs. `train.yaml` is the base; `eval.yaml` and `rollout.yaml` extend it. `experiment/` holds the experiment presets and `env/` the simulators. |
| `scripts/` | Shell wrappers for setup, training, evaluation, rollouts and tests. See [scripts/README.md](scripts/README.md). |

## Setup

Requires conda and an NVIDIA driver >= 525.

```bash
scripts/setup.sh    # creates the `diffpolicy` conda env (Python 3.12) and installs requirements-lerobot.txt + requirements-dev.txt
wandb login         # or pass wandb.mode=offline to every command
```

`requirements.txt` installs torch from the CUDA 12.6 wheels (the default PyPI build needs driver >= 580) and keeps torch, wandb and diffusers within the ranges that lerobot 0.6.1 supports. `requirements-lerobot.txt` installs lerobot with the `pusht` and `aloha` simulators (gym-pusht, gym-aloha 0.1.x with MuJoCo). Datasets are downloaded to the Hugging Face cache on first use.

## Selecting an experiment

Every command takes the same experiment override; PushT is the default:

```bash
scripts/train.sh                                    # PushT
scripts/train_aloha.sh                              # ALOHA Transfer Cube (= scripts/train.sh experiment=aloha_transfer_cube)
```

`experiment=aloha_transfer_cube` ([config/experiment/aloha_transfer_cube.yaml](config/experiment/aloha_transfer_cube.yaml)) sets the dataset of all three splits (`repo_id`), the run name `diffusion_aloha_transfer_cube`, the image preprocessing and the simulator (`env=aloha_transfer_cube`). Pass it to `eval.sh` and `rollout.sh` too, so they use the ALOHA test split and simulator.

The policy reads its feature names and dimensions from the dataset metadata, so no observation key or size is configured by hand. Images are resized and cropped inside the policy (`model.resize_shape`, `model.crop_ratio`, `model.crop_shape`, `model.crop_is_random`), so training, offline evaluation and rollouts preprocess identically. The values are saved with the checkpoint. PushT keeps its native 96x96 images. ALOHA resizes the whole 480x640 frame to 96x128 (aspect ratio kept) without cropping, because at the start of an episode the grippers sit at the left and right image borders. With `resize_shape` set, LeRobot 0.6.1 derives the crop from `crop_ratio` (1.0 = no crop) and ignores `crop_shape`.

The ResNet18 vision backbone starts from random weights with BatchNorm by default. `model.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1` initializes it from torchvision's ImageNet weights, and `model.use_group_norm=true` replaces its BatchNorm layers with GroupNorm, as in the original Diffusion Policy. LeRobot rejects the two together, since swapping the normalization layers would invalidate the pretrained weights. Both values are saved with the checkpoint; checkpoints saved without `use_group_norm` load with BatchNorm.

## Usage

```bash
# Train (best.pth is selected on the validation loss)
scripts/train_aloha.sh trainer.use_ema=true train_dataloader.num_workers=8

# Offline evaluation: denoising loss on the held-out test split
scripts/eval.sh <run_dir>/checkpoints/best.pth experiment=aloha_transfer_cube
scripts/eval.sh <run_dir>/checkpoints/best.pth experiment=aloha_transfer_cube apply_ema=true

# Closed-loop evaluation: success rate in the simulator, with one MP4 per episode
scripts/rollout.sh <run_dir>/checkpoints/best.pth experiment=aloha_transfer_cube n_episodes=50
scripts/rollout.sh <run_dir>/checkpoints/best.pth experiment=aloha_transfer_cube n_episodes=50 apply_ema=true

scripts/test.sh                                     # test suite
```

Drop `experiment=aloha_transfer_cube` for PushT. Every script forwards extra arguments as Hydra overrides, e.g. `scripts/train.sh trainer.iterations=50000`. `NUM_GPUS=<n> scripts/train.sh` trains on several GPUs and `ENV_NAME=<env>` picks another conda env.

To resume training, pass `resume_from=<run_dir>/checkpoints/last` with the same experiment. `apply_ema=true` evaluates the EMA weights and requires a checkpoint trained with `trainer.use_ema=true`.

ALOHA frames are decoded from 480x640 AV1 videos, so data loading is slow with the default `num_workers=0`. On a 22-thread laptop CPU, one batch of 64 took 0.59 s with 0 workers and 0.12 s with 8. Pass e.g. `train_dataloader.num_workers=8`; the ALOHA experiment gives the validation loader the same count. Mixing 0 and >0 workers between the two loaders crashes with `decodeAVFrame ... Invalid data`: LeRobot caches open video decoders per process, and workers forked after the main process has decoded inherit them. The training iteration count is the shared default (`trainer.iterations=10000`); set it explicitly for a full ALOHA run.

### Pretrained policy

The Hub checkpoint `lerobot/diffusion_pusht` is stored in the pre-0.4 LeRobot format. Convert it once into `ckpt/diffusion_pusht`, which is the rollout default:

```bash
scripts/download_pretrained.sh
scripts/rollout.sh n_episodes=10
```

There is no pretrained ALOHA Diffusion Policy, so the ALOHA experiment has no `pretrained_path` default and rollouts need `checkpoint_path`. Another converted LeRobot Diffusion Policy is used with `pretrained_path=<dir>`. `rollout.py` checks that the policy's input and output features, and their shapes, match what the selected simulator produces, and stops with an error otherwise.

## Offline loss versus closed-loop success

- `eval.py` and validation report the diffusion **denoising loss** on held-out demonstrations: how well the model predicts the noise added to demonstrated action chunks. It is cheap and selects `best.pth`, but it does not measure whether the policy completes the task.
- `rollout.py` runs the policy **in closed loop** in the simulator and reports task success. This is the performance measure of a policy. A lower denoising loss does not imply a higher success rate.

`rollout.py` writes `videos/metrics.json` with `n_episodes`, `n_successes`, `success_rate`, `mean_sum_reward` (mean total reward per episode), `mean_max_reward`, `mean_length` (steps) and one entry per episode (seed, success, reward, length, video). Success is the simulator's own `is_success` flag, read from the finished episode's `final_info`:

- PushT: the T block covers more than 95% of the goal region.
- ALOHA Transfer Cube (gym-aloha 0.1.x): the step reward reaches 4, i.e. the cube touches the left gripper finger and not the table. The episode terminates at that step. The reward stages are 1 (right gripper touches the cube), 2 (lifted), 3 (left gripper touches it), 4 (transferred).

The ALOHA condition is checked on a single step, as in ACT and LeRobot's own evaluation. A policy that knocks the cube up with the left finger can therefore count as a success. An untrained policy did so once in testing. Watch the videos before trusting a success rate from an early checkpoint.

Episode `i` uses seed `seed + i` for both the simulator (initial cube or block pose) and the diffusion noise, so an episode is reproducible on its own and different checkpoints are compared on the same initial states. Episodes last at most `max_steps`, by default the simulator's time limit (PushT 300 steps at 10 Hz, ALOHA 400 steps at 50 Hz). Videos play at the simulator's rate.

The PushT videos show the denoising of each action chunk and the plan being executed (`diffusion_overlay`, `denoising_frames`). The overlay draws 2-D positions, so it is disabled for ALOHA, whose actions are joint targets.

## Outputs

Each run writes to its own directory, along with its resolved config, log file and W&B files:

| Command | Directory |
| --- | --- |
| train | `outputs/<name>/<date>/<time>/`, with `checkpoints/best.pth` (weights only) and `checkpoints/last/` (full state for resuming) |
| eval | `outputs/<name>/eval/<date>/<time>/metrics.json` |
| rollout | `outputs/<name>/rollout/<date>/<time>/videos/` (MP4s and `metrics.json`) |

`<name>` is the `name` config value: `diffusion_pusht` by default, `diffusion_aloha_transfer_cube` for the ALOHA experiment.

## Data splits

Episodes are shuffled with `seed` and split by whole episode: 10% validation, 10% test and the rest for training (`val_fraction` and `test_fraction` in `config/dataloader/dataset/*.yaml`, which must match across the three files). For ALOHA this is 40/5/5 episodes. Training selects `best.pth` on the validation loss. `eval.py` uses the test split by default; pass `split=val` to evaluate on the validation split. Metrics are logged as `eval/<split>/<metric>`.

## Notes

- Each evaluation samples fresh diffusion noise, so the loss varies slightly between runs.
- Normalization statistics come from the full dataset metadata, not from the training episodes only, so a small amount of information leaks into validation and test.
- Checkpoints store the model class path (`src.models.model.DiffusionPolicyWrapper`). Renaming the class breaks loading of older checkpoints.
