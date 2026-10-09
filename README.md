# Diffusion Policy on PushT

Diffusion Policy trained on [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht) with LeRobot, Accelerate, Hydra and W&B.

## Layout

| Path | Contents |
| --- | --- |
| `main.py` | Training entry point. |
| `eval.py` | Offline evaluation of a checkpoint (denoising loss) on the test or val split. |
| `rollout.py` | Closed-loop rollouts in the PushT simulator, saved as MP4 videos. |
| `src/dataset/dataset.py` | LeRobot dataset with observation/action windows and an episode-disjoint train/val/test split. |
| `src/models/model.py` | `DiffusionPolicyWrapper`: LeRobot's `DiffusionPolicy` and its pre/postprocessors as an `nn.Module`. |
| `src/trainer/` | Training loop, validation, EMA, checkpointing and resume. |
| `src/inference/inference.py` | Evaluation loop used by `eval.py`. |
| `config/` | Hydra configs. `train.yaml` is the base; `eval.yaml` and `rollout.yaml` extend it. |
| `scripts/` | Shell wrappers for setup, training, evaluation, rollouts and tests. See [scripts/README.md](scripts/README.md). |

## Setup

Requires conda and an NVIDIA driver >= 525.

```bash
scripts/setup.sh    # creates the `diffpolicy` conda env (Python 3.12) and installs requirements-lerobot.txt + requirements-dev.txt
wandb login         # or pass wandb.mode=offline to every command
```

`requirements.txt` installs torch from the CUDA 12.6 wheels (the default PyPI build needs driver >= 580) and keeps torch, wandb and diffusers within the ranges that lerobot 0.6.1 supports. The dataset is downloaded to the Hugging Face cache on first use.

## Usage

```bash
scripts/train.sh                                   # train
scripts/eval.sh <run_dir>/checkpoints/best.pth     # test-split loss
scripts/rollout.sh <run_dir>/checkpoints/best.pth  # simulator videos
scripts/test.sh                                    # test suite
```

Every script forwards extra arguments as Hydra overrides, e.g. `scripts/train.sh trainer.iterations=50000 trainer.use_ema=true`. `NUM_GPUS=<n> scripts/train.sh` trains on several GPUs and `ENV_NAME=<env>` picks another conda env.

To resume training, pass `resume_from=<run_dir>/checkpoints/last`. To evaluate EMA weights, pass `apply_ema=true` to `eval.sh` or `rollout.sh`.

### Pretrained policy

The Hub checkpoint `lerobot/diffusion_pusht` is stored in the pre-0.4 LeRobot format. Convert it once into `ckpt/diffusion_pusht`, which is the rollout default:

```bash
scripts/download_pretrained.sh
scripts/rollout.sh n_episodes=10
```

## Outputs

Each run writes to its own directory, along with its resolved config, log file and W&B files:

| Command | Directory |
| --- | --- |
| train | `outputs/<name>/<date>/<time>/`, with `checkpoints/best.pth` (weights only) and `checkpoints/last/` (full state for resuming) |
| eval | `outputs/<name>/eval/<date>/<time>/metrics.json` |
| rollout | `outputs/<name>/rollout/<date>/<time>/videos/` |

`<name>` is the `name` config value (`diffusion_pusht` by default).

## Data splits

Episodes are shuffled with `seed` and split by whole episode: 10% validation, 10% test and the rest for training (`val_fraction` and `test_fraction` in `config/dataloader/dataset/*.yaml`, which must match across the three files). Training selects `best.pth` on the validation loss. `eval.py` uses the test split by default; pass `split=val` to evaluate on the validation split. Metrics are logged as `eval/<split>/<metric>`.

## Notes

- Validation and `eval.py` report the diffusion **denoising loss**, which is not the same as task success. Use `rollout.py` to measure success in the simulator.
- Each evaluation samples fresh diffusion noise, so the loss varies slightly between runs.
- Normalization statistics come from the full dataset metadata, not from the training episodes only, so a small amount of information leaks into validation and test.
- Checkpoints store the model class path (`src.models.model.DiffusionPolicyWrapper`). Renaming the class breaks loading of older checkpoints.
