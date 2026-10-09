# Diffusion Policy project, built by modifying `skeleton` directly

This is a **full copy** of the uploaded `skeleton` project, not an overlay or subclass package. The existing `src/trainer/trainer.py` and `src/inference/inference.py` are edited in place. All existing pipeline functions and infrastructure remain.

## Where Diffusion Policy lives

| File | Responsibility |
| --- | --- |
| `src/dataset/dataset.py` | Reads `lerobot/pusht` via LeRobot, constructs observation/action windows, and performs an episode-disjoint train/validation/test split (`val_fraction` and `test_fraction`, 10% each by default). |
| `src/models/model.py` | Wraps LeRobot's `DiffusionPolicy` and its pre/postprocessors as your project's model (`DiffusionPolicyWrapper`). |
| `src/trainer/trainer.py` | Your original `TrainingPipeline`, with `compute_loss()` implemented for diffusion and a new `eval()` alias for in-training validation. All checkpoint, Accelerate, EMA, scheduler, resume, logging and `validate()` behavior is retained. |
| `src/inference/inference.py` | Your original `InferencePipeline`, with `compute_metrics()` implemented for offline diffusion loss; `predict()`, `metric_weight()`, `on_prediction()` and `run()` retained. |
| `main.py` / `eval.py` | Your original training and evaluation entry points, unchanged. |
| `rollout.py` | Extra entry point for closed-loop PushT simulation, recording MP4 videos. |
| `config/` | Original Hydra config hierarchy, with dataset/model settings for diffusion. |

No new `diffusion_trainer.py` or `diffusion_inference.py` subclasses exist.

## Install

```bash
python -m pip install -r requirements-lerobot.txt
```

`requirements-lerobot.txt` includes the original `requirements.txt`. LeRobot downloads the `lerobot/pusht` dataset from the Hugging Face Hub to its cache. `data_root` is preserved in the Dataset constructor for template compatibility but does not override the LeRobot cache.

## Train and evaluate

```bash
python main.py wandb.mode=offline
python eval.py checkpoint_path=/ABSOLUTE/PATH/TO/checkpoints/best.pth wandb.mode=offline
python eval.py checkpoint_path=/ABSOLUTE/PATH/TO/checkpoints/best.pth split=val  # evaluate on the validation split instead
```

Training uses the train split and selects `best.pth` on the validation split. `eval.py` evaluates on the held-out test split by default, which is never seen during training or checkpoint selection. Metrics are logged as `eval/<split>/<metric>`.

Validation and standalone `eval.py` compute held-out **denoising loss**. This is not the same as closed-loop success in simulation. The training pipeline retains `validate()`, and adds `eval()` (which delegates to `validate()`) so all existing fixtures still work.

## Simulated policy videos

Pretrained official checkpoint:

```bash
python rollout.py pretrained_path=lerobot/diffusion_pusht
```

Checkpoint trained within this project:

```bash
python rollout.py checkpoint_path=/ABSOLUTE/PATH/TO/checkpoints/best.pth
```

Videos are saved under `data/videos`.

## Important limitations

- The published pretrained checkpoint and the newly initialized project's model may have different architectures. `rollout.py` loads the pretrained checkpoint directly without trying to fit it into the project's model.
- Statistics are loaded from overall LeRobot dataset metadata, not recomputed from train-only episodes. This introduces a small leakage into strict offline validation.
- Each validation pass samples fresh diffusion noise. The reported denoising loss therefore has random variation.
- An end-to-end run with the actual LeRobot/Accelerate stack has **not** been performed here. Python syntax and mocked integration checks are included.
- The original test `test_project_placeholders_and_empty_cycle` was updated because dataset/model placeholders now have real implementations. All other skeleton tests and fixture classes are preserved.
