# Scripts

Entry-point wrappers. Each one activates the conda env (`diffpolicy` by default, override with `ENV_NAME=...`), runs from the repo root, and forwards extra arguments as Hydra overrides.

| Script | What it does |
| --- | --- |
| `setup.sh` | Creates the conda env (Python 3.12) if missing and installs `requirements-lerobot.txt` and `requirements-dev.txt`. |
| `train.sh` | Runs `main.py`. Output goes to `outputs/<name>/<date>/<time>/`. `NUM_GPUS=n` launches with `accelerate launch --multi_gpu`. |
| `eval.sh <checkpoint>` | Runs `eval.py` on the test split. Add `split=val` for the validation split. |
| `rollout.sh [checkpoint]` | Runs `rollout.py` in the PushT simulator and saves videos. Without a checkpoint, uses the pretrained policy in `ckpt/diffusion_pusht`. |
| `download_pretrained.sh` | Downloads `lerobot/diffusion_pusht` (or `--repo-id`) from the Hub and converts it to the lerobot 0.6 format in `ckpt/diffusion_pusht` (or `--output-dir`). |
| `test.sh` | Runs pytest. Extra arguments go to pytest. |
| `_env.sh` | Shared helper sourced by the others. Not meant to be run directly. |

Examples:

```bash
scripts/train.sh trainer.iterations=50000 trainer.use_ema=true
scripts/train.sh resume_from=outputs/diffusion_pusht/<date>/<time>/checkpoints/last
scripts/eval.sh outputs/diffusion_pusht/<date>/<time>/checkpoints/best.pth apply_ema=true
scripts/rollout.sh outputs/diffusion_pusht/<date>/<time>/checkpoints/best.pth n_episodes=10
scripts/download_pretrained.sh && scripts/rollout.sh n_episodes=10
scripts/test.sh -k config
```

Keep reusable implementations in `src/` and scripts focused on orchestration.
