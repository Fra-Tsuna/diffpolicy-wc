# Scripts

Entry-point wrappers. Each one activates the conda env (`diffpolicy` by default, override with `ENV_NAME=...`), runs from the repo root, and forwards extra arguments as Hydra overrides.

| Script | What it does |
| --- | --- |
| `setup.sh` | Creates the conda env (Python 3.12) if missing and installs `requirements-lerobot.txt` and `requirements-dev.txt`. |
| `train.sh` | Runs `main.py` (PushT; `experiment=aloha_transfer_cube` for ALOHA). Output goes to `outputs/<name>/<date>/<time>/`. `NUM_GPUS=n` launches with `accelerate launch --multi_gpu`. |
| `eval.sh <checkpoint>` | Runs `eval.py`: offline denoising loss on the test split. Add `split=val` for the validation split and the checkpoint's `experiment=...`. |
| `rollout.sh [checkpoint]` | Runs `rollout.py`: closed-loop episodes in the experiment's simulator (PushT, or ALOHA with `experiment=aloha_transfer_cube`), saving MP4 videos and `metrics.json` with the success rate. Without a checkpoint, PushT uses the pretrained policy in `ckpt/diffusion_pusht`; ALOHA has no pretrained policy and needs a checkpoint. |
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
scripts/train.sh experiment=aloha_transfer_cube trainer.use_ema=true train_dataloader.num_workers=8 val_dataloader.num_workers=8
scripts/eval.sh outputs/diffusion_aloha_transfer_cube/<date>/<time>/checkpoints/best.pth experiment=aloha_transfer_cube apply_ema=true
scripts/rollout.sh outputs/diffusion_aloha_transfer_cube/<date>/<time>/checkpoints/best.pth experiment=aloha_transfer_cube n_episodes=50 apply_ema=true
scripts/test.sh -k config
```

Keep reusable implementations in `src/` and scripts focused on orchestration.
