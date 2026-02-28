# SMB 8-3 Training/Eval (Raw Pipeline)

This level is sensitive to action-index quantization. Use the raw-button pipeline in `smb83_pipeline.py`.

## Quick Commands

```bash
cd /home/v/01_projects/11_games/speedrun/retro_rl

# 1) Audit raw vs index behavior for the baseline seed
super_mario_bros/run_8_3.sh audit

# 2) Train from baseline seed (raw GA)
super_mario_bros/run_8_3.sh train \
  --seed super_mario_bros/optimizer/runs/smb_8_3/recording_000.json \
  --population 12 --generations 6 \
  --output-dir super_mario_bros/optimizer/runs/smb_8_3/raw_ga_8_3

# 3) Resume training from checkpoint only (no seed required)
super_mario_bros/run_8_3.sh train \
  --resume super_mario_bros/optimizer/runs/smb_8_3/raw_ga_8_3/ga_raw_best.json \
  --population 12 --generations 6 \
  --output-dir super_mario_bros/optimizer/runs/smb_8_3/raw_ga_8_3

# 4) Evaluate any raw output repeatedly
super_mario_bros/run_8_3.sh eval \
  --actions super_mario_bros/optimizer/runs/smb_8_3/raw_ga_8_3/ga_raw_best_final.json \
  --runs 20
```

## Artifacts

- `ga_raw_best.json`: rolling checkpoint (updated each save interval)
- `ga_raw_best_final.json`: final raw run with cumulative generation metadata

## Notes

- `platformer_common optimize --raw --resume ...` currently requires a seed; this wrapper removes that requirement.
- `audit` is a fast sanity check that catches lossy index mappings on SMB 8-3.
