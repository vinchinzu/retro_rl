# SMB 8-4 Pipeline (Segmented)

Use `smb84_pipeline.py` for end-to-end 8-4 diagnose/train/eval with area-aware chaining.

## Quick Commands

```bash
cd /home/v/01_projects/11_games/speedrun/retro_rl

# 1) Diagnose state/candidate health
super_mario_bros/run_8_4.sh diagnose --candidate-runs 2

# 2) Optional: train one segment (seg3 is timing-sensitive; raw recommended)
super_mario_bros/run_8_4.sh train --segments 3 --mode raw --iterations 1500 --validation-runs 5

# 3) Build best-candidate manifest for all 8-4 segments
super_mario_bros/run_8_4.sh manifest

# 4) Evaluate standalone + robust stitched chain (state handoff)
super_mario_bros/run_8_4.sh eval --chain --chain-mode state --runs 5

# Optional: strict live single-session chain (debug transition drift)
super_mario_bros/run_8_4.sh eval --chain --chain-mode live --runs 1

# 5) One-shot manifest + eval
super_mario_bros/run_8_4.sh run --chain --chain-mode state --runs 3
```

## Outputs

- Manifest: `super_mario_bros/optimizer/smb84_manifest.json`
- Diagnose report: `super_mario_bros/optimizer/smb84_diagnose.json`
- Eval report: `super_mario_bros/optimizer/smb84_eval_report.json`

## Notes

- This pipeline validates segment start states against expected 8-4 area pointers:
  - seg1 `0x65`, seg2 `0xE5`, seg3 `0x65`, seg4 `0x02`, seg5 `0x65`
- Candidate scoring prefers recordings that complete consistently across repeated runs.
- `chain-mode state` is default and stable for regression checks.
- `chain-mode live` keeps one emulator session and is useful for diagnosing transition desync in real chained play.
