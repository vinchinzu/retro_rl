# SMB Overnight Status (2026-02-27)

## Scope
- Focused on uncommitted work in `super_mario_bros/` and `super_mario_editor/`.
- Audited artifact noise vs source-of-truth files.
- Ran quick sanity checks for `hybrid`, `fullgame`, and `8-4` flows.
- Fixed obvious reproducibility blockers.

## Artifact Audit
Useful (kept):
- New SMB pipelines/wrappers/docs/tests:
  - `super_mario_bros/hybrid_pipeline.py`
  - `super_mario_bros/fullgame.py`
  - `super_mario_bros/smb84_pipeline.py`
  - `super_mario_bros/smb83_pipeline.py`
  - `super_mario_bros/run_fullgame.sh`, `run_8_4.sh`, `run_8_3.sh`, `run_hybrid_bg.sh`
  - `super_mario_bros/tests/test_hybrid_pipeline.py`
  - `super_mario_bros/FULLGAME_PIPELINE.md`, `README_8_3.md`, `README_8_4.md`
- New `super_mario_editor/` source tree (Kotlin modules + Python tooling/docs).
- Canonical SMB chained/segment states under:
  - `super_mario_bros/custom_integrations/SuperMarioBros-Nes-v0/*.state`

Junk/transient (now ignored or removed):
- Removed duplicate/stray loose state files outside canonical integration paths.
- Ignored generated SMB optimizer artifacts (`hybrid_report*.json`, `fullgame_*report*.json`, `smb84_*.json`, registry snapshots, concat files).
- Ignored generated map exports under `super_mario_bros/maps/`.
- Ignored `super_mario_editor` debug render artifacts (`*.png`, `*.mp4`, `layer_test/`).

## Reproducibility Fixes
- Made `super_mario_bros/run_fullgame.sh` fail fast with actionable setup error if `.venv` is missing (matches 8-3/8-4 wrappers).
- Removed absolute machine-specific paths in:
  - `super_mario_editor/extend_level.py`
  - `super_mario_editor/self_eval.py`
- Added CLI args to both scripts so ROM paths are overrideable.

## Sanity Checks (quick)
1. Hybrid helper unit tests
- Command:
  - `PYTHONPATH=. .venv/bin/python -m pytest -q super_mario_bros/tests/test_hybrid_pipeline.py`
- Result:
  - `6 passed in 0.04s`

2. Hybrid flow smoke
- Command:
  - `PYTHONPATH=. .venv/bin/python -m super_mario_bros.hybrid_pipeline analyze --route smb_any_percent --selection-context chained --segments smb_8_4 --max-candidates 1 --eval-runs 1 --force-eval --report /tmp/smb_hybrid_smoke.json --registry /tmp/smb_registry_smoke.json`
- Result:
  - Flow executed and wrote report/registry.
  - Current selected `smb_8_4` candidate is incomplete in chained context (expected from existing artifacts).

3. Fullgame flow smoke (train + eval)
- Train command:
  - `super_mario_bros/run_fullgame.sh train --route smb_any_percent --max-candidates 1 --workers 1 --manifest /tmp/smb_fullgame_manifest_smoke.json`
- Eval command:
  - `super_mario_bros/run_fullgame.sh eval --route smb_any_percent --manifest /tmp/smb_fullgame_manifest_smoke.json --report /tmp/smb_fullgame_eval_smoke.json --mode both`
- Result summary:
  - Standalone selected route: `8/8` completed.
  - Chained selected route: breaks at `1-2` (dies), so chained route is still weak.

4. 8-4 flow smoke (manifest + eval)
- Manifest command:
  - `super_mario_bros/run_8_4.sh manifest --segments all --candidate-runs 1 --max-candidates 1 --output /tmp/smb84_manifest_smoke.json`
- Eval command:
  - `super_mario_bros/run_8_4.sh eval --segments 1,2,5 --manifest /tmp/smb84_manifest_smoke.json --runs 1 --chain --chain-mode state --report /tmp/smb84_eval_smoke.json`
- Result summary:
  - seg1 completes; seg2/seg5 currently incomplete; seg3/seg4 had no selected candidate in the minimal manifest run.

## Notes
- This pass is a smoke/sanity pass, not a long retrain.
- Main blocker remaining is policy quality in chained/full-route contexts (not tooling reproducibility).
