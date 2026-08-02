# TASK SM-ARCH-CLEAN-TRACK: Clean-track / continuous structure debt

## Recipe step
structure

## Model
Planner / implement

## Wave type
refactor

## Context

Strict code-quality review (2026-08-02) of the Clean-track + shared-video
working tree. Product direction is right; structure had clear debt before
merge-quality approval.

See residual for the living do-list.

## Own files (this card / residual work)

- `super_metroid/routes/runtime.py`
- `super_metroid/routes/continuous.py`
- `super_metroid/scripts/record/continuous.py`
- `super_metroid/tests/test_clean_track.py`
- residual: `docs/tasks/SM-ARCH-CLEAN-TRACK-residual.md`

## Do not

- Change assisted default tip or program gate Intervention class
- Force-pass Clean continuous without dual green evidence
- Touch TMNT in this SM structure card (separate TMNT residual)

## Acceptance (SM structure)

- [x] Single `resolve_clean_resources` helper (no copy-pasted ternaries)
- [x] No `inspect.signature` dispatch in `run_to`
- [x] Morph uses `finish_report` (same integrity dialect as bombs+)
- [x] Dead `RouteSession.video_config` removed
- [ ] (optional follow-up) Collapse thin post-Supers `run_start_to_*` kwargs surface
- [ ] (optional) Promote `clean_artifact_stem` to shared harness after TMNT adopt

## Verify

```bash
uv run pytest super_metroid/tests/test_clean_track.py super_metroid/tests/test_video_presets.py -q
```
