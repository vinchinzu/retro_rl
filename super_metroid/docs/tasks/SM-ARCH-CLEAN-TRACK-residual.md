# Residual — SM-ARCH-CLEAN-TRACK (code-review do-list)

**Source:** strict code-quality review of Clean track + video + continuous
wiring (2026-08-02). Living checklist — check items off as landed.

## Done (SM — this pass)

| Item | Notes |
|------|--------|
| `resolve_clean_resources` | Single helper in `routes/runtime.py`; all runners + CLI use it |
| Drop `inspect.signature` in `run_to` | Capability flags gate optional kwargs; early runners share common kwargs |
| Morph → `finish_report` | Same integrity / assist payload path as bombs+; morph JSON still emits `video_path` |
| Dead `RouteSession.video_config` | Removed; writer holds config |
| `video_evidence` import | Top-level `retro_harness.video.probe_video_evidence` |

## Open — Super Metroid (structure)

| Priority | Item | Why |
|----------|------|-----|
| P2 | Stop growing `continuous.py` via flag plumbing on every thin `run_start_to_*` | File already >1.7k; tip-spec path should own options once |
| P2 | Extract early-runner shared harness (morph/bombs/spore/supers) like `run_post_supers_tip` | Still four bespoke clean+assist+finish blocks |
| P3 | `VideoCaptureConfig` mutual exclusion | `start_room_id` + `start_frame` both set → silent frame ignore; fail in `__post_init__` |
| P3 | `AssistLike` protocol: explicit energy/ammo counters | Drop `getattr(telemetry, "ammo", {})` soft path in `resource_writes_zero` |
| P3 | `intervention_class` from actual enable bits | Today binary Clean vs Resource-assisted; partial assist-off is under-labeled |
| P3 | Cathedral pure phase helpers | `play_cathedral_entrance_to_cathedral` multi-phase soup → named phase fns when next residual touches it |

## Open — cross-game / TMNT (not SM own-files)

| Priority | Item | Why |
|----------|------|-----|
| P1 | `tmnt_iv/scripts/record_full_hard_run.py` over 1k lines | Extract clean integrity + CLI path helpers before more flags |
| P1 | TMNT adopt `retro_harness.video` | Shared video claimed; TMNT still local footer/pipe in same change set |
| P2 | Promote `clean_artifact_stem` once | Identical helper in SM runtime + `tmnt_iv/paths.py` |
| P2 | Dual `clean` vs `clean_artifacts` stem policy | Partial assist-off rewrites stems without full Clean integrity — simplify or document hard |

## Open — product (Clean track evidence)

| Item | Notes |
|------|--------|
| Clean bombs dual green + STATUS | `SM-CLEAN-STAB` / `SM-CLEAN-STATUS` cards |
| Do not claim Clean full clear | Assisted Frog remains program gate |

## Suggested next cards

1. **SM-ARCH-EARLY-RUNNERS** — one harness for morph→supers (optional; after Clean product green).
2. **T4-ARCH-CLEAN-DECOMPOSE** — TMNT recorder under 1k + shared video (TMNT own-files).
3. Continue spine: **SM-K4-CATH-03** pure (product geometry; orthogonal to this residual).
