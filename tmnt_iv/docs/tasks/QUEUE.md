# TMNT IV task queue

Planner (Grok / human) owns design, natural-entry judgment, STATUS, and
integrity. Executors take **one card per session**.

| Doc | Role |
|-----|------|
| **[TRIAGE.md](TRIAGE.md)** | Critical path, parallel tracks |
| **[BACKLOG.md](BACKLOG.md)** | Full ticket list by epic |
| [PROCESS.md](PROCESS.md) | Multi-entry, stabilize, residual schema |
| [TASK_TEMPLATE.md](../TASK_TEMPLATE.md) | Card format |
| [CLEAN_TRACK.md](../CLEAN_TRACK.md) | Clean privilege-reduction contract |
| [CLEAN_PLAYBOOK.md](../CLEAN_PLAYBOOK.md) | Play lessons |
| [STATUS.md](../STATUS.md) | Verified facts |
| [BASELINE_METRICS.md](../BASELINE_METRICS.md) | Assisted continuous metrics |

## Process gates (non-negotiable)

1. **Multi-entry first** (checkpoint + continuous-faithful / power-on) before
   continuous Clean claim.
2. **Stabilize wave** after policy knobs land (suite + assisted dry-run before
   more knobs).
3. **One knob / residual** → next card ID + one change.
4. **Serialize** hot modules: `policy.py`, `record_full_hard_run.py`,
   `STATUS.md`, `BASELINE_METRICS.md`.
5. **Dual track:** Clean suite progress ≠ assisted demotion; grind ≠ continuous.
6. **Force-pass ban:** scaffolds never claim suite/continuous green.

## ★ Live tips (2026-08-01)

| Gate | Status | Evidence / action |
|------|--------|-------------------|
| Power-on → hard credits (assisted) | **continuous** | 00:57:19.635 / 4,667 dmg / 65 e-heals / 0 lives |
| Clean infra (paths / CLI / integrity) | **done** | `tests/test_clean_track.py`; `--clean` defaults |
| Stage 1 Clean suite | **done** | `probe_stage1_clean --suite` 2/2 |
| Stage 2 Alleycat Clean | **in progress** | Metalhead Clean; residual post-pizza 0x5E pile-ons |
| Stage 3 Sewer Clean | **in progress** | LiveHard; residual 0x1C spikes |
| ★ Clean full continuous | **open** | After stages + form-2 |
| Assisted polish | **open** | Techno / Prehist / Starbase / WK buckets |

```bash
# Assisted baseline re-verify
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run --dry-run

# Clean continuous (infra ready; stages still RED — expect fail until S2–S9)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run --clean --dry-run

# Stage Clean suites (existing)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage1_clean --suite
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --suite
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage3_clean --suite
```

## Epic board (product path → Clean full run)

```text
✅ M8 assisted continuous hard credits
✅ Stage 1 Clean suite (Big Apple)
✅ Clean infra (CONTRACT / ARTIFACTS / CLI / INTEGRITY)
▶  Stage 2 Alleycat Clean suite  ← YOU ARE HERE
▶  Stage 3 Sewer Clean suite
⬜  Stages 4–8 Clean suites
⬜  Stage 9 + form-2 Clean (no iframe write)
⬜  ★ Clean continuous hard credits (T4-CLEAN-FULL)
⬜  STATUS secondary Clean section
```

Parallel: **A assisted** damage/heal/iframe polish (`T4-ASSIST-*`).

### Ticket recipe (each Clean stage)

| Step | Kind | Owner |
|------|------|-------|
| 1 | `probe suite` heal=none multi-entry | Executor |
| 2 | `policy knob` if RED (one change) | Executor |
| 3 | `stabilize` suite re-verify (+ assisted dry-run if shared knob) | planner-serial |
| 4 | next stage or continuous | planner |

Whole-run Clean inserts infra cards first, then `T4-CLEAN-FULL` after stages.

## Ready now (Wave-2 — stage Clean suites)

| Track | Cards | Notes |
|-------|-------|-------|
| **CLEAN stages** | `T4-CLEAN-S2`, `T4-CLEAN-S3` | Multi-entry heal=none; one-knob if RED |
| **A assisted** | `T4-ASSIST-TECHNO` | Largest damage bucket; serialize policy |
| **CLEAN infra** | done | residuals under `T4-CLEAN-*-residual.md` |

## Parallel tracks

| Track | What | Integrity |
|-------|------|-----------|
| **A assisted** | Improve already-green continuous | Keep 0 lives lost; update BASELINE after dry-run |
| **CLEAN** | Zero e-HP + zero iframe | `*_clean` artifacts only; STATUS secondary |
| **Grind / lab** | Ollama grind, slash lab | Dual-track only; not continuous evidence |

## Clean track detail

Contract: [`CLEAN_TRACK.md`](../CLEAN_TRACK.md). Does **not** block assisted
polish. Primary STATUS gate remains assisted full clear.

| Gate | Status | Card |
|------|--------|------|
| Dual-path docs | **done** | [`T4-CLEAN-CONTRACT`](T4-CLEAN-CONTRACT.md) |
| Artifact isolation `_clean` | **done** | [`T4-CLEAN-ARTIFACTS`](T4-CLEAN-ARTIFACTS.md) |
| `--clean` CLI + flag wiring | **done** | [`T4-CLEAN-CLI`](T4-CLEAN-CLI.md) |
| Zero assist integrity | **done** | [`T4-CLEAN-INTEGRITY`](T4-CLEAN-INTEGRITY.md) |
| S1 Clean suite | **done** | (playbook / STATUS) |
| S2 Alleycat Clean | open | [`T4-CLEAN-S2`](T4-CLEAN-S2.md) |
| S3 Sewer Clean | open | [`T4-CLEAN-S3`](T4-CLEAN-S3.md) |
| S4–S8 Clean | open | `T4-CLEAN-S4`…`S8` |
| S9 form-2 Clean | open | [`T4-CLEAN-S9`](T4-CLEAN-S9.md) |
| ★ Full Clean continuous | open (after stages) | [`T4-CLEAN-FULL`](T4-CLEAN-FULL.md) |
| Stab / STATUS | gated | `T4-CLEAN-STAB` / `T4-CLEAN-STATUS` |

## Assisted improve detail

| Card | Goal | Baseline |
|------|------|----------|
| [`T4-ASSIST-TECHNO`](T4-ASSIST-TECHNO.md) | Cut Technodrome damage | 1,022 |
| [`T4-ASSIST-PREHIST`](T4-ASSIST-PREHIST.md) | Cut Prehistoric / Slash | 861 |
| [`T4-ASSIST-STARBASE`](T4-ASSIST-STARBASE.md) | Cut Starbase damage | 749 |
| [`T4-ASSIST-WK`](T4-ASSIST-WK.md) | Cut Wounded Knee | 579 |
| [`T4-ASSIST-HEALS`](T4-ASSIST-HEALS.md) | Fewer e-heals (< 65) | 65 |
| [`T4-ASSIST-IFRAME`](T4-ASSIST-IFRAME.md) | Fewer form-2 frames | 4,635 |
| [`T4-ASSIST-DRYRUN`](T4-ASSIST-DRYRUN.md) | Re-record assisted baseline | planner gate |

## Hygiene

- Living markdown cards are **ready / in-flight** work — scaffold from
  `TASK_TEMPLATE.md` when promoting a backlog row.
- Archive residuals after the successor card exists.
- After assisted improve promotion: update `BASELINE_METRICS.md` + `STATUS.md`.
- After Clean continuous: secondary STATUS only; keep primary intervention
  class until program decides Clean is the published default.
