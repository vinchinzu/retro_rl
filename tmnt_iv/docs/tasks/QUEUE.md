# TMNT IV task queue

Planner (Grok / human) owns design, natural-entry judgment, STATUS, and
integrity. Executors take **one thin card per session** — never an epic shell.

| Doc | Role |
|-----|------|
| **[CLEAN_LADDER.md](CLEAN_LADDER.md)** | ★ Thin Clean rungs (PROBE→…→SUITE) |
| **[TRIAGE.md](TRIAGE.md)** | Critical path, parallel tracks |
| **[BACKLOG.md](BACKLOG.md)** | Full ticket list by epic |
| [PROCESS.md](PROCESS.md) | Multi-entry, stabilize, residual schema |
| [TASK_TEMPLATE.md](../TASK_TEMPLATE.md) | Card format |
| [CLEAN_TRACK.md](../CLEAN_TRACK.md) | Clean privilege-reduction contract |
| [CLEAN_PLAYBOOK.md](../CLEAN_PLAYBOOK.md) | Play lessons |
| [STATUS.md](../STATUS.md) | Verified facts |
| [BASELINE_METRICS.md](../BASELINE_METRICS.md) | Assisted continuous metrics |

## Process gates (non-negotiable)

1. **Thin rungs:** Clean stage progress is PROBE / BOSS / LATE / REACH / CKPT /
   BRIDGE / SUITE / STAB — not one “green the stage” card
   ([CLEAN_LADDER.md](CLEAN_LADDER.md)).
2. **Pizza-only ≫ assist:** metric REACH wins count; full stage_advance is a
   later rung. Never force-pass SUITE from unit tests.
3. **Multi-entry** before continuous Clean claim (CKPT + BRIDGE → SUITE).
4. **Stabilize** after KEEP knobs (suite + assisted dry-run; no new knobs).
5. **One knob / residual** → next **thin** card ID + one change.
6. **Serialize** hot modules: `policy.py`, `record_full_hard_run.py`,
   `STATUS.md`, `BASELINE_METRICS.md`.
7. **Dual track:** Clean progress ≠ assisted demotion.
8. **Executors never STATUS-promote** or invent suite numbers (JSON only).

## ★ Live tips (2026-08-02)

| Gate | Status | Evidence / action |
|------|--------|-------------------|
| Power-on → hard credits (assisted) | **continuous** | 00:57:19.635 / 4,667 dmg / 65 e-heals / 0 lives |
| Clean infra | **done** | `tests/test_clean_track.py`; `--clean` defaults |
| Stage 1 Clean suite | **done** | `probe_stage1_clean --suite` 2/2 |
| Stage 2 Alleycat Clean | **rungs** | BOSS+LATE **done**; CKPT+BRIDGE **RED** (`clean_suite.json` **2/4** — do not claim 3/4) |
| Stage 3 Sewer Clean | **rungs** | Start `T4-CLEAN-S3-PROBE`; LiveHard; residual 0x1C |
| ★ Clean full continuous | **open** | `T4-CLEAN-FULL-ATTEMPT` only when measuring; expect RED |
| Assisted polish | **open** | TECHNO shell → spawn PROBE/KNOB |

### Ready now (pick **one**)

| Track | Card | Why thin |
|-------|------|----------|
| CLEAN S2 | [`T4-CLEAN-S2-PROBE`](T4-CLEAN-S2-PROBE.md) | Re-baseline JSON after policy churn |
| CLEAN S2 | [`T4-CLEAN-S2-EDGE`](T4-CLEAN-S2-EDGE.md) | One residual pack edge-wait knob |
| CLEAN S2 | [`T4-CLEAN-S2-REACH`](T4-CLEAN-S2-REACH.md) | Metric win on full Stage2 (not full clear) |
| CLEAN S3 | [`T4-CLEAN-S3-PROBE`](T4-CLEAN-S3-PROBE.md) | LiveHard baseline only |
| CLEAN measure | [`T4-CLEAN-FULL-ATTEMPT`](T4-CLEAN-FULL-ATTEMPT.md) | Clean dry-run death stage (expect RED) |
| A assisted | `T4-ASSIST-TECHNO` shell → PROBE first | Damage cut is multi-card |

**Do not assign:** epic `T4-CLEAN-S2` / `S3` / `FULL` as the session card.

```bash
# Assisted baseline re-verify (stabilize only)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run --dry-run

# Clean continuous (expect fail until rungs green)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run --clean --dry-run

# Stage Clean suites
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage1_clean --suite
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --suite
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage3_clean --suite
```

## Epic board (product path)

```text
✅ M8 assisted continuous hard credits
✅ Stage 1 Clean suite
✅ Clean infra
▶  S2 rungs: BOSS+LATE done → REACH/EDGE → CKPT → BRIDGE → SUITE  ← YOU ARE HERE
▶  S3 rungs: PROBE → REACH/BOSS → CKPT → BRIDGE → SUITE
⬜  S4–S8 thin rungs (after INFRA-PROBE-Sn)
⬜  S9 WAVE + F2 (no iframe)
⬜  ★ Clean continuous (FULL-ATTEMPT → STAB → STATUS secondary)
```

Parallel: **A assisted** PROBE/KNOB polish (serialize `policy.py`).

### Ticket recipe (each Clean **rung**)

| Step | Kind | Owner |
|------|------|-------|
| 1 | PROBE (JSON only) | Flash / Gemini |
| 2 | REACH or one KNOB/EDGE | Luna / Gemini |
| 3 | CKPT / BRIDGE when metrics ready | Luna |
| 4 | SUITE verify (no knobs) | Flash |
| 5 | STAB suite + assisted dry-run | Flash; planner on regression |

## Clean track detail

| Gate | Status | Card(s) |
|------|--------|---------|
| Infra | **done** | CONTRACT / ARTIFACTS / CLI / INTEGRITY |
| S1 suite | **done** | playbook / STATUS |
| S2 rungs | active | [T4-CLEAN-S2](T4-CLEAN-S2.md) children |
| S3 rungs | active | [T4-CLEAN-S3](T4-CLEAN-S3.md) children |
| S4–S8 | gated | epic shells + INFRA-PROBE |
| S9 F2 | gated | [T4-CLEAN-S9](T4-CLEAN-S9.md) |
| ★ Full Clean | open | [FULL-ATTEMPT](T4-CLEAN-FULL-ATTEMPT.md) |
| Stab / STATUS | gated | STAB / STATUS (planner) |

## Assisted improve detail

| Shell | Goal | Baseline | Execute as |
|-------|------|----------|------------|
| [T4-ASSIST-TECHNO](T4-ASSIST-TECHNO.md) | Cut Technodrome dmg | 1,022 | PROBE→KNOB→STAB |
| [T4-ASSIST-PREHIST](T4-ASSIST-PREHIST.md) | Prehistoric / Slash | 861 | same |
| [T4-ASSIST-STARBASE](T4-ASSIST-STARBASE.md) | Starbase | 749 | same |
| [T4-ASSIST-WK](T4-ASSIST-WK.md) | Wounded Knee | 579 | same |
| [T4-ASSIST-HEALS](T4-ASSIST-HEALS.md) | e-heals &lt; 65 | 65 | same |
| [T4-ASSIST-IFRAME](T4-ASSIST-IFRAME.md) | form-2 frames down | 4,635 | feeds S9-F2 |
| [T4-ASSIST-DRYRUN](T4-ASSIST-DRYRUN.md) | BASELINE promote | planner | planner |

## Hygiene

- Living cards = ready / in-flight. Epics are trackers only.
- Residual numbers must match suite JSON; force-pass ban.
- After assisted KEEP: STAB dry-run before more knobs; BASELINE = planner.
- After Clean continuous: STATUS **secondary** only.
