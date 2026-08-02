# TMNT IV backlog review & triage

**As of:** 2026-08-01 (STATUS + CLEAN_PLAYBOOK + Super Metroid process import).

No public GitHub Issues — the live backlog lives in docs:

| Board | Role |
|-------|------|
| [`QUEUE.md`](QUEUE.md) | Live executor wave + ★ tips |
| [`BACKLOG.md`](BACKLOG.md) | Epic summary + ticket list |
| [`PROCESS.md`](PROCESS.md) | Multi-entry, one-knob, residual schema |
| [`../CLEAN_TRACK.md`](../CLEAN_TRACK.md) | Clean privilege-reduction contract |
| [`../CLEAN_PLAYBOOK.md`](../CLEAN_PLAYBOOK.md) | Play lessons |
| [`../STATUS.md`](../STATUS.md) | Verified facts + maturity gate |
| [`../BASELINE_METRICS.md`](../BASELINE_METRICS.md) | Assisted continuous metrics |

## Current state

| Fact | Value |
|------|--------|
| Maturity | **M8** (continuous hard clear + credits) |
| Best continuous | Power-on → hard staff/cast credits |
| Frames / time | **206,718f / 00:57:19.635** (assisted dry-run) |
| Integrity | 0 state loads / 0 stage writes / 0 lives lost / 0 A-special |
| Assists | 65 emergency HP + 4,635 form-2 iframe frames |
| Runtime / intervention | Bronze / Resource-assisted + Protection-assisted |
| Next publication | **Bronze / Clean** (same M8 maturity) |
| Process | Multi-entry + one-knob + dual-track; planner owns continuous + STATUS |

## Critical path (Clean full run — sequential for integrity)

Infra → stage suites (order fixed by playbook) → form-2 play solution →
continuous Clean → STATUS secondary.

| Priority | Ticket / work | Notes |
|----------|---------------|-------|
| **done** | `T4-CLEAN-CONTRACT` + `ARTIFACTS` + `CLI` + `INTEGRITY` | `--clean` + `*_clean` stems + integrity |
| **P0** | `T4-CLEAN-S2` Alleycat multi-entry suite | Early/mid Foot packs after pizza window |
| **P0** | `T4-CLEAN-S3` Sewer suite via LiveHardStage3 | Residual 0x1C spikes; last-life fade artifact |
| P1 | `T4-CLEAN-S4`…`S8` stage suites | One stage at a time; playbook anti-patterns |
| P1 | `T4-CLEAN-S9` Starbase + form-2 **without** iframe | Hard Clean gate |
| **P1 tip** | `T4-CLEAN-FULL` ★ continuous Clean dry-run | After stages green |
| P1 | `T4-CLEAN-STAB` → `T4-CLEAN-STATUS` | Dual re-verify + secondary STATUS |

Do **not** claim whole-run Clean while any stage suite is RED, and do **not**
skip form-2 iframe removal.

## Parallel tracks (do not block Clean ladder)

### 1. Assisted improve (track A)

Already-green continuous; cut damage and assist counts. Checkpoint knobs that
hurt full dry-run stay **parked** (Slash spin 40 lesson).

| Priority | Ticket | Bucket (baseline share) |
|----------|--------|-------------------------|
| P2 | `T4-ASSIST-TECHNO` | Technodrome **1,022** (21.9%) |
| P2 | `T4-ASSIST-PREHIST` | Prehistoric / Slash **861** (18.4%) |
| P2 | `T4-ASSIST-STARBASE` | Starbase **749** (16.0%) |
| P2 | `T4-ASSIST-WK` | Wounded Knee **579** (12.4%) |
| P3 | `T4-ASSIST-HEALS` | Drive e-heals below 65 without life loss |
| P3 | `T4-ASSIST-IFRAME` | Shrink form-2 hold toward 0 (feeds Clean S9) |
| P3 | `T4-ASSIST-DRYRUN` | Re-record assisted baseline after knobs |

### 2. Local grind / lab

`run_local_grind_agent.py` + slash pattern lab. Dual-track only; KEEP does not
auto-merge to production.

### 3. Optional / parked

| Item | Why parked |
|------|------------|
| Stage 1 HazardAvoid production | Jump-through killed Clean |
| Slash `spin_dodge_adx=40` | Probe win, continuous +807 dmg |
| Global pizza seek | Skull soft-lock |
| Empty-screen walk `RIGHT+Y` | Stutter; no Clean benefit |

## Near-term focus (1–2 weeks)

1. **Clean infra wave** — **done** (paths + `--clean` + integrity).
2. **S2 + S3 Clean suites** to suite green (biggest open segment gates).
3. Parallel assisted Technodrome damage cut if policy files serialize cleanly.
4. Form-2 dodge research as soon as mid-route Clean stages stabilize.

## What Super Metroid taught us (apply here)

| SM learning | TMNT IV application |
|-------------|---------------------|
| Dual track: assisted spine + Clean parallel | M8 assisted primary; Clean secondary STATUS |
| Infra before tip (`*_clean` stems) | Never overwrite `tmnt_iv_full_hard_*` |
| Pure-first | Multi-entry suite before continuous claim |
| One-knob + stabilize | One policy group per card; re-verify assisted after shared edits |
| Residual → next card ID | Every RED names one next ticket |
| Clean fail ≠ demote assisted | STATUS primary stays assisted until Clean continuous green |
| Force-pass ban | Units/scaffolds ≠ suite/continuous evidence |
| Planner owns continuous STATUS | Executors propose only |
