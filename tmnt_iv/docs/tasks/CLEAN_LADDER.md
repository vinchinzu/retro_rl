# Clean stage ladder — thin goals

**Pizza-only Clear is much harder than emergency-HP assist.** Do not give
executors “green the whole stage suite” as one card. Progress is a **ladder of
partial goals**; each rung is a separate card with its own GREEN/RED.

Process: [PROCESS.md](PROCESS.md) · Queue: [QUEUE.md](QUEUE.md) ·
Template: [TASK_TEMPLATE.md](../TASK_TEMPLATE.md).

## Why thin rungs

| Mode | What “success” means | Difficulty |
|------|----------------------|------------|
| **Assisted** | Emergency HP ≤16→80 + form-2 iframe | Already continuous green |
| **Clean** | Pizza-only (`heal=none`); 0 e-HP writes; 0 lives lost | Stage suites still RED after S1 |

A weak executor can **probe**, **A/B one constant**, or **fill residual from
JSON**. It cannot invent multi-knob theory and STATUS-promote in one session.

## Standard rungs (every stage Sn)

Order is fixed. Skip only rungs already GREEN with artifact proof.

| # | Card suffix | Goal (acceptance) | Policy edit? | Model |
|---|-------------|-------------------|--------------|-------|
| 0 | **PROBE** | Run suite / single entries; residual table **copied from JSON only** | No | Flash / Gemini |
| 1 | **BOSS** | Boss entry (or LiveHard boss path) → `stage_advance`, 0 lives | One stage-local knob **only if RED** | Luna / Gemini |
| 2 | **LATE** | Pre-boss / late mid checkpoint → `stage_advance` | One knob if RED | Luna / Gemini |
| 3 | **REACH** | Full-stage checkpoint: improve **metric** only (see below) | One knob if RED | Luna / Gemini |
| 4 | **CKPT** | Full-stage fight-ready checkpoint → `stage_advance` | One knob if RED | Luna |
| 5 | **BRIDGE** | Continuous-faithful entry (Stage{n-1}_Clear / power-on / LiveHard) → `stage_advance` | One knob if RED | Luna |
| 6 | **SUITE** | All required suite entries green (verify from `clean_suite.json`) | No new knobs | Flash + planner |
| 7 | **STAB** | Re-run suite + assisted dry-run; report deltas; **no knobs** | No | Flash / Gemini |

**Epic shell** `T4-CLEAN-Sn` is **not** an executor ticket — it tracks child
status only. Executors take **one child card** per session.

### REACH metrics (partial credit — not suite green)

When full checkpoint still dies, REACH cards may GREEN if **one** of:

- **Farther:** higher `frames` before `life_loss` / timeout (same entry)
- **Safer:** lower `damage_taken` **or** higher `min_hp` at same progress band
- **Softer hits:** lower `max_hit` (e.g. eliminate 24-dmg pile-ons)
- **Window:** named progress/frame band no longer contains the lethal hit

Residual must paste before/after from JSON. REACH never claims SUITE green.

### Hard bans on every Clean card

- Do **not** edit `STATUS.md` / QUEUE gate rows / `BASELINE_METRICS.md`
- Do **not** invent suite numbers — only quote `recordings/stageN_clean_track/*.json`
- Do **not** change constants listed under residual **Rejected knobs** without a
  planner override line on the card
- Do **not** land two interacting knobs in one card
- Do **not** re-open [CLEAN_PLAYBOOK.md](../CLEAN_PLAYBOOK.md) bans
- Suite / continuous green never claimed from unit tests alone

## Stage board (rung status)

Update when artifacts change. Evidence paths under `tmnt_iv/recordings/`.

### S1 Big Apple — **SUITE done**

| Rung | Status | Note |
|------|--------|------|
| SUITE | **done** | `stage1_clean_track/clean_suite.json` 2/2 |

### S2 Alleycat (stage byte 1) — **active**

Evidence: `stage2_clean_track/clean_suite.json` (2026-08-02: **2/4**).

| Rung | Card | Status | Note |
|------|------|--------|------|
| PROBE | [T4-CLEAN-S2-PROBE](T4-CLEAN-S2-PROBE.md) | ready | Re-baseline after any policy churn |
| BOSS | [T4-CLEAN-S2-BOSS](T4-CLEAN-S2-BOSS.md) | **done** | Metalhead `Boss2` clear |
| LATE | [T4-CLEAN-S2-LATE](T4-CLEAN-S2-LATE.md) | **done** | `Stage2_Clear_w17_*` clear |
| REACH | [T4-CLEAN-S2-REACH](T4-CLEAN-S2-REACH.md) | open | Kill post-pizza 24-dmg / survive farther on `Stage2` |
| CKPT | [T4-CLEAN-S2-CKPT](T4-CLEAN-S2-CKPT.md) | open | Full `Stage2` → stage_advance |
| BRIDGE | [T4-CLEAN-S2-BRIDGE](T4-CLEAN-S2-BRIDGE.md) | open | `stage1_clear` → stage_advance (not timeout) |
| SUITE | [T4-CLEAN-S2-SUITE](T4-CLEAN-S2-SUITE.md) | gated | All required entries |
| STAB | [T4-CLEAN-S2-STAB](T4-CLEAN-S2-STAB.md) | gated | After any KEEP knob |
| EDGE | [T4-CLEAN-S2-EDGE](T4-CLEAN-S2-EDGE.md) | open | Residual next: pack micro-pause / left-flank hold |

Epic: [T4-CLEAN-S2](T4-CLEAN-S2.md). Residual: [T4-CLEAN-S2-residual](T4-CLEAN-S2-residual.md).

### S3 Sewer (stage byte 2) — **active**

Prefer **`LiveHardStage3` (lives=2)**. Last-life `Stage3`/`Boss3` fade artifact
is not a Clean gate.

| Rung | Card | Status | Note |
|------|------|--------|------|
| PROBE | [T4-CLEAN-S3-PROBE](T4-CLEAN-S3-PROBE.md) | ready | LiveHard suite baseline |
| BOSS | [T4-CLEAN-S3-BOSS](T4-CLEAN-S3-BOSS.md) | open | Rat King finish holds on LiveHard path |
| REACH | [T4-CLEAN-S3-REACH](T4-CLEAN-S3-REACH.md) | open | Cut residual 0x1C spikes / farther LiveHard |
| CKPT | [T4-CLEAN-S3-CKPT](T4-CLEAN-S3-CKPT.md) | open | LiveHard full stage_advance |
| BRIDGE | [T4-CLEAN-S3-BRIDGE](T4-CLEAN-S3-BRIDGE.md) | open | Stage2_Clear → sewer live |
| SUITE | [T4-CLEAN-S3-SUITE](T4-CLEAN-S3-SUITE.md) | gated | Required LiveHard entries |
| STAB | [T4-CLEAN-S3-STAB](T4-CLEAN-S3-STAB.md) | gated | After KEEP knob |

Epic: [T4-CLEAN-S3](T4-CLEAN-S3.md).

### S4–S8 — **gated on probe infra + prior stages**

For each stage: same rungs. First executor card is always **PROBE** (may need
`T4-INFRA-PROBE-Sn` script scaffold first).

| Stage | Epic | First thin cards |
|-------|------|------------------|
| S4 Technodrome | [T4-CLEAN-S4](T4-CLEAN-S4.md) | INFRA-PROBE-S4 → S4-PROBE → BOSS (duo) → … |
| S5 Prehistoric | [T4-CLEAN-S5](T4-CLEAN-S5.md) | INFRA-PROBE-S5 → S5-PROBE → BOSS (Slash) → … |
| S6 Skull | [T4-CLEAN-S6](T4-CLEAN-S6.md) | INFRA-PROBE-S6 → …; **no global pizza seek** |
| S7 Wounded Knee | [T4-CLEAN-S7](T4-CLEAN-S7.md) | INFRA-PROBE-S7 → … |
| S8 Neon | [T4-CLEAN-S8](T4-CLEAN-S8.md) | INFRA-PROBE-S8 → … |

Child cards for S4–S8 are **spawned when the stage is unlocked** (copy
S2-PROBE / S2-BOSS templates). Do not open S4 CKPT while S2/S3 SUITE red
unless planner explicitly parallelizes **probe-only** work.

### S9 Starbase + form-2 — **hard Clean gate**

| Rung | Card | Status | Note |
|------|------|--------|------|
| PROBE | spawn when unlocked | open | Needs `probe_stage9_clean` |
| WAVE | Starbase waves pizza-only | open | Separate from form-2 |
| F2 | Form-2 kill, **iframe frames == 0** | open | Hard gate; pairs ASSIST-IFRAME |
| SUITE / STAB | after WAVE+F2 | gated | |

Epic: [T4-CLEAN-S9](T4-CLEAN-S9.md).

### ★ Full continuous Clean

| Card | Goal |
|------|------|
| [T4-CLEAN-FULL-ATTEMPT](T4-CLEAN-FULL-ATTEMPT.md) | Run `--clean --dry-run`; residual = first death stage only |
| [T4-CLEAN-FULL](T4-CLEAN-FULL.md) | Epic / green claim (planner) after stages |
| [T4-CLEAN-STAB](T4-CLEAN-STAB.md) | Dual re-verify |
| [T4-CLEAN-STATUS](T4-CLEAN-STATUS.md) | STATUS secondary (planner only) |

## Assisted track (also thin)

Assisted continuous is already green. Damage cards are **not** “cut stage to
zero” — same partial ladder:

| Suffix | Goal |
|--------|------|
| **PROBE** | Checkpoint / RaphFullHard metrics only; no policy |
| **KNOB** | One named constant; probe metrics before/after |
| **STAB** | Assisted dry-run deltas; no BASELINE self-apply |
| **DRYRUN** | Planner: re-record + BASELINE promote |

Epics `T4-ASSIST-TECHNO` etc. remain shells; spawn PROBE/KNOB when executing.

## Executor session recipe (Gemini-safe)

1. Read **only** the child card + listed “Read first” paths.
2. Run the **one** verify command on the card.
3. If policy allowed: change **one** named knob; else zero code.
4. Write residual: Result, Verify paste from JSON, Next card ID, One change.
5. Stop. No STATUS. No second knob. No “while I’m here.”
