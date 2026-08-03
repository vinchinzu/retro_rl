# TMNT IV backlog (ticket index)

Atomic cards for Clean full run + assisted continuous improvement. Living
executor detail lives in individual `T4-*.md` files; this file is the index.

Process: [PROCESS.md](PROCESS.md) · Queue: [QUEUE.md](QUEUE.md) ·
Triage: [TRIAGE.md](TRIAGE.md) · **Thin Clean rungs:** [CLEAN_LADDER.md](CLEAN_LADDER.md).

**Rule:** epic shells (`T4-CLEAN-Sn`, `T4-CLEAN-FULL`, fat `T4-ASSIST-*`) are
**not** executor tickets. Executors take **one child card** per session.
Pizza-only Clear ≫ emergency-HP assist — prefer REACH/PROBE over “suite green.”

## Epic: CLEAN (privilege reduction → Bronze / Clean full clear)

### Infra (done)

| ID | Goal | Status |
|----|------|--------|
| T4-CLEAN-CONTRACT | Docs dual-path + ASSIST pointer | **done** |
| T4-CLEAN-ARTIFACTS | `*_clean` report/video stems | **done** |
| T4-CLEAN-CLI | `--clean` disables e-HP + iframe | **done** |
| T4-CLEAN-INTEGRITY | Fail clean run if assists > 0 | **done** |
| T4-CLEAN-S1 | Stage 1 pizza-only suite | **done** (STATUS) |

### S2 Alleycat — thin children (active)

| ID | Goal | Status |
|----|------|--------|
| [T4-CLEAN-S2](T4-CLEAN-S2.md) | Epic shell | tracker |
| T4-CLEAN-S2-PROBE | Suite baseline from JSON | ready |
| T4-CLEAN-S2-BOSS | Metalhead pizza-only | **done** |
| T4-CLEAN-S2-LATE | Pre-boss w17 pizza-only | **done** |
| T4-CLEAN-S2-REACH | Full Stage2 metric progress | open |
| T4-CLEAN-S2-EDGE | 0x5E pack edge-wait one knob | open |
| T4-CLEAN-S2-CKPT | Full Stage2 stage_advance | open |
| T4-CLEAN-S2-BRIDGE | stage1_clear continuous entry | open |
| T4-CLEAN-S2-SUITE | All required entries | gated |
| T4-CLEAN-S2-STAB | Suite + assisted dry-run | gated |

### S3 Sewer — thin children (active)

| ID | Goal | Status |
|----|------|--------|
| [T4-CLEAN-S3](T4-CLEAN-S3.md) | Epic shell | tracker |
| T4-CLEAN-S3-PROBE | LiveHard baseline | ready |
| T4-CLEAN-S3-BOSS | Rat King LiveHard path | open |
| T4-CLEAN-S3-REACH | 0x1C / metric progress | open |
| T4-CLEAN-S3-CKPT | LiveHard stage_advance | open |
| T4-CLEAN-S3-BRIDGE | Stage2_Clear → sewer | open |
| T4-CLEAN-S3-SUITE | Required entries | gated |
| T4-CLEAN-S3-STAB | Suite + assisted dry-run | gated |

### S4–S8 — epics only until unlocked

| ID | Goal | Status |
|----|------|--------|
| [T4-CLEAN-S4](T4-CLEAN-S4.md)…[S8](T4-CLEAN-S8.md) | Epic shells + child ID list | open / gated |
| T4-INFRA-PROBE-S4…S8 | `probe_stageN_clean.py` scaffolds | open |

Spawn PROBE→BOSS→REACH→CKPT→BRIDGE→SUITE→STAB when stage unlocks
([CLEAN_LADDER.md](CLEAN_LADDER.md)).

### S9 + continuous

| ID | Goal | Status |
|----|------|--------|
| [T4-CLEAN-S9](T4-CLEAN-S9.md) | Epic: WAVE + **F2 no iframe** | open |
| T4-INFRA-PROBE-S9 | probe + form-2 entries | open |
| [T4-CLEAN-FULL-ATTEMPT](T4-CLEAN-FULL-ATTEMPT.md) | Clean dry-run; residual first death | ready (expect RED) |
| [T4-CLEAN-FULL](T4-CLEAN-FULL.md) | Epic green claim | gated |
| T4-CLEAN-STAB | Dual re-verify Clean continuous | gated |
| T4-CLEAN-STATUS | STATUS secondary Clean section | gated (planner) |

## Epic: ASSIST (improve already-green continuous)

Fat cards below are **shells**. Execute as PROBE → KNOB → STAB → planner DRYRUN.

| ID | Goal | Status |
|----|------|--------|
| T4-ASSIST-TECHNO | Technodrome damage (PROBE/KNOB children) | ready shell |
| T4-ASSIST-PREHIST | Prehistoric / Slash | ready shell |
| T4-ASSIST-STARBASE | Starbase | ready shell |
| T4-ASSIST-WK | Wounded Knee | ready shell |
| T4-ASSIST-HEALS | Emergency heals below 65 | ready shell |
| T4-ASSIST-IFRAME | Form-2 iframe frames down | ready shell |
| T4-ASSIST-DRYRUN | Assisted dry-run + BASELINE | planner gate |

Assisted child pattern (spawn when executing):

| Suffix | Goal |
|--------|------|
| `*-PROBE` | Metrics only; no policy |
| `*-KNOB` | One named constant; before/after probe |
| `*-STAB` | Dry-run deltas; no BASELINE self-apply |

## Epic: INFRA / TOOLING

| ID | Goal | Status |
|----|------|--------|
| T4-INFRA-PROBE-S4…S9 | Stage Clean probe scripts | open |
| T4-INFRA-RAPH-SRC | Expand RaphFullHard* states | open |

## Counts (approx)

| Epic | Executor-ready thin cards | Epic shells / gated | Done rungs |
|------|---------------------------:|--------------------:|-----------:|
| CLEAN S2–S3 | ~12 open/ready | 2 shells | S2 BOSS+LATE, S1 suite, infra |
| CLEAN S4–FULL | 1 ATTEMPT + INFRA | epics | — |
| ASSIST | spawn on execute | 7 shells | 0 |
