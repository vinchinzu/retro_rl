# EPIC T4-CLEAN-S3: Sewer Surfin' Clean (shell — not an executor ticket)

## Recipe step
epic / tracker

## Model
Planner only

## Children (one per session)

See [CLEAN_LADDER.md](CLEAN_LADDER.md#s3-sewer-stage-byte-2--active).

| Order | Card | Goal | Status |
|------:|------|------|--------|
| 0 | [T4-CLEAN-S3-PROBE](T4-CLEAN-S3-PROBE.md) | LiveHard suite baseline | ready |
| 1 | [T4-CLEAN-S3-BOSS](T4-CLEAN-S3-BOSS.md) | Rat King finish on LiveHard path | open |
| 2 | [T4-CLEAN-S3-REACH](T4-CLEAN-S3-REACH.md) | Cut 0x1C spikes / farther LiveHard | open |
| 3 | [T4-CLEAN-S3-CKPT](T4-CLEAN-S3-CKPT.md) | LiveHard full stage_advance | open |
| 4 | [T4-CLEAN-S3-BRIDGE](T4-CLEAN-S3-BRIDGE.md) | Stage2_Clear → sewer | open |
| 5 | [T4-CLEAN-S3-SUITE](T4-CLEAN-S3-SUITE.md) | Required entries green | gated |
| 6 | [T4-CLEAN-S3-STAB](T4-CLEAN-S3-STAB.md) | Suite + assisted dry-run | gated |

## Context

- Stage byte **2**. Prefer **`LiveHardStage3` (lives=2)**.
- Last-life `Stage3`/`Boss3` post-kill `0x0B` fade death is a **checkpoint
  artifact** — not a Clean gate.
- Residual: 0x1C spikes; dumpster thrash rejected; spike LEFT thrash rejected.
- Evidence: `recordings/stage3_clean_track/`.

## Do not

- Hand this epic to an executor as the session card.
- Gate on last-life Boss3 fade artifact.
