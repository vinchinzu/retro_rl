# TASK SM-ESCAPE-01: Tourian escape room-chain scaffold (dev epic)

## Recipe step
boss pipeline / endgame scaffold (continuous deferred)

## Model
Luna

## Wave type
implement

## Own files only
- `combat/escape.py` (**create**) **or** `dev/escape_scaffold.py` (**create**)
  — pick one home; prefer `combat/escape.py` if it only holds pure helpers
- `tests/test_escape_scaffold.py` (**create**)
- optional note

Do not edit continuous / STATUS. Prefer **not** editing mother_brain_dev unless
import-only. No protocol wrap race.

## Context
- Probes exist: `scripts/probe/mother_brain.py` capture-escape / run-escape.
- States: `dev_escape_room1.state`, MB anchors.
- Goal: structured multi-room escape scaffold (rooms 1–4 + landing site
  constants) with timeout-bounded stubs — not continuous gold.

## Read first
- `scripts/probe/mother_brain.py`
- `dev/mother_brain_dev.py` (if present) room constants
- `docs/BOSS_PIPELINE.md` endgame notes if any
- `combat/protocol.py` (read only)

## Do
1. Constants for escape rooms + landing site.
2. Stub `play_escape_room_n` or single `play_escape_chain_scaffold` with
   per-room timeouts and evidence dict.
3. Unit tests: constants + callable + timeout field.
4. Residual: natural MB defeat + continuous escape = PLANNER-GATE.

## Acceptance
- [ ] Scaffold + tests
- [ ] Non-claims: not continuous clear

## Verify
```bash
uv run pytest super_metroid/tests/test_escape_scaffold.py -q
```
