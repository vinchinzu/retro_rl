# TASK SM-K4.4-PURE-R13: Floor-reclimb trajectory → Phase C on full pure

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna / Grok

## Wave type

implement

## Own files only

- `routes/kpdr/bubble_mountain_params.py` — floor-reclimb constants
- `routes/kpdr/bubble_mountain_mid.py` — deep floor runway + Phase-C sticky climb
- `routes/kpdr/bubble_mountain.py` — climb handoff sticky init
- `docs/tasks/SM-K4.4-PURE-R13-residual.md` — required PROCESS residual
- Tip board docs (AGENTS / QUEUE / phase ladder / BUBBLE TODO)

Do not edit `continuous.py`, `STATUS.md`, CATH controllers, or progression.

## Read first

- [`SM-K4.4-PHASE-LADDER.md`](SM-K4.4-PHASE-LADDER.md)
- [`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md)
- `SM-K4.4-PURE-R12-residual.md` — rejected list is load-bearing

## Context

R12 restored lip stand_pin; full pure still **phase_c_hit=False** on lip arc.
R12 recon: cannot hold min_y≤280 **and** Phase C on the same lip one-shot.
Suggested: new trajectory class (floor / enemy-clear / velocity-matched).

## Do

1. Named trajectory: after height class, if deep (y≥480), floor runway
   ~(270–310, y≥500) charged HJ + WJ (place: Phase C ~(302,428)).
2. Sticky right-structure mode once Phase C fires (no re-drop to floor).
3. Full pure from CATH-04 source; dump Phase C handoff state.
4. Residual PROCESS fields.

## Acceptance

- [x] Full pure Phase C (x≥300 y≤430 usable predicate) **or** honest BLOCKED
- [x] Full pure min_y≤280 (no regress)
- [ ] Full pure top_reached **preferred**; required for hop GREEN
- [ ] Ordinary `0xB07A` if door lands; successor only if pure GREEN
- [x] Unit green; residual; no STATUS claim

## Verify

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin_r13.json --no-red-diag

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
```
