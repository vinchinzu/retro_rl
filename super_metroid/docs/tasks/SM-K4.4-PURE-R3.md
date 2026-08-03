# TASK SM-K4.4-PURE-R3: Bubble standing mid re-pin before open-loop

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — after lower climb, force standing mid re-pin
  (pose 1/2, vy=0, x∈[90,160], y≤400) then keep R2 open-loop launch/climb.
  Leave CATH geometry, door phase, and open-loop WJ pattern alone unless
  compose requires.
- `docs/tasks/SM-K4.4-PURE-R3-residual.md` — required PROCESS residual.

Do not edit `continuous.py`, `STATUS.md`, CATH controllers, or progression ranks.

## Source and contract

- Preferred source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`
- Optional mid isolation (dev knob only):
  `scratch/post_bubble_mid_climb_pure.state` (lower pin ≈x77–112 y369–402)
- Expected room: `0xACB3` Bubble Mountain
- R3 target: full pure reproduces mid-iso height class (**min_y≤260**), then
  peak-cross toward top band y≤200 / x≥300; compose ordinary Bat Cave `0xB07A`
  if top lands
- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**

## Context

- R2 residual: mid open-loop shipped; full pure still tops **min_y≈388**
  while mid isolation from standing pin reaches **min_y≈260**. Handoff after
  lower climb is the gap (broad mid break vs save-door standing pin).
- Maprando strat 154: standing save-door platform → run-jump cavity right
  wall → walljump with HiJump. Far-right SC wall remains height trap (x hard-cap).
- R3 one-change: force standing mid re-pin before R2 launch so full pure
  matches mid-iso start.

## Do

1. One named change: after lower climb, settle/HJ-retry until standing mid pin
   (pose 1/2, vy=0, x∈`_BUBBLE_MID_STAND_X`, y≤`_BUBBLE_MID_Y`), then R2
   open-loop launch → alt WJ → peak cross.
2. Keep wrong-door hard-avoid + cavity x cap.
3. Pure probe; successor state only if full GREEN to Bat.
4. Residual PROCESS fields; no continuous/STATUS claim.

## Acceptance

- [ ] Source loads at `0xACB3` (CATH-04 pin band)
- [ ] Full pure min_y≤260 (mid-iso class) and/or top band y≤200 / x≥300
- [ ] Ordinary `0xB07A` without warp / item grants (compose if top lands)
- [ ] Successor state only if pure GREEN
- [ ] Unit/registration green
- [ ] Residual PROCESS fields; no continuous/STATUS claim

## Verify

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
```

## Residual routing

- GREEN → `SM-K4.5-PURE` (Bat Cave → Speed Hall) or compose/stabilize
- RED → next one named phase (peak-cross timing / door-1 runway / cavity wall pin)

### PROCESS residual (required on exit)

Result · Files changed · Verify paste · Acceptance · Residual risks ·
Next action · Non-claims · Probe pin.
