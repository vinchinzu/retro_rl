# TASK SM-K4.4-PURE-R7: Bubble second-hop / peak-cross to top band

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — after R6 lip HJ reaches mid-iso height
  (y≤280), re-seat and **second-hop / peak-cross** so full pure hits top
  band (y≤200, x≥300) in `0xACB3`. Leave R5 lower path + R6 lip launch pad
  alone unless height regresses. Door phase only if top lands.
- `docs/tasks/SM-K4.4-PURE-R7-residual.md` — required PROCESS residual.

Do not edit `continuous.py`, `STATUS.md`, CATH controllers, or progression ranks.

## Source and contract

- Preferred source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`
- Optional mid isolation:
  `scratch/post_bubble_mid_climb_pure.state`
- Expected room: `0xACB3` Bubble Mountain
- R7 target: full pure `top_reached=True` (y≤200, x≥300); ordinary Bat Cave
  `0xB07A` if door compose lands. Keep min_y≤260 and pin.
- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**

## Context

- R6 residual: solid lip launch; full pure pin True **min_y=260**
  top_reached=False end ~(324,474).
- First HJ from lip reaches height class; often fails to re-seat for a higher
  shelf / right-wall WJ cross.
- Hard-cap x to avoid Single Chamber outer-wall trap.

## Do

1. One named change: second-hop / peak-cross retention after lip climb
   reaches y≤280.
2. Keep R5 lower + R6 lip; wrong-door avoid; cavity x cap.
3. Pure probe; successor only if full GREEN to Bat.
4. Residual PROCESS fields; no continuous/STATUS claim.

## Acceptance

- [ ] Source loads at `0xACB3`
- [ ] Full pure min_y≤260 and `standing_mid_pinned=True` (no regression)
- [ ] Full pure top band y≤200 x≥300 still in `0xACB3`
- [ ] Ordinary `0xB07A` if door lands; successor only if pure GREEN
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

- GREEN → compose/stabilize
- RED → next one named cross/door knob
