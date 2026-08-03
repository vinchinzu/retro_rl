# TASK SM-K4.4-PURE-R8: Bubble right-shelf re-seat then top hop

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — after R7 peak-cross reaches right-structure
  x (x≥300) with height class, **re-seat on right shelf** (y≤370, prefer
  y≤340, x∈[320,390]) then charged HJ to top band (y≤200, x≥300) in
  `0xACB3`. Leave R5 lower + R6 lip + R7 height-class gate alone unless
  max_x/min_y regress. Door phase only if top lands.
- `docs/tasks/SM-K4.4-PURE-R8-residual.md` — required PROCESS residual.

Do not edit `continuous.py`, `STATUS.md`, CATH controllers, or progression ranks.

## Source and contract

- Preferred source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`
- Optional mid isolation:
  `scratch/post_bubble_mid_climb_pure.state`
- Expected room: `0xACB3` Bubble Mountain
- R8 target: full pure `top_reached=True` (y≤200, x≥300); ordinary Bat Cave
  `0xB07A` if door compose lands. Keep min_y≤280, pin, launched.
- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**

## Context

- R7 residual: peak-cross max_x=387 min_y=270 pin True top still red;
  end ~(326,484). Place-proven shelves `(384,363)…(336,219)` one-hop to top.
- Gap: natural re-seat on those shelves (not just max_x flyby / floor thrash).
- Hard-cap x to avoid Single Chamber outer-wall trap.

## Do

1. One named change: right-shelf re-seat retention after peak-cross reaches
   right-structure x, then shelf→top hop.
2. Keep R5 lower + R6 lip + R7 height-class peak-cross; wrong-door avoid;
   cavity x cap.
3. Pure probe; successor only if full GREEN to Bat.
4. Residual PROCESS fields; no continuous/STATUS claim.

## Acceptance

- [ ] Source loads at `0xACB3`
- [ ] Full pure min_y≤280 and `standing_mid_pinned=True` (no regression)
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
- RED → next one named door/cross knob
