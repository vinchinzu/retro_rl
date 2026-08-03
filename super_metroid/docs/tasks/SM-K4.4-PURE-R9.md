# TASK SM-K4.4-PURE-R9: Open-loop land right shelf then top hop

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — after R6 lip height class, **scripted
  open-loop** (or short hop sequence from recon) that lands **grounded** on
  a right-structure shelf solid class `(384,363)` / `(368,331)` / `(352,283)`
  then charged HJ to top band. Leave R5 lower + R6 lip pad alone unless height
  regresses. Prefer offline recon first.
- `docs/tasks/SM-K4.4-PURE-R9-residual.md` — required PROCESS residual.
- Optional diagnostic only: `scripts/probe/` recon helper (not pure proof).

Do not edit `continuous.py`, `STATUS.md`, CATH controllers, or progression ranks.

## Source and contract

- Preferred source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`
- Expected room: `0xACB3` Bubble Mountain
- R9 target: full pure `top_reached=True`; ordinary `0xB07A` if door lands
- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**

## Context

- R7/R8: peak-cross max_x=387 min_y=270 pin True top red; reactive
  shelf_drop no advance. Place shelves one-hop to top in isolation.
- Mid reseat nubs ~(140–175, 270–295) too narrow to chain.

## Do

1. One named change: open-loop / sequenced land on right shelf then top hop.
2. Keep R5/R6; wrong-door avoid; cavity x cap.
3. Pure probe; successor only if full GREEN to Bat.
4. Residual PROCESS fields; no continuous/STATUS claim.

## Acceptance

- [x] Source loads at `0xACB3`
- [x] Full pure pin + min_y≤280 (no regression) — **min_y=260**
- [ ] Full pure top band y≤200 x≥300 in `0xACB3` — **RED** (see residual)
- [ ] Ordinary `0xB07A` if door lands; successor only if pure GREEN
- [x] Unit/registration green
- [x] Residual PROCESS fields; no continuous/STATUS claim

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
