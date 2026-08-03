# TASK SM-K4.4-PURE-R6: Bubble mid open-loop peak-cross from live pin

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — retune **phase 2 open-loop / peak-cross**
  from the live full-pure standing save-door pin (R5 lands
  `standing_mid_pinned=True`). Target min_y≤260 and preferably top band
  y≤200 x≥300 still in `0xACB3`. Leave R5 lower ledge path and door phase
  alone unless pin regresses.
- `docs/tasks/SM-K4.4-PURE-R6-residual.md` — required PROCESS residual.

Do not edit `continuous.py`, `STATUS.md`, CATH controllers, or progression ranks.

## Source and contract

- Preferred source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`
- Optional mid isolation (dev):
  `scratch/post_bubble_mid_climb_pure.state`
- Expected room: `0xACB3` Bubble Mountain
- R6 target: full pure min_y≤260 and/or top band; ordinary Bat Cave
  `0xB07A` if compose lands
- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**

## Context

- R5 residual: lower-left multi-hop shipped; full pure
  `standing_mid_pinned=True` launched=True min_y≈364 top_reached=False.
- Mid-iso: pin True min_y≈292 (still shy of prior ≈260 peak).
- Maprando strat 154: standing save-door platform → run-jump cavity WJ.
- Hard-cap x to avoid Single Chamber outer-wall height trap.

## Do

1. One named change: retune mid open-loop launch / alternating WJ / peak-cross
   so full pure climbs past pin height (min_y≤260 class) and toward top band.
2. Keep R5 lower path, wrong-door avoid, cavity x cap.
3. Pure probe; successor only if full GREEN to Bat.
4. Residual PROCESS fields; no continuous/STATUS claim.

## Acceptance

- [ ] Source loads at `0xACB3`
- [ ] Full pure `standing_mid_pinned=True` (no R5 regression)
- [ ] Full pure min_y≤260 preferred; top band / `0xB07A` if lands
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

- GREEN → compose/stabilize or door if needed
- RED → next one named open-loop / door knob
