# TASK SM-K4.4-PURE-R4: Bubble lower climb exit onto save-door pin

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — lower climb exit targets save-door pin band
  (`x∈[77,160]`, `y≤400`, stand-pin poses) so R3 re-pin starts on mid-iso
  handoff. Leave open-loop WJ / door phase / CATH alone unless compose requires.
- `docs/tasks/SM-K4.4-PURE-R4-residual.md` — required PROCESS residual.

Do not edit `continuous.py`, `STATUS.md`, CATH controllers, or progression ranks.

## Source and contract

- Preferred source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`
- Optional mid isolation (dev):
  `scratch/post_bubble_mid_climb_pure.state`
- Expected room: `0xACB3` Bubble Mountain
- R4 target: full pure `standing_mid_pinned=True` and **min_y≤260**, then
  peak-cross / top band / ordinary Bat Cave `0xB07A` if compose lands
- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**

## Context

- R3 residual: re-pin phase shipped; mid-iso still pins + min_y≈260; full pure
  min_y≈364 with `standing_mid_pinned=False` because lower exits broad mid
  (`100≤x≤320`) and re-pin cannot reach save-door platform in budget.
- Mid-iso source class: pose=26 x≈98 y≈374 |vy|≈1.
- Maprando strat 154: standing save-door platform → run-jump cavity WJ.

## Do

1. One named change: lower climb break / HJ bias exits on save-door pin band
   (not broad cavity mid), then existing R3 re-pin + R2 open-loop.
2. Keep wrong-door hard-avoid + cavity x cap.
3. Pure probe; successor state only if full GREEN to Bat.
4. Residual PROCESS fields; no continuous/STATUS claim.

## Acceptance

- [ ] Source loads at `0xACB3` (CATH-04 pin band)
- [ ] Full pure `standing_mid_pinned=True` and min_y≤260
- [ ] Ordinary `0xB07A` without warp / item grants (if top lands)
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

- GREEN → `SM-K4.5-PURE` or compose/stabilize
- RED → next one named phase (peak-cross / open-loop retune / door)

### PROCESS residual (required on exit)

Result · Files changed · Verify paste · Acceptance · Residual risks ·
Next action · Non-claims · Probe pin.
