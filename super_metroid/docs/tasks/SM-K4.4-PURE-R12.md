# TASK SM-K4.4-PURE-R12: New trajectory into right air band (not period thrash)

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement (after RECON if needed)

## Own files only

- `routes/kpdr/k4_norfair.py` — **one named trajectory change** (not
  `_BUBBLE_MIDHIGH_Y` / WJ period alone). Must put natural pure into
  place-proven right band `(x≥340, y∈[280,370])` or grounded shelf, then
  reuse R9 LEFT shelf HJ / period-8 WJ. Freeze R5 lower + R6 lip unless
  height regresses.
- `docs/tasks/SM-K4.4-PURE-R12-residual.md` — required PROCESS residual.
- Optional: `scripts/probe/` recon (not pure proof).

Do not edit `continuous.py`, `STATUS.md`, CATH controllers, or progression.

## Read first

- [`SM-K4.4-PHASE-LADDER.md`](SM-K4.4-PHASE-LADDER.md)
- [`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md) — stagnation rule
- `SM-K4.4-PURE-R11-residual.md` — rejected list is load-bearing

## Context

R11 fixed spin-apex false-land (true ground only). Full pure still
**max_x=349 min_y=260 phase_c_hit=False top red** — same approach gap.
Place air/shelf → top still holds. Lip/mid dash, left WJ, bombs, floor WJ
rejected (see R11 residual).

## Banned without new pin

- Lip walk-left / pre-charge run (height regress)
- Mid-iso dash without enemy clearance plan
- Left-column top hunt (ceiling ~y228)
- Period / mid-high window-only tweaks

## Do

1. RECON: name exact trajectory with numbers (place+vel or save dump).
2. One named controller change implementing that trajectory.
3. Full pure from CATH-04 source; successor only if GREEN to Bat.
4. Residual PROCESS fields.

## Acceptance

- [ ] Full pure Phase C or better (x≥300 y≤430 usable) **or** honest BLOCKED
- [ ] Full pure min_y≤280 (no regress)
- [ ] Full pure top_reached **preferred**; required for hop GREEN
- [ ] Ordinary `0xB07A` if door lands; successor only if pure GREEN
- [ ] Unit green; residual; no STATUS claim

## Verify

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
```
