# TASK SM-K4.4-PURE-R5: Bubble lower-left ledge path to save-door pin

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — replace lower HJ dir-bias climb with a
  **dedicated lower-left ledge path** (waypoints → one scripted sub-phase)
  that lands the mid-iso save-door pin (`x∈[77,160]`, `y∈[350,400]`,
  stand-pin poses). Keep R3 re-pin + R2 open-loop + door phase unless compose
  requires a tiny glue change.
- `scripts/probe/bubble_lower_left_recon.py` — optional place/grid recon
  (diagnostic only; not pure evidence).
- `docs/tasks/SM-K4.4-PURE-R5-residual.md` — required PROCESS residual.

Do not edit `continuous.py`, `STATUS.md`, CATH controllers, or progression ranks.

## Source and contract

- Preferred source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`
- Optional mid isolation (dev):
  `scratch/post_bubble_mid_climb_pure.state` (loads pose≈26 x≈105 y≈370)
- Expected room: `0xACB3` Bubble Mountain
- R5 target: full pure `standing_mid_pinned=True` and **min_y≤260** class
  (or honest pin at save-door platform before open-loop), then peak-cross /
  top band / ordinary Bat Cave `0xB07A` if compose lands
- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**

## Context

- R3/R4: re-pin + lower pin-band exit shipped; mid-iso still pins + min_y≈260;
  full pure min_y≈364 with `standing_mid_pinned=False`.
- Dir bias alone cannot put Samus on save-door platform from node-3 entry;
  lower shelves favor cavity mid-right (~x200–320).
- Working handoff: pose=26 x≈98–105 y≈370–374 |vy|≤1.
- Maprando strat 154: standing save-door platform → run-jump cavity WJ.
- Save door node 2 block `[0, 23]` → pixel y≈368.

## Do

1. Recon (place/grid and/or short controller probes) from CATH-04 source for
   (x,y) waypoints on the **left column** that connect entry (~48,637) to
   save-door pin (~105,370).
2. One named change: encode those waypoints as a **scripted lower sub-phase**
   (not more HJ dir bias). Exit when `_on_mid_iso_pin`.
3. Keep wrong-door hard-avoid + cavity x cap; do not retune door phase.
4. Pure probe; successor state only if full GREEN to Bat.
5. Residual PROCESS fields; no continuous/STATUS claim.

## Acceptance

- [ ] Source loads at `0xACB3` (CATH-04 pin band)
- [ ] Full pure `standing_mid_pinned=True` (save-door pin band)
- [ ] Full pure min_y≤260 preferred (mid-iso height class); if only pin lands,
      residual must still report honest min_y / pin metrics
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

- GREEN → `SM-K4.5-PURE` or compose/stabilize / open-loop retune if top still red
- RED → next one named phase (waypoint retune / open-loop / door)

### PROCESS residual (required on exit)

Result · Files changed · Verify paste · Acceptance · Residual risks ·
Next action · Non-claims · Probe pin.
