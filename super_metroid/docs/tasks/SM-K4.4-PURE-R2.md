# TASK SM-K4.4-PURE-R2: Bubble mid open-loop walljump (standing pin)

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — `play_bubble_to_bat_cave` mid phase open-loop
  (leave CATH geometry, lower climb, door phase unless compose requires).
- `docs/tasks/SM-K4.4-PURE-R2-residual.md` — required PROCESS residual.

Do not edit `continuous.py`, `STATUS.md`, CATH controllers, or progression ranks.

## Source and contract

- Preferred source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`
- Optional mid isolation:
  `scratch/post_bubble_mid_climb_pure.state` (lower pin ≈x77–112 y369–402)
- Expected room: `0xACB3` Bubble Mountain
- R2 target: top band **y≤200 / x≥300** still in `0xACB3`, then ordinary
  Bat Cave `0xB07A` via top-right green Super door
- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**

## Context

- SM-K4.4-PURE-R1 residual: lower climb real (`mid_reached`); pure min_y≈388;
  mid walljump not pure-green. Maprando strat 154: standing save-door platform
  → run-jump cavity right wall → walljump twice with HiJump. Far-right SC wall
  is a height trap (x≥400 ~y360).
- R1 one-change request: scripted open-loop from **standing** pin (not free-air
  place): run-up + jump + consecutive fresh-A WJ, hard-cap x&lt;400.

## Do

1. One named change: mid phase open-loop from standing save-door pin
   (setup run-up/charge → climb → peak cross with fresh-A WJ).
2. Keep wrong-door hard-avoid + cavity x cap.
3. Pure probe; successor state only if full GREEN to Bat.
4. Residual PROCESS fields; no continuous/STATUS claim.

## Acceptance

- [ ] Source loads at `0xACB3` (CATH-04 pin band)
- [ ] Top band y≤200 / x≥300 still in `0xACB3` (or full Bat if compose)
- [ ] Ordinary `0xB07A` without warp / item grants (compose)
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
