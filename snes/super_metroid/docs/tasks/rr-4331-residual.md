## Residual — rr-4331 Ceres tail-tank spine + elev re-solve

### Result
PARTIAL

Takeoff windows are a **shared room type** (`super_metroid.takeoff` only —
not re-exported from `rooms/` or `ceres`). Knockback is
`skills.knockback.is_knockback`. D-pad `LEFT`/`RIGHT` and shoulder `L`/`R`
are different buttons (`SNES_DPAD_*` / `SNES_SHOULDER_*`).

Product-path bugs from the prior review stay closed in unit tests: inbound
gs 9/11 is not leave; failed tail-tank raises; shaft hands off only at the
s10 seat; pad walk uses `_CERES_ELEV_SHIP_X` until gs 32.

Live climb from `ceres_elev_enter.state` still does **not** leave (no gs 32
/ Landing). Not re-probed this turn.

KPDR **Ceres Station** clock starts at first elev ordinary control
(gs=8), not title. Goal **1:35** is that split, any debris seed.

### Files changed
- `takeoff.py` — hop types + `approach_window` / `shoulder_pump_button`
- `retro_harness/controls.py` — `SNES_DPAD_*` / `SNES_SHOULDER_*` names
- `ram.py` — `GS_ORDINARY` / `GS_DEAD` / `GS_CERES_LEAVE`
- `routes/kpdr/ceres/` — no local KB/latch clones; slim package `__init__`
- `combat/ceres_ridley.py` — `POSE_KNOCKBACK` + `GS_DEAD`
- `docs/ARCHITECTURE.md`, `AGENTS.md`

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_takeoff.py \
  snes/super_metroid/tests/test_ceres_elev_climb.py \
  snes/super_metroid/tests/test_ceres_ridley_combat.py \
  snes/super_metroid/tests/test_ceres_arm_pump.py \
  snes/super_metroid/tests/test_door_kinematics.py \
  retro_harness/tests/test_controls.py -q
```

No elev probe this turn. Last live bench (unchanged):

| Policy | Frames | Seconds | Clock | End |
|--------|-------:|--------:|-------|-----|
| kinematic windows | 1,506 | 25.059 | 00:25.10 | timeout best_y=524, no 475 land |

Ceres Station (first elev control → Landing), product `morph.json`:

| Split | Frames | Seconds | Clock |
|-------|-------:|--------:|-------|
| product (wait Ridley) | 10,688 | 177.840 | 02:58.13 |
| goal | 5,709 | 94.994 | 01:35.15 |
| still need | 4,979 | 82.847 | 01:22.98 |

### Acceptance
- [x] Tail-tank on the spine (no wait flag)
- [x] Same elev enter pin after tail-tank + reverse
- [x] Pin 571 / 475 / 363 seats from that pin
- [x] Takeoff is kinematic (not N-frame recipes / not hillclimb)
- [x] Takeoff type is shared (`takeoff.PlatformHop`), not Ceres-only
- [x] D-pad LEFT/RIGHT ≠ shoulder L/R
- [x] Inbound gs 9/11 is not leave; failed tail-tank raises; ship pad walks x
- [ ] Elev leave (gs 32 / Landing) from the pin
- [ ] Ceres Station 1:35 from first control, any seed
- [ ] `--to morph` re-record (do not overwrite `recordings/morph.json` red)

### Residual risks
- Live climb still short-hops off 571 (peak y=524) and sails off the ledge
- Shared window is the contract; the 571→475 numbers still miss the land
- Window GA (`ceres_elev_ga.py`) not run this session

### Next action (required)
- **Next card ID:** rr-4331 (same) — finish elev leave from the pin
- **One change:** land 475 from the 571 seat with a real takeoff window
  (momentum + x_sub + L/R), then 363, then ship. Do not use hillclimb.
  Tune `CERES_ELEV_HOPS` (now `PlatformHop` data), not a new hop type.
- **Source state:** `custom_integrations/SuperMetroid-Snes/scratch/ceres_elev_enter.state`

### Non-claims
- Did not STATUS-promote / re-record `recordings/morph.json` / claim 1:35
- Did not keep wait as a product fallback
- Did not run a live elev probe or a long GA

### Probe pin (if pure/geometry)
room=0xDF45 pose=10 x=216 y=651 door_transition=0 last_pin=ceres_ridley_enter
pin 571 x=60 pose 10; 475 x=130 pose 166; 363 x=180 pose 166 (open-loop
sweep, not the live controller)
