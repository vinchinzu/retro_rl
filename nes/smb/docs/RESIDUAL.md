# Residual observation

Same lattice idea as Super Metroid `R(τ)`, on SMB. The stepper is a search
model. Emulator replay is ground truth. This is not a route-clear claim.

## Observation map

| Name | RAM | Width | Lattice | Notes |
|------|-----|-------|---------|-------|
| `x` | `$006D+$0086` | u16 | Oπ | absolute pixel X |
| `y` | `$00CE` | u8 | Oπ | pixel Y; 1-1 floor about 176 |
| `pose` | `$000E` | u8 | Oπ | `0x08` controllable, including in the air |
| `room` | `$075F/$0760/$0750` | packed | Oπ | `(world<<16)\|(level<<8)\|area` |
| `sub_x` | `$0400` | u8 | Oσ | X position subpixel |
| `sub_y` | `$0416` | u8 | Oσ | Y position subpixel |
| `enemy0_active` | `$000F` | u8 | Oσ+ | slot 0 flag |
| `enemy0_type` | `$0016` | u8 | Oσ+ | slot 0 type |
| `energy` | `$075A` | u8 | O† | lives |
| `dead` | `$000E` / `$0770` / `y` | flag | O† | dying, game over, pit |
| `velocity_x` | `$0057` | s8 | field | first-diff only |
| `velocity_y` | `$009F` | s8 | field | first-diff only |
| `frame_counter` | `$0009` | u8 | lag | desynced tape index |
| `on_ground` | `$001D==0` | flag | field | air is `1` |
| `x_force` | `$0705` | u8 | stepper | `Player_X_MoveForce` |
| `running_speed` | `$0703` | u8 | stepper | `RunningSpeed`; no-L/R `$D0` if set |
| `y_move_force` | `$0433` | u8 | stepper | `Player_Y_MoveForce` |
| `vertical_force` | `$0709` | u8 | stepper | rising / current gravity |
| `vertical_force_down` | `$070A` | u8 | stepper | fall gravity |
| `jump_origin_y` | `$0708` | u8 | stepper | A-release height gate |

`R(τ) = (fd_σ+, fd_σ, fd_π, fd_†)`. `None` means that level held for the
horizon. Oπ holds: keep as a search model, not a route clear. Oσ broke and
Oπ holds: check the emulator. Room change or O†: reject. `$0009` diverge:
tag lag.

## Measure

Short Level1_1 tapes live in `smb.residual_harness.SEGMENTS` (idle, walk,
jump, run-jump, land, brake, LEFT, rejump).

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.measure_residual
uv run pytest nes/smb/tests/test_residual.py -q
```

Documented end state of that stepper (first live pass 2026-08-13, later
fixes in the same note, not re-run here):

- A-release uses `ImposeGravity`. Air X uses the walk tables unless
  `|vx| >= 0x19`.
- Landing keeps leftover `$0416` and `$0709`. Do not snap `sub_y` to 0.
  The next jump's InitJS wipes `$0416`.
- Takeoff-frame air X uses walk `$98` unless `|vx| >= 0x19`.
- InitJS jump bands follow `|vx|` at takeoff (`JumpMForceData` /
  `FallMForceData` / `PlayerYSpdData`). Swim indices are not modeled.
- No Left/Right uses `$98` unless `RunningSpeed` is latched or
  `|vx| >= $21` (then `$D0`).
- LEFT from rest subtracts to `$FED0`, not a sign-magnitude `-$0130`.
- Air walk-max keeps `x_force`. Clamping snaps `vx` only.

## Modules

- `smb.observation`: RAM to `Observation`, `PlayerPhysics`, `World`
- `smb.approx.step`: `player, action, world` to `player`
- `smb.residual`: `SMB_LATTICE` and `compute_residual_profile`
- `retro_harness.residual`: shared profile and lattice scan
- `smb.residual_harness`: stepper, fceumm, and `R(τ)`

## Next

Collision as a `World` query. Not a route gate.
