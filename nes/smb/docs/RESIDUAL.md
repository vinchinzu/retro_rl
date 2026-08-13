# Residual observation — Super Mario Bros. (NES)

Same practical lattice as Super Metroid `R(τ)`, applied to the simplest
platformer already in the repo. The approximate stepper is a **search** model,
not ground truth. Emulator replay is authoritative.

## Observation map

| Name | RAM | Width | Lattice | Notes |
|------|-----|-------|---------|-------|
| `x` | `$006D+$0086` | u16 | Oπ | absolute pixel X |
| `y` | `$00CE` | u8 | Oπ | pixel Y; 1-1 floor ≈ 176 |
| `pose` | `$000E` | u8 | Oπ | `0x08` controllable; stays `0x08` in air |
| `room` | `$075F/$0760/$0750` | packed | Oπ | `(world<<16)\|(level<<8)\|area` |
| `sub_x` | `$0400` | u8 | Oσ | X **position** subpixel |
| `sub_y` | `$0416` | u8 | Oσ | Y **position** subpixel |
| `enemy0_active` | `$000F` | u8 | Oσ+ | slot 0 flag |
| `enemy0_type` | `$0016` | u8 | Oσ+ | slot 0 type |
| `energy` | `$075A` | u8 | O† | lives |
| `dead` | `$000E` / `$0770` / `y` | flag | O† | dying, game over, pit |
| `velocity_x` | `$0057` | s8 | field | first-diff only |
| `velocity_y` | `$009F` | s8 | field | first-diff only |
| `frame_counter` | `$0009` | u8 | lag | desynced tape index |
| `on_ground` | `$001D==0` | flag | field | air is `1` |
| `x_force` | `$0705` | u8 | stepper | `Player_X_MoveForce` |
| `y_move_force` | `$0433` | u8 | stepper | `Player_Y_MoveForce` |
| `vertical_force` | `$0709` | u8 | stepper | rising / current gravity |
| `vertical_force_down` | `$070A` | u8 | stepper | fall gravity |
| `jump_origin_y` | `$0708` | u8 | stepper | A-release height gate |

`R(τ) = (fd_σ+, fd_σ, fd_π, fd_†)`. `None` means that level held for the horizon.

Planner: Oπ holds → keep as search model (not route-clear). Oσ broke / Oπ holds
→ emu spot-check. Room or O† → hard-reject. `$0009` diverge → tag `lag`.

## First measurement

Short Level1_1 tapes in `smb.residual_harness.SEGMENTS`:

| Segment | Input | Why |
|---------|-------|-----|
| `idle` | 24 none | control: physics should hold |
| `walk` | 24 RIGHT | grounded accel + subpixel |
| `jump` | 4 A + 20 idle | takeoff / gravity (lands at f25) |
| `run_jump` | 30 RIGHT+B+A | air control + run accel |
| `jump_to_land` | 4 A + 28 idle | standing jump through land + settle |
| `run_jump_to_land` | 60 RIGHT+B+A | run-jump through land |
| `run_then_jump` | 16 RIGHT+B + 4 A + 16 RIGHT+B | takeoff-frame air X |

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.measure_residual
uv run pytest nes/smb/tests/test_residual.py -q
```

First live Level1_1 pass (2026-08-13, fceumm `Level1_1.state`):

| Segment | Horizon | `R(τ)=(fdσ+, fdσ, fdπ, fd†)` | First field | Cause |
|---------|--------:|------------------------------|-------------|-------|
| idle | 25 | `(—, —, —, —)` | — | — |
| walk | 25 | `(—, —, —, —)` | — | — |
| jump | 25 | `(5, 5, 7, —)` | subpixels | collision (A-release gravity) |
| run_jump | 31 | `(2, 2, 6, —)` | subpixels | collision (ground accel used in air) |

Walk holds Oπ and Oσ for the whole 24-input tape (x 40→51, xs=14). A longer
RIGHT hold still matches pixels/subpixels out to 80f; at 120f only Oσ+ breaks
(`enemy0` spawn, fdσ+=96) — the physics residual still holds. Jump first
broke `$0416` at f5 (`sub_y` 48 vs 64) then pixels at f7. Run-jump used
grounded run accel in air.

After `rr-ep6l` (A-release `ImposeGravity` + air walk tables unless `|vx|≥0x19`):

| Segment | Horizon | `R(τ)=(fdσ+, fdσ, fdπ, fd†)` | First field | Cause |
|---------|--------:|------------------------------|-------------|-------|
| idle | 25 | `(—, —, —, —)` | — | — |
| walk | 25 | `(—, —, —, —)` | — | — |
| jump | 25 | `(—, —, —, —)` | — | — |
| run_jump | 31 | `(—, —, —, —)` | — | — |

Gravity-only (before air X) already moved jump to a full hold; run-jump stayed
`(2, 2, 6, —)` on `$0400` until air X used walk `$98`/`$18`. Grounded walk,
grounded run (30f), and a short walk-jump also hold.

After `rr-phwv` (land keeps `$0416` / leftover `$0709`; do not snap `sub_y` to 0):

| Segment | Horizon | `R(τ)=(fdσ+, fdσ, fdπ, fd†)` | First field | Cause |
|---------|--------:|------------------------------|-------------|-------|
| idle | 25 | `(—, —, —, —)` | — | — |
| walk | 25 | `(—, —, —, —)` | — | — |
| jump | 25 | `(—, —, —, —)` | — | — |
| run_jump | 31 | `(—, —, —, —)` | — | — |
| jump_to_land | 33 | `(—, —, —, —)` | — | — |
| run_jump_to_land | 61 | `(—, —, —, —)` | — | — |

Landing snaps pixel Y and zeros `velocity_y` / `$0433`. ImposeGravity leftover
`$0416` stays (`128` on the standing 4-A land, `64` on the 60f run-jump) and
`$0709` stays at `0x70`. Land-then-walk / land-then-run / land-then-rejump
also hold. Short tapes still land before f25 / f53, so they never saw this.

After `rr-kez8` (takeoff-frame air X: leave-ground uses walk `$98` unless
`|vx|≥0x19`):

| Segment | Horizon | `R(τ)=(fdσ+, fdσ, fdπ, fd†)` | First field | Cause |
|---------|--------:|------------------------------|-------------|-------|
| idle | 25 | `(—, —, —, —)` | — | — |
| walk | 25 | `(—, —, —, —)` | — | — |
| jump | 25 | `(—, —, —, —)` | — | — |
| run_jump | 31 | `(—, —, —, —)` | — | — |
| jump_to_land | 33 | `(—, —, —, —)` | — | — |
| run_jump_to_land | 61 | `(—, —, —, —)` | — | — |
| run_then_jump | 37 | `(—, —, —, —)` | — | — |

16f RIGHT+B then A: takeoff `xf` 140→36 (walk `$98`), not 140→112 (run `$E4`).
Walk-then-jump and a skid-jump (RIGHT+B then LEFT+A) also hold.

Next break is **jump tables from `|vx|`**, not another air-X order stub. 24f
RIGHT+B then A: fdσ=26 — takeoff `vf` is `$1E` not `$20` (`sub_y` 32 vs 30).
32f run then A: fdπ=33, `vy=-5` vs `-4`, `vf=$28`. Sibling leftovers (not that
stub): air walk-max wipes `xf` (longer 16+4+40 tape fdσ=42 after land); LEFT
first-kick fdσ=1; brake after RIGHT uses `$98` not `FRICTION $D0` (fdσ=19).

## Modules

- `smb.observation` — RAM → structured obs
- `smb.approx.step` — pure `obs, action → obs` (flat ground + A-release + air X + land YMF + takeoff air X)
- `smb.residual` — `compute_residual_profile`
- `smb.residual_harness` — stepper + fceumm + `R(τ)`
