# Agent Instructions — f_zero

Scripted SNES completion agent for **F-Zero** (continuous control track; maturity M2, M3 in progress).

## Identity

| Field | Value |
|-------|-------|
| Status | boot verified; Mute City race-start checkpoint |
| Integration | `FZero-Snes` |
| Shared ROM zip | `roms/Super Nintendo/F-Zero.zip` |
| Local ROM | `f_zero/roms/` (via `scripts/setup_rom.py`) |

## Purpose

Build a reliable segment/policy stack for F-Zero: discover useful RAM,
script controllers from development save states, then later stitch segments into
longer continuous runs. Full title-to-credits continuous evaluation is a later
goal, not the first milestone.

## Next milestone

Sticky lap index and rank RAM; then a full Mute City race (5 laps) or
natural-entry from boot rather than `MuteCity.state`.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- Reuse `retro_harness/` (scripted completion) primitives; do not fork shared helpers without need.
- Line length 88; type hints; `uv run --frozen pytest` for tests.

## Scripts

- `scripts/boot_probe.py` — reset → Grand Prix/Blue Falcon/beginner/Mute City
- `scripts/ram_probe.py` — acceleration and LEFT/RIGHT differential probe
- `scripts/run_mute_city_lap.py` — centerline + recovery from `MuteCity.state`
- `scripts/setup_rom.py` — extract/link the shared ROM

## RAM quick ref

Raw speed `0x0002` (calibrate only after countdown), lateral `0x007F`, camera
Y `0x00A6`. Finish-line: HUD `0x00B8` bit 4 (`X laps left`) rising edge — not
a sticky lap counter. Crash-out: `0x00C3` bit 6 exploded / bit 7 lost, or
signed power `0x00C9` `< 0`. Heading Angle8 `0x0BD1` vs checkpoint facing
`0x00C5`. Rank remains open.
