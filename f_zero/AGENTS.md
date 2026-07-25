# Agent Instructions — f_zero

Scripted SNES completion agent for **F-Zero** (continuous control track; maturity M2).

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

From `MuteCity.state`, complete one lap without crashing out (centerline
follow + basic recovery).

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- Reuse `snes_oneshot/` primitives; do not fork shared helpers without need.
- Line length 88; type hints; `uv run --frozen pytest` for tests.

## Scripts

- `scripts/boot_probe.py` — reset → Grand Prix/Blue Falcon/beginner/Mute City
- `scripts/ram_probe.py` — acceleration and LEFT/RIGHT differential probe
- `scripts/setup_rom.py` — extract/link the shared ROM

## RAM quick ref

Race state `0x0046` (`1` live), track state `0x0047` (`1` live), raw speed
word `0x0002` (calibrate only after countdown), lateral `0x007F`, fine lateral
`0x00A6`. Lap, rank, energy, heading, and collision state remain open.
