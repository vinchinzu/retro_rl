# Agent Instructions — magical_quest

Scripted SNES completion agent for **The Magical Quest Starring Mickey Mouse** (platforming track; maturity M2).

## Identity

| Field | Value |
|-------|-------|
| Status | boot verified; controllable `Stage1.state` |
| Integration | `MagicalQuest-Snes` |
| Shared ROM zip | `roms/Super Nintendo/Magical Quest starring Mickey Mouse, The.zip` |
| Local ROM | `magical_quest/roms/` (via `scripts/setup_rom.py`) |

## Purpose

Build a reliable segment/policy stack for The Magical Quest Starring Mickey Mouse: discover useful RAM,
script controllers from development save states, then later stitch segments into
longer continuous runs. Full title-to-credits continuous evaluation is a later
goal, not the first milestone.

## Next milestone

Enter the 1-1 house door (wizard room). RIGHT from `Stage1.state` already
pins Mickey on the door (~788 frames, HP>0). UP has not opened it yet.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- Reuse `retro_harness/` (scripted completion) primitives; do not fork shared helpers without need.
- Line length 88; type hints; `uv run --frozen pytest` for tests.

## Scripts

- `scripts/boot_probe.py` — reset → one-player/default Mickey → Stage 1
- `scripts/ram_probe.py` — controlled LEFT/RIGHT progress deltas
- `scripts/setup_rom.py` — extract/link the shared ROM
- `python -m magical_quest.policy` — RIGHT from `Stage1.state` to the 1-1 door

## RAM quick ref

Player/world X `0x0024`, screen Y `0x0027`, horizontal progress `0x002A`,
gameplay-active `0x02C0` (`1`), current hearts `0x02B1`, max hearts `0x02B0`,
lives `0x0372`. First-door stop: `player_x >= 374` and `player_y >= 36` and
`health > 0` (house wall, ~788 frames of RIGHT). Velocity/grounded, room id
after entering the house, and enemies remain open.
