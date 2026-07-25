# Agent Instructions — battle_clash

Scripted SNES completion agent for **Battle Clash** (oneshot ladder #9, tier 2).

## Identity

| Field | Value |
|-------|-------|
| Status | input blocked |
| Integration | `BattleClash-Snes` |
| Shared ROM zip | `roms/Super Nintendo/Battle Clash.zip` |
| Local ROM | `battle_clash/roms/` (via `scripts/setup_rom.py`) |

## Purpose

Build a reliable segment/policy stack for Battle Clash: discover useful RAM,
script controllers from development save states, then later stitch segments into
longer continuous runs. Full title-to-credits continuous evaluation is a later
goal, not the first milestone.

## Next milestone

Add Super Scope cursor/trigger injection to the emulator bridge, then reach the
first boss.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- Reuse `snes_oneshot/` primitives; do not fork shared helpers without need.
- Line length 88; type hints; `uv run --frozen pytest` for tests.
