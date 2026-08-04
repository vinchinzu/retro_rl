# Agent Instructions — super_double_dragon

Scripted SNES completion agent for **Super Double Dragon** (linear combat track; maturity M3).

## Identity

| Field | Value |
|-------|-------|
| Status | Missions 1–2 complete; Mission 3/5 transition work |
| Integration | `SuperDoubleDragon-Snes` |
| Shared ROM zip | `roms/Super Nintendo/Super Double Dragon.zip` |
| Local ROM | `super_double_dragon/roms/` (via `scripts/setup_rom.py`) |

## Purpose

Build a reliable segment/policy stack for Super Double Dragon: discover useful RAM,
script controllers from development save states, then later stitch segments into
longer continuous runs. Full title-to-credits continuous evaluation is a later
goal, not the first milestone.

## Next milestone

Recover the natural Mission 3 gym stair handoff (`0x19 -> 0x1A`), then
complete the Chin brothers and replace the current Stage 4 transition clone.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- Reuse `retro_harness/` (scripted completion) primitives; do not fork shared helpers without need.
- Line length 88; type hints; `uv run --frozen pytest` for tests.
