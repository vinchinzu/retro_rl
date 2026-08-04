# Agent Instructions — joe_and_mac

Scripted SNES completion agent for **Joe & Mac** (platforming track; maturity M2).

## Identity

| Field | Value |
|-------|-------|
| Status | boot verified |
| Integration | `JoeAndMac-Snes` |
| Shared ROM zip | `roms/Super Nintendo/Joe & Mac - Caveman Ninjas.zip` |
| Local ROM | `joe_and_mac/roms/` (via `scripts/setup_rom.py`) |

## Purpose

Build a reliable segment/policy stack for Joe & Mac: discover useful RAM,
script controllers from development save states, then later stitch segments into
longer continuous runs. Full title-to-credits continuous evaluation is a later
goal, not the first milestone.

## Next milestone

Clear the first traversable segment from `Stage1.state` (move right, jump
gaps, and attack the nearest threat).

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- Reuse `retro_harness/` (scripted completion) primitives; do not fork shared helpers without need.
- Line length 88; type hints; `uv run --frozen pytest` for tests.
