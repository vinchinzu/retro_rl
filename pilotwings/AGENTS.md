# Agent Instructions — pilotwings

Scripted SNES completion agent for **Pilotwings** (continuous control track; maturity M2).

## Identity

| Field | Value |
|-------|-------|
| Status | boot verified |
| Integration | `Pilotwings-Snes` |
| Shared ROM zip | `roms/Super Nintendo/Pilotwings.zip` |
| Local ROM | `pilotwings/roms/` (via `scripts/setup_rom.py`) |

## Purpose

Build a reliable segment/policy stack for Pilotwings: discover useful RAM,
script controllers from development save states, then later stitch segments into
longer continuous runs. Full title-to-credits continuous evaluation is a later
goal, not the first milestone.

## Next milestone

Complete the Lesson 1 light-plane objective from `Lesson1Plane.state`.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- Reuse `snes_oneshot/` primitives; do not fork shared helpers without need.
- Line length 88; type hints; `uv run --frozen pytest` for tests.
