# Agent Instructions — rival_turf

Scripted SNES completion agent for **Rival Turf!** (linear combat track; maturity M2).

## Identity

| Field | Value |
|-------|-------|
| Status | boot verified; fight-ready `Stage1.state` |
| Integration | `RivalTurf-Snes` |
| Shared ROM zip | `roms/Super Nintendo/Rival Turf!.zip` |
| Local ROM | `rival_turf/roms/` (via `scripts/setup_rom.py`) |

## Purpose

Build a reliable segment/policy stack for Rival Turf!: discover useful RAM,
script controllers from development save states, then later stitch segments into
longer continuous runs. Full title-to-credits continuous evaluation is a later
goal, not the first milestone.

## Next milestone

Clear the opening two-enemy street lock from `Stage1.state`, then detect the
camera/progress unlock.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- Reuse `snes_oneshot/` primitives; do not fork shared helpers without need.
- Line length 88; type hints; `uv run --frozen pytest` for tests.

## Scripts

- `scripts/boot_probe.py` — reset → one-player/Jack → opening combat lock
- `scripts/ram_probe.py` — controlled X/Y action deltas from `Stage1`
- `scripts/setup_rom.py` — extract/link the shared ROM

## RAM quick ref

Run state `0x00AB` (`1` active), player actor `0x0200`, player X
`0x0202`, player Y `0x0205`. Enemy slots, HP, camera, and stage are still open.
