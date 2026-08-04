# Agent Instructions — star_fox

Scripted SNES completion agent for **Star Fox** (technical-showcase track).

## Identity

| Field | Value |
|-------|-------|
| Integration | `StarFox-Snes` |
| Shared ROM zip | `roms/Super Nintendo/Star Fox.zip` |
| ROM revision | USA Rev 2 |
| Runtime tier | Bronze (RAM-conditional, controller-only actions) |

## Working Rule

- Develop Route 1 one stage at a time from save states, then chain the route.
- Store save states under `custom_integrations/StarFox-Snes/`.
- Store recordings, screenshots, reports, and RAM probes under `star_fox/`.
- Keep Star Fox RAM maps and flight policy here; elevate only reusable helpers
  to `retro_harness/` (scripted completion).
- Headless probes use SDL dummy video/audio drivers.

## First Milestone

Boot USA Rev 2 from reset, select Route 1, and obtain a repeatable Corneria
clear while preserving at least one life.
