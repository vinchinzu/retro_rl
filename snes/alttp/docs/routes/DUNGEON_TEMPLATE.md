# Dungeon route — <name>

Copy this file for one dungeon. Fill every heading. Do not write STATUS from it. A save-state pin is not continuous. Facts come from `z3-json-data` or a live scan, not from a hint line.

## Entry

| Field | Value |
|-------|--------|
| Pin | |
| Room | |
| First hop | |

## Requirements

Items the route needs, taken from z3 logic.

| Item | Why | z3 source |
|------|-----|-----------|
| | | |

## Room table

One row per room the route enters. Geometry stays in `alttp/maps/` json. Do not copy approach xy into Python.

| Room | Doors + types | Chests | Sprites | Notes |
|------|---------------|--------|---------|-------|
| | | | | |

## Keys and big key

Where each small key is taken, and where the big key is taken. Do not poke keys on a quoted run.

## Big chest

Room, contents, and what opens it.

## Mechanics

Puzzles this dungeon uses (shutters, push blocks, pits, layers, followers). Fight only for a shutter, a push block, or an item the route takes.

## Boss

| Field | Value |
|-------|--------|
| Sprite ids | |
| HP byte | |
| Phase byte | |
| What hurts it | |
| When | |

## Prize and exit

Prize (pendant, crystal, or other) and the leave back to the overworld. Leave proof is a RAM glance, not an MP4.

## Save points

Power-on save points only. Name the pin and its predecessor. A `*Entry` pin is development only.

| Save point | Predecessor | What it proves |
|------------|-------------|----------------|
| | | |

## Open reds

What is still red. Do not weaken a check to close a row.

| Red | Evidence | Blocker |
|-----|----------|---------|
| | | |
