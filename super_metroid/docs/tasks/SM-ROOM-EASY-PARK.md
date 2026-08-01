# TASK SM-ROOM-EASY-PARK: Park Crab Hole + Grapple Tutorial residuals

## Recipe step
docs

## Model
Flash

## Wave type
implement

## Own files only
- `docs/tasks/SM-ROOM-EASY-PARK-note.md` (create)
- `docs/tasks/QUEUE.md` (Crab + Grapple rows only → parked)

## Context
- EASY-02→02C: still wrong exit `0xCF80` after span extend + UP bias.
- EASY-03→R3: same pin x=21 pose-138 after land_shoot + door-open knobs → PLANNER-GATE.
- Stop one-knob spam; park both for planner rewrite later.

## Do
1. Park note with both pin tables.
2. QUEUE rows → parked; next = planner rewrite placeholders only in note.
3. No policy edits.

## Verify
```bash
test -f super_metroid/docs/tasks/SM-ROOM-EASY-PARK-note.md
rg -n "EASY-02|EASY-03|park" super_metroid/docs/tasks/QUEUE.md | head -20
```
