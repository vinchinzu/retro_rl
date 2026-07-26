# Assist Contract — TMNT IV

Runtime observation: **Bronze** (game-specific read-only RAM permitted).  
Intervention class: **Resource-assisted + Protection-assisted**.

## Allowed writes

| Assist | Trigger | Write | Notes |
|--------|---------|-------|-------|
| Emergency HP | HP ≤ 16 | restore HP to 80 | Counted per intervention in manifest |
| Form-2 iframe hold | Super Shredder form 2 | hold iframe timer at 1 | Counted per frame; demutation bypass |

## Forbidden writes

- Stage / progress / boss flags
- Lives grants except natural pickups
- Inventory or character unlocks
- Mid-run save-state loads

## Natural (Clean-compatible) heals

Ground pizza boxes (`char 0x30`, see `ram_map.md`) fully restore HP when
picked up with controller input. Collecting pizza is **not** an assist.
The Clean-track goal is to survive on pizza + better play with
`emergency_hp` interventions at **0** and form-2 iframe guard at **0**.

## Reporting

Every continuous clear manifest must include intervention counts for HP
restores and iframe-guard frames. Do not label assisted runs as Clean.
