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

## Reporting

Every continuous clear manifest must include intervention counts for HP
restores and iframe-guard frames. Do not label assisted runs as Clean.
