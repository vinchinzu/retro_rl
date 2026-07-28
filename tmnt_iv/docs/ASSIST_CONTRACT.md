# Assist Contract — TMNT IV

Runtime observation: **Bronze** (game-specific read-only RAM permitted).  
Intervention class: **Resource-assisted + Protection-assisted** (whole-run).  
Clean track (Stage 1 done; rest in progress): **Clean** = both assists at 0.

See **`docs/CLEAN_PLAYBOOK.md`** for permanent play rules when removing
assists stage-by-stage.

## Allowed writes (production low-assist)

| Assist | Trigger | Write | Notes |
|--------|---------|-------|-------|
| Emergency HP | HP ≤ 16 | restore HP to 80 | Fixed contract value; above Raphael's natural 48 HP; counted per intervention |
| Form-2 iframe hold | Super Shredder form 2 | hold iframe timer at 1 | Counted per frame; demutation bypass |

## Forbidden writes

- Stage / progress / boss flags
- Lives grants except natural pickups
- Inventory or character unlocks
- Mid-run save-state loads

## Natural (Clean-compatible) heals

Ground pizza boxes (`char 0x30`, see `ram_map.md`) fully restore HP when
picked up with controller input. Collecting pizza is **not** an assist.

**Clean** = survive on pizza + better play with:

- `emergency_hp` interventions = **0**
- form-2 iframe guard frames = **0**
- no A-special

Stage 1 Clean suite is verified pizza-only
(`scripts/probe_stage1_clean.py --suite`). Later stages keep emergency
until their own heal=none multi-entry suite is green; then drop assists
for that stage / whole run per playbook order.

## Reporting

Every continuous clear manifest must include intervention counts for HP
restores and iframe-guard frames. Do not label assisted runs as Clean.
