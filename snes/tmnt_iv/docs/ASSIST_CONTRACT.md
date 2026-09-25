# Assist contract: TMNT IV

Runtime observation: **Bronze** (game-specific read-only RAM permitted).
Intervention class: **Resource-assisted + Protection-assisted** (whole-run).
Clean means both assists at 0. Stage 1 pizza-only is verified. Later stages
are not. Play traps live in `AGENTS.md`.

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
(`scripts/probe_clean.py --stage 1 --suite`). Later stages keep emergency
HP until their own heal=none multi-entry suite is green.

## Reporting

Every continuous clear manifest must include intervention counts for HP
restores and iframe-guard frames. Do not label assisted runs as Clean.

## Clean mode (parallel track)

**Clean** means both emergency HP restore and form-2 iframe hold are **off**:
zero resource restores and zero protection writes. Observation may still be
Bronze (read-only RAM). Natural pizza pickup is not an assist.

Clean is a **parallel** privilege-reduction workstream; it does not replace
this assisted contract or the primary M8 continuous hard clear.

Artifact isolation: Clean runs use `*_clean` stems and must never
overwrite assisted `tmnt_iv_full_hard_*` baselines. Ready work:
`bd ready -l tmnt_iv`.

Hard constraints:

- Default continuous CLI remains resource + protection assisted.
- Clean runs must not overwrite assisted `tmnt_iv_full_hard_*` baselines.
- STATUS primary program gate stays assisted until an explicit program
  decision changes it; Clean results are documented as a secondary track.
