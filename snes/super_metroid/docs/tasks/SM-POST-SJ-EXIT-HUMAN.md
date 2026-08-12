# Post–Space Jump exit human — Spring Ball → Plasma → LN Main Hall

**Tape:** `tasks/post_sj_exit_human.json` (**80,368f**, assist ON, anchors ON ×62)  
**Start:** `scratch/post_space_jump.state` Space Jump `0xD9AA` items `0x7325`  
**End:** LN **Main Hall** `0xB236` ~(1152,648) items **`0x7327`** beams **`0x100F`**  
**Extract / tail:** `tasks/post_sj_exit_human_extract.json`, `*_tail.json`

## Policy

- Shape guide only (long take + checkpoint reloads).
- Live anchors are authoritative; do not open-loop full replay.
- Next product targets: Ridley / Golden Torizo path from Main Hall.

## Pins (boot-verified)

| ID | Path | Room | Loadout | CLI |
|----|------|------|---------|-----|
| **post_ln_main_hall** | `scratch/post_ln_main_hall.state` | `0xB236` ~(1152,648) p155 | items `0x7327` beams `0x100F` | **`--from main-hall`** |
| post_ln_elevator_save | `scratch/post_ln_elevator_save.state` | `0xB1BB` ~(200,139) | same | `--from ln-elev-save` |
| post_spring_ball | `scratch/post_spring_ball.state` | `0xD6D0` ~(379,362) | items `0x7327` (Spring) | `--from post-spring-ball` |

Collects this take:

- **Spring Ball** f37796 `0x7325`→`0x7327` (room `0xD6D0`)
- **Plasma Beam** beams `0x1007`→`0x100F` (Plasma room detour; not item bit)

## Path shape (60 hops — high level)

| Leg | Rooms | Notes |
|-----|-------|-------|
| Leave Maridia east | SJ→Draygon→Precious→Colosseum→Cactus | restarts after checkpoint |
| Plasma detour | Plasma Spark→Kassiuz→Plasma→back | beams `0x100F` |
| Spring Ball | Pants→Shaktool→`0xD6D0` | spring collect |
| Maridia exit | sand halls→Crab Hole→East/Glass→Warehouse | |
| Norfair approach | Business→Frog→Speedway→Bubble→Kronic→**Lava Dive** | |
| LN entry | LN elev → save → **Main Hall** | F5 end |

## Next

```bash
# DONE: Ridley + Screw → Landing Site (see SM-POST-MAIN-HALL-HUMAN.md)
# Continue from post-boss Landing Site → G4 / Tourian
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from post-bosses --name g4_tourian_human --no-guide

uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \
  snes/super_metroid/tasks/post-main-hall.json --summary
```
