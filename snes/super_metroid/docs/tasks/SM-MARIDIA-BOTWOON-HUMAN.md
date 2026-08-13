# Maridia Botwoon path human — Main Street → Draygon → Space Jump

**Tape:** `tasks/maridia_botwoon_path_human.json` (**58,670f**, assist ON, anchors ON)  
**Start:** `scratch/post_grapple_main_street.state` Main Street `0xCFC9` ~(391,1979) items `0x7125`  
**End:** Precious `0xD78F` ~(55,651) items **`0x7325`** (Space Jump + Grapple + Gravity)  
**Extract / tail:** `tasks/maridia_botwoon_path_human_extract.json`, `*_tail.json`  
**Anchors:** 23 live dumps under `tasks/maridia_botwoon_path_human_anchors/`

## Policy

- Shape guide only — grapple / Maridia movement was sloppy; do not paste RLE.
- Checkpoint reloads during the take renumbered some `frame` fields; hop
  **dwell uses trace indices** (fixed in `human_tape.build_room_hops`).
- Prefer **live anchors** over open-loop replay.

## Space Jump / next-segment pins (verified boot)

| ID | Path | Room | xy | Items | Frame | CLI |
|----|------|------|-----|-------|------:|-----|
| **post_space_jump** | `scratch/post_space_jump.state` | `0xD9AA` Space Jump | ~(85,155) p138 | **`0x7325`** | 52049 item_delta | **`--from post-space-jump`** |
| post_space_jump_precious | `scratch/post_space_jump_precious.state` | `0xD78F` Precious | ~(39–55,651) | `0x7325` | 53491 enter | `--from post-space-jump-precious` |
| post_draygon_precious | `scratch/post_draygon_precious.state` | `0xD78F` Precious | ~(55,651) p1 | `0x7325` | 58669 F5 | `--from post-draygon` |

Primary next-segment start: **post_space_jump** (SJ just collected, ordinary gs=8).

Also kept: Botwoon entry, pre-SJ room enter, Draygon entry (under `scratch/post_*_human.state`).

## Skill / hop shape (20 room visits)

| Segment | Rooms (approx) | Notes |
|---------|----------------|-------|
| Main Street climb | `0xCFC9` | long open |
| Everest thrash | `0xD0B9` | sloppy grapple — exclude gold |
| Crab / Pseudo / Bug / Watering | `0xD1A3`…`0xD13B` | backtrack loop |
| Aqueduct → Botwoon hall | `0xD5A7`→`0xD617` | |
| **Botwoon** | `0xD95E` | fight |
| Halfie / Colosseum | `0xD913`→`0xD72A` | long Colosseum dwell |
| Precious → **Draygon** | `0xD78F`→`0xDA60` | |
| **Space Jump** | `0xD9AA` | collect f52049 `0x7125`→`0x7325` |
| Return Precious | `0xDA60`→`0xD78F` | F5 end |

## Next

```bash
# Preferred: continue from SJ chozo room
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from post-space-jump --name post_sj_exit_human --no-guide

# Or from Precious post-Draygon F5 seat
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from post-draygon --name maridia_exit_human --no-guide

uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \
  snes/super_metroid/tasks/maridia_botwoon_path_human.json --summary
```
