# Post–Main Hall human — Screw → Ridley → Landing Site

**Tape:** `tasks/post-main-hall.json` (**121,220f**, assist ON, anchors ON ×73)  
**Start:** `scratch/post_ln_main_hall.state` Main Hall `0xB236` ~(1152,648) items `0x7327` beams `0x100F`  
**End:** Landing Site `0x91F8` ~(1152,1088) p155 items **`0x732F`** beams **`0x100F`**  
**Extract / end:** `tasks/post-main-hall_extract.json`, `tasks/post-main-hall_end.state`  
**Boss bit:** Norfair `boss_bits[2]` **6→7** at Ridley Tank (Ridley defeated)

## Policy

- Shape guide only (long take + F5 checkpoints; thrash in Worst Room / escape maze).
- Live anchors are authoritative; do not open-loop full replay.
- Next product targets: **G4 statues → Tourian → Mother Brain** from Landing Site.

## Pins (boot-verified)

| ID | Path | Room | Loadout | CLI |
|----|------|------|---------|-----|
| **post_bosses_landing_site** | `scratch/post_bosses_landing_site.state` | `0x91F8` ~(1152,1088) p155 | items `0x732F` all 4 bosses | **`--from post-bosses`** |
| post_screw_attack | `scratch/post_screw_attack.state` | `0xB6C1` ~(171,667) p137 | items **`0x732F`** (Screw) | `--from post-screw` |
| post_ridley_tank | `scratch/post_ridley_tank.state` | `0xB698` ~(216,143) | items `0x732F` Ridley bit | `--from post-ridley` |
| post_ridley_farming | `scratch/post_ridley_farming.state` | `0xB37A` ~(50,142) | after leave Ridley | `--from post-ridley-farming` |

Primary next-segment start: **post_bosses_landing_site** (ship, ordinary gs=8).

Collects this take:

- **Screw Attack** f10857 `0x7327`→`0x732F` (room `0xB6C1`)
- **Ridley** defeated during hop `0xB32E` (Norfair boss bit; tank enter f68857)

## Path shape (71 room hops — high level)

| Leg | Rooms | Notes |
|-----|-------|-------|
| GT / Screw | Main Hall→Acid Statue→GT→**Screw** | collect f10857; GT thrash before/after |
| LN approach | Fast Ripper→Pillars→**Worst Room**×2→Mickey | long Worst Room dwells |
| Amphitheatre → pirates | Amph→Kihunter shaft→Fireflea→Wasteland→**Metal Pirates** | long pirate room |
| **Ridley** | Plowerhouse→Farming→**Ridley**→Tank→exit | fight ~f60896–68705 |
| LN escape | Wasteland reverse→Fireflea maze / Springball / Escape PB thrash | multi-loop |
| Norfair exit | Musketeers→Single Chamber→Bubble→Farm→Speedway→Frog | |
| Brinstar return | Business→Warehouse→tunnels→Bat→**Red Tower** thrash | GHZ detour |
| Crateria | Hellway→Caterpillar elev→Kihunter→Tube→**Landing Site** | F5 end at ship |

## Next

```bash
# G4 statues / Tourian free-record from post-boss Landing Site
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from post-bosses --name g4_tourian_human --no-guide

# Mid-route resumes
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from post-screw --name ln_post_screw_human --no-guide
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from post-ridley --name ln_post_ridley_exit_human --no-guide

uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \
  snes/super_metroid/tasks/post-main-hall.json --summary
```
