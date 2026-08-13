# Maridia / Grapple human tape — hops + anti-desync

**Tape:** `tasks/maridia_grapple_human.json` (**44,039f**, ~12.2 min, assist ON)  
**Start:** `scratch/post_gravity_caterpillar.state` Caterpillar `0xA322` ~(70,1419) items `0x3125`  
**Trace end:** Main Street `0xCFC9` ~(317,1963) pose 2 items **`0x7125`** (Grapple+Gravity; beams `0x1007`)  
**Extract:** `tasks/maridia_grapple_human_extract.json` (offline hops; no full replay)

## Policy

- Recording is a **shape guide** for pure skills — not gold open-loop to paste.
- **Never** open-loop replay multi-minute tapes to “recover” pins. Assist-on
  replay of this tape desynced at **f29057** in Grapple Tutorial 3 return.
- **Live anchors** (guided_human default): room enter + item delta + F6 manual
  → `tasks/<name>_anchors/` + `*_anchors.json`. Prefer those over end-only F5.
- Original F5 Main Street end pin was **lost** to a desynced overwrite. Trace
  fingerprint of the intended end is still authoritative for *where* the run
  finished; binary state must be re-locked with anchors.

## Recovered pins (assist-sync while room matched trace)

| ID | Path | Room | Items | Frame | Use |
|----|------|------|-------|------:|-----|
| post-Croc farm | `scratch/post_crocomire_farming_human.state` | `0xAA82` ~(19,139) | `0x3125` | 16757 | pure post-Croc / Grapple approach |
| post-Grapple | `scratch/post_grapple_beam_human.state` | `0xAC2B`→`0xAC00` | **`0x7125`** | ~24720 collect / 25954 leave | **next-phase start** `--from post-grapple` |

Grapple first seen ~**f24720** in `0xAC2B` (items `0x3125`→`0x7125`).

## Skill groups (from trace hops)

| Skill id | Frames | Notes |
|----------|--------|-------|
| `caterpillar_to_red` | 0–4622 | Hellway + Red Tower |
| `red_to_glass` | 2467–7490 | Bat→Below Spazer→West→Glass (tube) |
| `glass_to_business` | 6146–8973 | East→Warehouse→Business |
| `business_to_crocomire` | 8160–16756 | Gate→Crumble→Speedway→**Croc fight ~5.2k** |
| `croc_to_grapple` | 11541–25954 | Farm/save/shaft/Cosine/PreGrapple/**collect** |
| `grapple_tutorials_return` | (return leg) | Tutorial 1→2→3 swings |
| `grapple_return_business` | …→37884 | Croc Escape → Business |
| `business_meander` | 37885–41035 | **EXCLUDE** thrash: Frog bounce + Cathedral peek |
| `business_to_main_street` | …→44038 | elev→East→Glass→**Main Street** |

Full door list: 43 hops in the extract JSON.

## Main Street re-lock (2026-08-10) — **GREEN pin**

| Field | Value |
|-------|-------|
| Task | `tasks/maridia_main_street_human.json` (**14170f**, assist on, anchors on) |
| Start | `post_grapple_beam_human` Tutorial 1 `0xAC00` ~(236,121) items `0x7125` |
| End | Main Street `0xCFC9` ~(391,1979) pose 2 items `0x7125` beams `0x1007` |
| Canonical pin | `scratch/post_grapple_main_street.state` |
| End verify | fingerprint matches last trace row |
| F6 mid | Croc Escape `0xAA0E` ~(211,139) → `scratch/post_grapple_croc_escape_human.state` |
| Extract | `tasks/maridia_main_street_human_extract.json` / `*_tail.json` |

### Return hops (this take — 13 rooms)

| Hop | Frames | Room |
|-----|--------|------|
| Tutorials 1→2→3 | 0–3752 | `0xAC00`→`0xABD2`→`0xAB64` |
| Shaft→Farm→Croc→Speedway | 3753–6166 | `0xAB07`…`0xA923` |
| **Croc Escape** | 6167–8523 | `0xAA0E` (F6 @ f8035) |
| Business (long) | 8524–12816 | `0xA7DE` ~4293f |
| elev→East→Glass→**Main Street** | 12817–14169 | → `0xCFC9` |

Living product seam is now `scratch/full_start_v1_main_street.state`
(F5 of `./play grapple` → Main Street). `./play main-street` continues from
that pin; the standalone `post_grapple_main_street` take stays on disk.

```bash
# Next: Maridia deeper from locked Main Street pin
./snes/super_metroid/play main-street full_start_v1
# or a named standalone take:
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from main-street --name maridia_botwoon_path_human --no-guide

uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \
  snes/super_metroid/tasks/maridia_main_street_human.json --summary
```

**Anchor note:** this take only stored boot + F6 + end (room_enter was a bug —
`_last_room` updated during door transition). Fixed in `human_tape.py`; future
takes dump every room enter on first ordinary frame.

## Recording contract (anti-desync)

| Artifact | Role |
|----------|------|
| `tasks/<name>.json` | frames + trace (+ items/beams per row) |
| `tasks/<name>_end.state` | gzip end pin + `metadata.end_fingerprint` |
| `tasks/<name>_anchors/` | gzip dump per room enter / item delta / F6 |
| `tasks/<name>_anchors.json` | index of anchors (frame, room, xy, items, path) |

`extract_human_tape.py` verifies end fingerprint vs last trace row when present.
