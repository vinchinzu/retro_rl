# Spazer human guide chunks (not gold)

**Source:** `tasks/spazer_from_charge_human.json` (Charge → full path, guide OFF)  
**Sliced:** `tasks/spazer_from_charge_chunks.json`

## Policy

- Recording is a **guide** for shape/timing — **not** a gold open-loop to replay verbatim.
- Prefer short learned primitives over pasting full RLE.
- Morph bomb = **X**.
- **Do not re-record the full Charge→climb path** just for TOP-MID. Use pure
  post-Spazer pins (below).

## Prefer pure pins over re-thrash

Door / collect / return are already pure-green. Refresh + human-record only
the residual drop:

```bash
# Refresh pure chain (no human)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure below-spazer-to-spazer \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/pre_spazer_door_with_charge.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_spazer_entry_pure.state
uv run python snes/super_metroid/scripts/probe/kpdr.py pure spazer-collect \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_spazer_entry_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_spazer_collect_pure.state
uv run python snes/super_metroid/scripts/probe/kpdr.py pure spazer-return-to-below \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_spazer_collect_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_spazer_return_pure.state

# Clean TOP-MID human only (start at return handoff ~(380,155), Spazer held)
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from post-spazer-return --route spazer-top-drop --name spazer_top_drop_human --no-guide
# F5 when mid y≥220 (or floor). Do NOT RIGHT into Super. No enemy thrash.
```

| Pin | Path | Use |
|-----|------|-----|
| post-collect | `scratch/post_spazer_collect_pure.state` | Spazer Room ~(171,171) beams `0x1004` |
| post-return | `scratch/post_spazer_return_pure.state` | Below top ~(380,155) — **TOP-MID source** |

## Chunks (climb/bomb — shape only; FLOOR-MID + CLIMB-LAND green)

| Chunk | Frames | Pin | Learn |
|-------|--------|-----|-------|
| `floor_to_mid` | 7095–7697 (603f) | (45,395)p1 → **(59,235)p1** | Standing mid seat **GREEN** |
| `mid_to_node4` | 7697–7794 (**98f**) | (59,235)p1 → ~(91,91) | WJ/crest jump **GREEN** |
| `node4_to_super_door` | 7794–8320 (527f) | node4 → Super enter | Morph-tunnel + Super door (**mainline** in `play_below_spazer_to_spazer`) |
| `spazer_collect_return_handoff` | 8321–9402 | Spazer → handoff ~(364,155) | Collect+return (pure green; prefer pure pins) |

## EXCLUDED (do not train / do not repeat)

| Segment | Frames | Reason |
|---------|--------|--------|
| `return_enemy_fall_and_floor_thrash` | **9403–11066** | Fall caught by enemy + messy floor recovery toward West |

TOP-MID must be a **clean** drop skill, not this thrash — record from
`post-spazer-return` only.

## Clean TOP-MID human (2026-08-05) — **GREEN pure**

| Field | Value |
|-------|-------|
| Task | `tasks/spazer_top_drop_human.json` (**1355f**, assist on, guide off) |
| Start | `post_spazer_return_pure` ~(380,155) beams `0x1004` |
| End human | West Tunnel `0xCF54` ~(39,139) |
| Pure | `spazer-top-to-west` **1281f** ×2 → `0xCF54` |
| Checkpoint | `scratch/post_spazer_west_pure.state` |
| Shape | spin-left top → morph crawl left (bombs=X) → left drop → floor → RIGHT West |

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure spazer-top-to-west \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_spazer_return_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_spazer_west_pure.state
```
