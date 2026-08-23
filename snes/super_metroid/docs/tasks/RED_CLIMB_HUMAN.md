# Red Tower climb multi-take (Ice → Moat human spine)

Human multi-attempt board for **K5 hop 12** Red `0xA253` → Hellway `0xA2F7`
(`rr-av5s`). Sibling legs (Ice→Warehouse, Alpha PB→Moat) are **separate tapes**
joined by **live anchors**, not open-loop button concat.

## Verified sibling takes (2026-08-10 session)

| Tape | Role | Notes |
|------|------|-------|
| `tasks/ice_to_red_human` | Ice → Warehouse elev | Clean prefix (~5.1k) |
| `tasks/warehouse_to_red_human` | Warehouse → Alpha PB | Red dwell ~6.1k **sloppy WJ** |
| `tasks/alpha_pb_to_moat_human` | Alpha PB → Moat | Clean (~5.2k) end `0x95FF` ~(60,152) |

Red enter live pin (warehouse chain):

```text
tasks/warehouse_to_red_human_anchors/f002012_enter_0xA253_0xA253.state
```

Pure dual pin (agent / bot-check):

```text
scratch/post_ice_bat_to_red_pure.state
```

## Splice model (arbitrary many Red attempts)

```text
[Ice → … → Red enter]     fixed (pure dual or human prefix)
        │
        ├─ take01 Red climb ─┐
        ├─ take02 Red climb ─┼─ rank → BEST ──► hop-replay from that take's boot
        ├─ takeNN Red climb ─┘
        │
[Hellway → Alpha PB → Moat]  fixed (warehouse suffix / alpha_pb_to_moat)
```

**Splice means:** pick the best Red take on the board + hop-replay each piece
from its **own** live `boot` / `room_enter` / F6 pin.

**Splice does not mean:** glue `frames[]` arrays end-to-end (enemy RNG +
subpixel desync; same class of bug as full-tape open-loop).

## Record clean Red climbs (multi-take)

### A. Pure Red bottom (default product pin)

```bash
# Reload same pin forever; F5 = save take + next; ESC = end series
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --segment red-to-hellway --series red_climb_v1

# List / continue numbering
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --series red_climb_v1 --list
```

### B. Live human Red enter (chain enemy phase)

```bash
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --segment red-to-hellway-human --series red_climb_human_v1
```

### C. Any pin override (F6 mid seats, custom enter)

```bash
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --segment red-to-hellway --series red_climb_v1 \
  --from snes/super_metroid/tasks/warehouse_to_red_human_anchors/f002012_enter_0xA253_0xA253.state
```

### D. One-shot (no series loop)

```bash
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from red-bottom --name red_to_hellway_clean --no-guide
```

### Controls

| Key | Effect |
|-----|--------|
| **F6** | Manual mid pin (thin seat, ice tiers) — keep even if climb fails later |
| **F5 / F1** | Save take + end state; practice loop reloads start pin |
| **ESC / Q** | Discard take, end series |

### F6 checklist (Red climb)

1. Midplat / temp floor (~y1600–1450)
2. Thin seat **~(91, 587)** after period WJ
3. Ice-ripper tiers ~y495 / 391 / 295 / 207
4. Hellway door / enter → **F5**

Prefer dwell **≪ 6k**. Thrash takes still OK as inventory if Hellway is reached;
rank prefers shorter `red_dwell`.

## Reactive Ice/WJ checkpoint plan (probe / human only)

**Not wired into product `RoomAutopilot`.** Red climb stays human (or probe
scripts). Do not re-hardcode route-specific checkpoint trees into AP.

The pure probe has one independently verified, enemy-aware edge from the
natural Red bottom pin to the first frozen lower Ripper. It reads the live
Ripper X instead of assuming the human tape's patrol phase, freezes only in a
repeatable launch band, executes a consecutive WJ skill, and proves a grounded
landing on still-frozen support before handing control back.

```bash
uv run python snes/super_metroid/scripts/probe/red_ice_climb.py --save
# GREEN dual exact + 31 patrol phases total; 230..414f at 408..636 FPS
# partial only: lower_ripper_1; Hellway remains RED
```

Use the PNGs as the review surface and the JSON as coordinate truth:

| Artifact | Role |
|---|---|
| `docs/tasks/refs/red_tower_ice_first_edge.png` | Bottom context, action sequence, acceptance and non-claim |
| `docs/tasks/refs/red_tower_ice_checkpoint_plan.png` | Full ten-screen checkpoint/recovery tree |
| `routes/kpdr/data/red_tower_ice_checkpoint_plan.json` | Exact checkpoint bands, edge status, equipment contract |

Next edge is `lower_ripper_3 → lower_ripper_4` (enemy y=2048, 136px gap).
`r1 → r2` is dual-green **156f** ×2; `r2 → r3` is dual-green **108f** ×2
from the Ice-pin r2 pin (`red_ice_r2_to_r3.py`): wait until the Ripper is
on the facing (right) side, freeze at offset with UP+X (no d-pad), standing
Hi-Jump, drift from above. Shooting the left-side approach freezes overhead
and the hop bonks. Do not RIGHT+A from pose 3 (falls through as pose 81).

## Rank + pick best

```bash
uv run python snes/super_metroid/scripts/tools/rank_red_climb_takes.py \
  --series red_climb_v1

uv run python snes/super_metroid/scripts/tools/rank_red_climb_takes.py \
  --series red_climb_v1 \
  --write-manifest snes/super_metroid/tasks/red_climb_v1/splice_manifest.json
```

Grades:

| Grade | Meaning |
|-------|---------|
| GREEN | Hellway + red_dwell ≤ 4500 |
| YELLOW | Hellway but long thrash |
| RED | Never left Red |

## After a good take

```bash
# Hop inventory
uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \
  snes/super_metroid/tasks/red_climb_v1/red_climb_v1_take07.json --summary

# Dual hop-replay (enter → leave) — assist ON default
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/red_climb_v1/red_climb_v1_take07.json --hop 0 --dual

# Pure bot residual (separate product track)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure red-to-hellway \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ice_bat_to_red_pure.state \
  --no-red-diag
```

## Full Ice → Moat human board (current)

```text
ice_to_red_human              Ice → Warehouse elev
warehouse_to_red_human        Warehouse → Red → Hellway → Alpha PB  (sloppy Red)
  └─ REPLACE Red hop via red_climb_v1 best take
alpha_pb_to_moat_human        Alpha PB → Moat pre-spark  ✅ clean
```

Moat spark pure remains `rr-hhj` / `moat_spark_watch` — do not re-prove on this tape.

## Non-claims

- Multi-take human ≠ pure dual Green Red→Hellway
- Best human take ≠ continuous past Ice
- Splice board ≠ power-on STATUS tip
- `alpha_pb_to_moat_human` ≠ continuous Moat approach from power-on (`rr-dbu.9`)

See also: [FULL_STITCH_GAPS.md](FULL_STITCH_GAPS.md) · [rr-av5s-residual.md](rr-av5s-residual.md) · [HUMAN_TAPE_PIPELINE.md](HUMAN_TAPE_PIPELINE.md).
