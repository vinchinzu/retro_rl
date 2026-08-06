# Speed → Ice → Moat — human tape map + pure beads

**Bead epic:** `rr-dbu`  
**Primary tape:** `tasks/speed_to_ice_moat_human.json`  
(= `recordings/human_tasks/speed_to_ice_moat_human.json` symlink)  
**Start:** `scratch/post_speed_collected.state` (Speed room `0xAD1B`)  
**End state on disk:** `tasks/speed_to_ice_moat_human_end.state` /  
`scratch/speed_to_ice_moat_human_end.state`  
**Frames:** 5373 · duration ~89.6s · assist unlimited energy+ammo (not pure)

## Tape truth (important)

The filename is **aspirational**. The recording **never** reaches Ice (`0xA890`)
or Moat (`0x95FF`). It stops in **Double Chamber** past the blue gate after a
missile pack, without entering Wave (`0xADDE`).

KPDR pure stack after this tape still goes **Wave → Business → Ice → K5 Alpha
→ Crateria Moat**. Moat spark hop is already pure GREEN (`rr-hhj`).

## Human hop table (room-by-room)

| # | frames | room_hex | name | notes |
|---|--------|----------|------|-------|
| 1 | 0–389 (390) | `0xAD1B` | Speed | start ~(169,123) p2; exit left ~(19,139) p10 |
| 2 | 390–955 (566) | `0xACF0` | Speed Hall | left through to Bat Cave |
| 3 | 956–2130 (1175) | `0xB07A` | Bat Cave | pause/inventory f1631–1911 (281f); exit down to Bubble ~(19,395) |
| 4 | 2131–2636 (506) | `0xACB3` | Bubble Mountain | mid-right door → Single ~(492,395) p9 |
| 5 | 2637–3302 (666) | `0xAD5E` | Single Chamber | sel missiles=1 @ f3031; exit → Double ~(238,395) |
| 6 | 3303–5372 (2070) | `0xADAD` | Double Chamber | blue gate open; missile +5 @ f5206; **end ~(494,139)** |

### Double Chamber gate (human) — attach to `rr-re9`

| window | frames | geometry / inputs |
|--------|--------|-------------------|
| seat | ~4650–4710 | ~(378,139–160) missiles sel=1; R stand then R+X |
| peak volley | ~4714–4731 | A+R+X peak y≈108–111 pose 105 |
| fall X | ~4834–4848 | pure X pose 19 y122–160 |
| second volley | ~5022–5055 | sel beam=0; peak A+X+R |
| **past gate** | **f5126** | **x=413 y=135** spin/walk right |
| missile pack | f5206 | ~(494,139) missiles 15→20 |
| tape end | f5372 | still `0xADAD` ~(494,139) p9 — **no Wave room** |

## Map to existing pure cards / beads

| Hop | BACKLOG / residual | Bead | Status (product pure) |
|-----|--------------------|------|------------------------|
| Speed → Bubble | SM-K4.7 / SM-K4.7-PURE-residual | **rr-g4i** | closed GREEN |
| Bubble → Single | SM-K4.8 / SM-K4.8-BUBBLE-SINGLE-residual | **rr-yzv** | closed GREEN residual; **smoke re-check RED** → **rr-dbu.1** |
| Single → Double | SM-K4.9 / SM-K4.9-PURE-residual | **rr-g1b** | closed GREEN |
| Double → Wave PLM | SM-K4.10 / SM-K4.10-PURE-residual | **rr-re9** | **in_progress RED** (blue gate) |
| Continuous `--to wave` | SM-K4-TIP-WAVE | **rr-l0u** | open (blocked on re9) |
| Wave → Business | SM-K4.11-PURE | **rr-dbu.2** | open (blocked on re9) |
| Business → Ice Gate | SM-K4.12-PURE | **rr-dbu.3** | open |
| Ice Gate → Tutorial/Snake | SM-K4.13-PURE | **rr-dbu.4** | open |
| Snake → pre-Ice seat | SM-K4.14-PURE | **rr-dbu.5** | open |
| Ice Beam PLM | SM-K4.15-PURE | **rr-dbu.6** | open |
| Continuous `--to ice` | SM-K4-TIP-ICE | **rr-dbu.7** | open P2 (blocked on ice) |
| K5 Alpha stack | SM-K5.* tracker | **rr-dbu.8** | open P2 (blocked on ice) |
| Moat approach → pre-spark | SM-K6.0-PURE | **rr-dbu.9** | open P2; relates **rr-hhj** |
| Moat spark → West Ocean | SM-MOAT-SHINESPARK | **rr-hhj** | closed GREEN |

## Probe smoke (2026-08-06 recon)

```bash
SCR=snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch

# GREEN — Speed return
uv run python snes/super_metroid/scripts/probe/kpdr.py pure speed-return-to-bubble \
  --source $SCR/post_speed_collected.state --no-red-diag
# success=true (room Bubble path; frames≈2158 class)

# RED — Bubble→Single from current post_speed_return pin (vs residual GREEN)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure bubble-to-single-chamber \
  --source $SCR/post_speed_return_to_bubble_pure.state --no-red-diag
# success=false Single Chamber door missed; room=0xACB3 pose=164 xy=(475,395) frames=3558

# GREEN — Single→Double from saved post_bubble_to_single pin
uv run python snes/super_metroid/scripts/probe/kpdr.py pure single-to-double-chamber \
  --source $SCR/post_bubble_to_single_chamber_pure.state --no-red-diag
# success=true

# RED — Wave (rr-re9)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure double-chamber-to-wave \
  --source $SCR/post_single_to_double_chamber_pure.state --no-red-diag
# success=false Wave door missed; room=0xADAD xy=(475,409) frames=2995

# GREEN — Moat spark (rr-hhj already closed)
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py pure \
  --source $SCR/post_kihunter_pre_moat_spark.state
# GREEN room=0x93FE frames=721
```

| Segment | Result |
|---------|--------|
| `speed-return-to-bubble` | **GREEN** |
| `bubble-to-single-chamber` (from `post_speed_return_to_bubble_pure`) | **RED** (re-verify **rr-dbu.1**) |
| `single-to-double-chamber` (from saved SC pin) | **GREEN** |
| `double-chamber-to-wave` | **RED** (known **rr-re9**) |
| Ice / ship approach pure CLIs | **missing** (beads **rr-dbu.2–.9**) |
| Moat spark pure | **GREEN** (**rr-hhj**) |

## Serial dep spine (beads)

```
rr-g4i ✓ → rr-yzv ✓ → rr-g1b ✓ → rr-re9 ◐
                              ↘ rr-l0u (tip wave)
                              ↘ rr-dbu.2 → .3 → .4 → .5 → .6 Ice PLM
                                                 ↘ rr-dbu.7 tip ice
                                                 ↘ rr-dbu.8 K5 → rr-dbu.9 Moat approach ↔ rr-hhj ✓
rr-dbu.1 ready: re-verify Bubble→Single natural entry
```

## Next pure ready action

1. **`rr-re9`** — Pure Wave Beam PLM / Double Chamber blue gate (mainline; in_progress).  
   Human gate frames above + residual `SM-K4.10-PURE-residual.md`.
2. **`rr-dbu.1`** — Re-verify / fix Bubble→Single from `post_speed_return_to_bubble_pure`
   (natural-entry flake vs residual GREEN).

Do **not** STATUS-promote continuous tip (still **speed**, 130388f dual).  
Do **not** close `rr-re9` without pure Wave green.

## Non-claims

- No continuous `--to ice` / `--to moat` / `--to wave` integrity.
- No STATUS / MILESTONES promote.
- Human tape is assist ammo/energy; gate open is not pure evidence.
- Ice/Moat hops beyond Double Chamber are BACKLOG-serial, not tape-derived geometry.
