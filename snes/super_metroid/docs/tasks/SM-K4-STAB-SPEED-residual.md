# Residual — SM-K4-STAB-SPEED / rr-07b

## Result

GREEN

## Pure verify

| Segment | Source | Result |
|---------|--------|--------|
| `bat-cave-to-speed-hall` | `scratch/post_bat_cave_continuous.state` | GREEN room `0xACF0` frames=814 |
| `speed-hall-to-speed` | `scratch/post_bat_cave_to_speed_hall_pure.state` | GREEN room `0xAD1B` frames=1732 items `0x3105` |
| `speed-return-to-bubble` | `scratch/post_speed_collected.state` | GREEN room `0xACB3` frames=2158 |

## Continuous dual

Default tip remains **`speed`**. Both integrity re-records match baseline **130388f**:

| Report | frames | room | beams | items | loads | prog | deaths |
|--------|-------:|------|-------|-------|------:|-----:|-------:|
| baseline `speed_spazer_dual.json` | 130388 | 0xAD1B | 0x1004 | 0x3105 | 0 | 0 | 0 |
| `speed_spazer_stab.json` | 130388 | 0xAD1B | 0x1004 | 0x3105 | 0 | 0 | 0 |
| `speed_spazer_stab_dual.json` | 130388 | 0xAD1B | 0x1004 | 0x3105 | 0 | 0 | 0 |

Tail splits identical both stab runs: bat_cave@127684 → speed_hall@128505 → speed@129558.

## Files changed

- none (stabilize-only; no knobs)
- this residual note
- gitignored: `recordings/speed_spazer_stab.json`, `speed_spazer_stab_dual.json`

## Verify paste

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure bat-cave-to-speed-hall \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bat_cave_continuous.state --no-red-diag
# success=true room=0xACF0 frames=814

uv run python snes/super_metroid/scripts/probe/kpdr.py pure speed-hall-to-speed \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bat_cave_to_speed_hall_pure.state --no-red-diag
# success=true room=0xAD1B frames=1732 items=0x3105

uv run python snes/super_metroid/scripts/probe/kpdr.py pure speed-return-to-bubble \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_speed_collected.state --no-red-diag
# success=true room=0xACB3 frames=2158

uv run python snes/super_metroid/scripts/record/continuous.py --to speed --no-video \
  --report snes/super_metroid/recordings/speed_spazer_stab.json
# [GREEN] frames=130388 room=0xAD1B beams=0x1004

uv run python snes/super_metroid/scripts/record/continuous.py --to speed --no-video \
  --report snes/super_metroid/recordings/speed_spazer_stab_dual.json
# [GREEN] frames=130388 exact dual match
```

## Acceptance

- [x] Pure greens documented
- [x] Continuous dual green exact match 130388f
- [x] No code knobs
- [x] Close rr-07b stabilize GREEN

## Next action

- **Next card ID:** planner serial after Speed tip (e.g. pure Wave path / `rr-g4i` Speed return graph if still open) — `bd ready -l super_metroid`
- **One change:** none from this stabilize card
- **Source state:** n/a

## Non-claims

- Did not change controllers, geometry knobs, tip wiring, or STATUS promote
- Did not claim Speed return continuous tip (pure only re-verified)
- Did not invent knobs on RED (none needed — all GREEN)
