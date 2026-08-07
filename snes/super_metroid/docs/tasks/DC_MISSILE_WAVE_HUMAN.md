# Double Chamber missile → Wave — human take04 reference

**Reference take:** `tasks/dc_missile_v1/dc_missile_v1_take04.json`  
**Start:** `scratch/post_single_to_double_chamber_continuous_like.state` (Spazer `0x1004`)  
**Result:** Wave room `0xADDE` xy≈(171,123) missiles=20 frames=**2940**  
**Path data:** `routes/kpdr/data/dc_missile_wave_take04_paths.json`

P1 was hit-and-miss (floor falls). **P3–P5 (second phase)** on take04 is the
clean recipe to practice and to teach the bot free+runway.

## Practice

```bash
# Guide ON: purple = main phases, red = floor recover fallback
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --segment dc-missile-wave --series dc_missile_v1 --guide

# Re-parse any take → paths JSON
uv run python snes/super_metroid/scripts/tools/parse_human_take.py \
  snes/super_metroid/tasks/dc_missile_v1/dc_missile_v1_take04.json
```

| Control | Action |
|---------|--------|
| F5 / F1 | Save take |
| ESC / Q | Cancel |
| Guide colors | **Purple** main path · **Red** floor recover |

## Phases

| Phase | Frames (take04) | Goal | Fallback |
|-------|----------------:|------|----------|
| **P1** entry hop | 0–1030 | Upper platforms → gate seat ~(379,139) | Red path: floor y≈400 → climb → reseat |
| **P2** gate open | 1031–1162 | Open blue gate → past ~(480,139) | Re-seat if Kamer drops you |
| **P3** missile free | 1163–1574 | Pack ~x494; **RIGHT+B free 406f** to x≥510 | Never LEFT-only while frozen |
| **P4** runway dash | 1575–1714 | Backup ~x437 → dash edge ~600 | Stay y≈139; no spike floor |
| **P5** launch Super | 1715–2117 | Peak y≈60 → door WJ → Super → `0xADDE` | Abort if wall contact y>280 |
| **P6** Wave collect | 2118–2939 | Chozo / beams \|= 0x0001 | — |

## Second-phase timings (take04)

| Event | Frame | xy |
|-------|------:|-----|
| Past gate | 1163 | (480,139) |
| Missile collect | 1168 | (494,139) |
| Free x≥510 | 1574 | (510,139) — free **406f** |
| Edge ~600 | 1712 | runway dash done |
| Wave door | 2118 | room `0xADDE` |

## Fallback rules

1. **`y ≥ 300` left of gate (`x < 480`)** → red recover → reseat P2. Do not
   attempt Super from floor.
2. **Missile pin `x≈492` vx=0** → hold **RIGHT+B** until `x ≥ 510`.
3. **Door contact y > 280** → abort WJ; return to ledge runway (P4).

## Bot parity

```bash
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --segment dc-missile-wave --bot-check
# pure double-chamber-to-wave from cont-like leave; expect GREEN 2336f dual
```

Human take04 is longer (2940f) than pure bot (2336f) because of P1 floor
recovery — second phase free+runway matches the product knob.
