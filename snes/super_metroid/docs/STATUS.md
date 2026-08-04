# Status — Super Metroid

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M5 |
| Best verified result | Continuous power-on → **Bat Cave** (KPDR K4.4 first Bubble) |
| Last verification | 2026-08-03 |
| Runtime class | Bronze |
| Intervention class | Resource-assisted |
| Target | Continuous assisted power-on → ending/credits |
| Current assists | Current energy on Zebes + naturally unlocked current ammo |
| Default CLI tip | `bat_cave` |
| Acceptance | Natural Varia return + Cathedral first Bubble + R19 double-WJ fire / Super door → Bat Cave |
| Machine report | `recordings/bat_cave.json` + `_reverify.json` (**122,304f** each) |
| Save-state loads | 0 |
| Progression / capacity writes | 0 |
| Deaths | 0 |
| Video | No-video dual verification (first video still open) |
| Shared ROM SHA-256 | `12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72` |
| Milestone board | [routes/MILESTONES.md](routes/MILESTONES.md) |
| Backlog | [routes/BACKLOG.csv](routes/BACKLOG.csv) |
| Clean track (parallel) | Morph **green** 27,074f; next bombs — [CLEAN_TRACK.md](CLEAN_TRACK.md); does **not** change this gate |

## Current verified tip — Bat Cave (K4.4)

Two matching `--to bat_cave --no-video` runs (2026-08-03) reached ordinary Bat Cave
`0xB07A`. Integrity green: known transitions, ordered splits, **0** loads /
progression / capacity writes / deaths.

| Metric | Value |
|--------|------:|
| Total frames | **122,304** (~33.97 min @ 60 fps), ×2 |
| Business return | 114,750 |
| Bubble entry | 120,109 |
| Bat Cave entry | **122,182** |
| Final room | `0xB07A` ordinary gameplay |
| Checkpoint | `scratch/post_bat_cave_continuous.state` |
| Outcome | `bat_cave_reached` |

Side tip (still valid): Frog Save `--to frog` **114,923f** ×2
(`recordings/frog.json` + reverify).

★ Next pure: **Bat → Speed Hall** from `post_bubble_to_bat_pure` /
`post_bat_cave_continuous`.

## Continuous prefix tips (frames only)

| Tip | CLI | Frames | Notes |
|-----|-----|-------:|-------|
| Morph | `morph` | 27,074 | K0 |
| Bombs / Torizo | `bombs` | 47,132 | K0 |
| Spore exit | `spore` | 73,216 | K0 |
| Spore Supers | `supers` | 73,251 | K0 |
| Red Tower | `red_tower` | 80,445 | K1 |
| Bat Room | `bat` | 81,652 | K2 |
| Below Spazer | `below_spazer` | 82,300 | K2 |
| Warehouse | `warehouse` | 83,512 | K2 |
| Hi-Jump | `hijump` | 87,696 | K2 |
| Kraid entry | `kraid` | 97,170 | K2 |
| Varia | `varia` | 101,954 best / 104,382 re-verify | K3; keep best published |
| Business return | `business` | 113,723 | K3→K4 |
| Frog Save | `frog` | 114,923 | K4.0 side tip |
| **Bat Cave** | **`bat_cave`** | **122,304** | **K4.4 default tip** |

All listed tips are integrity-green continuous (0 loads / progression / capacity).
Detail and history: [routes/MILESTONES.md](routes/MILESTONES.md).

## Clean track

Morph green **27,074f** (`--to morph --clean`). ★ Next: bombs / Torizo Clean.
Contract: [CLEAN_TRACK.md](CLEAN_TRACK.md) · [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md).

## Reproduce

```bash
uv run python snes/super_metroid/scripts/record/continuous.py --to bat_cave --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to frog --no-video   # side tip
uv run python snes/super_metroid/scripts/record/continuous.py --to morph --clean
```

## Pointers

| Doc | Role |
|-----|------|
| [routes/MILESTONES.md](routes/MILESTONES.md) | Full milestone board |
| [routes/BACKLOG.csv](routes/BACKLOG.csv) | Ticket inventory |
| [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md) | Allowed assists |
| [CLEAN_TRACK.md](CLEAN_TRACK.md) | Clean intervention track |
| [plan.md](plan.md) | Future work only |
| [ARCHITECTURE.md](ARCHITECTURE.md) · [tasks/PROCESS.md](tasks/PROCESS.md) | Layers + pure-first process |
| [routes/ROUTE_KPDR.md](routes/ROUTE_KPDR.md) · [routes/KPDR_TRACKER.md](routes/KPDR_TRACKER.md) | KPDR spine / progress chart |
