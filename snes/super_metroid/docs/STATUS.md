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
| Clean track (parallel) | Morph tip **26,824f** assisted; re-run `--clean` when convenient — [CLEAN_TRACK.md](CLEAN_TRACK.md); does **not** change this gate |

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

★ Next: continuous **`--to speed`** dual integrity (`rr-d20`), then STATUS
promote Speed (`rr-cd0`). Pure Bat→Hall + Hall→collect green; spine tip wired.
Spazer warehouse dual **STATUS-promoted** (below). Bat Cave Spazer continuous
single **127,806f** green (not dual — default tip stays non-Spazer `bat_cave`).
Work: `bd ready -l super_metroid`.

## Continuous prefix tips (frames only)

| Tip | CLI | Frames | Notes |
|-----|-----|-------:|-------|
| Morph | `morph` | **26,824** | K0; Ceres arm-pump + elev top (was 27,074) |
| Bombs / Torizo | `bombs` | 47,132 | K0 |
| Spore exit | `spore` | 73,216 | K0 |
| Spore Supers | `supers` | 73,251 | K0 |
| Red Tower | `red_tower` | 80,445 | K1 |
| Bat Room | `bat` | 81,652 | K2 |
| Below Spazer | `below_spazer` | 82,300 / **84,880** w/ Charge | K2; Charge mainline `below_spazer_with_charge.json` |
| Warehouse | `warehouse` | 83,512 / 85,992 Charge / **89,416+90,904** Spazer dual | K2; Spazer mainline dual (below) |
| Hi-Jump | `hijump` | 87,696 | K2 |
| Kraid entry | `kraid` | 97,170 | K2 |
| Varia | `varia` | 101,954 best / 104,382 re-verify | K3; keep best published |
| Business return | `business` | 113,723 | K3→K4 |
| Frog Save | `frog` | 114,923 | K4.0 side tip |
| **Bat Cave** | **`bat_cave`** | **122,304** | **K4.4 default tip** (non-Spazer dual) |

All listed tips are integrity-green continuous (0 loads / progression / capacity).
Detail and history: [routes/MILESTONES.md](routes/MILESTONES.md).

## Spazer mainline — Warehouse dual (promoted prefix)

K2.2 Spazer on continuous spine through Warehouse. Two integrity-matching runs
(2026-08-05 / 2026-08-06): same tip room, beams, and zero load/prog/capacity/death
flags. Frame delta attributed to Spore combat variance — not a tip mismatch.
Does **not** change the program-gate default CLI tip (`bat_cave` **122,304f**).

| Metric | Run 1 | Run 2 |
|--------|------:|------:|
| Total frames | **89,416** | **90,904** |
| Final room | `0xA6A1` | `0xA6A1` |
| Beams | **`0x1004`** (Charge+Spazer) | **`0x1004`** |
| Outcome | warehouse entry (all final_conditions green) | `warehouse_entry` |
| Integrity core | 0 loads / prog / capacity / deaths | same + success |
| Video frame match | False (encoded clip only) | True |
| Report | `recordings/warehouse_with_spazer.json` | `recordings/warehouse_with_spazer_dual.json` |
| Video (optional) | `recordings/warehouse_with_spazer.mp4` | — |

Spazer spine **extend** (single continuous, not dual STATUS tip): `--to bat_cave`
with Spazer **127,806f** beams `0x1004` room `0xB07A`
(`recordings/bat_cave_spazer_cwu.json`, 2026-08-06). Default published tip
remains non-Spazer `bat_cave` until a dual Spazer bat_cave + promote.

## Clean track

Assisted morph tip is **26,824f**. Clean re-verify (`--to morph --clean`) still
open after the Ceres shave. ★ Next: bombs / Torizo Clean.
Contract: [CLEAN_TRACK.md](CLEAN_TRACK.md) · [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md).

## Reproduce

```bash
uv run python snes/super_metroid/scripts/record/continuous.py --to bat_cave --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to frog --no-video   # side tip
uv run python snes/super_metroid/scripts/record/continuous.py --to warehouse --no-video  # Spazer mainline
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
