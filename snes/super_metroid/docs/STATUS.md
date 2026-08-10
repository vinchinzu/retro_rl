# Status — Super Metroid

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M5 |
| Best verified result | Continuous power-on → **Wave Beam collect** (KPDR K4.10) |
| Last verification | 2026-08-06 |
| Runtime class | Bronze |
| Intervention class | Resource-assisted |
| Target | Continuous assisted power-on → ending/credits |
| Current assists | Current energy on Zebes + naturally unlocked current ammo |
| Default CLI tip | `wave` |
| Acceptance | Natural Wave Room entry + Wave Beam PLM (`beams …\|0x0001`) under Spazer mainline |
| Machine report | `recordings/wave.json` + `wave_dual.json` (**136,361f** each) |
| Save-state loads | 0 |
| Progression / capacity writes | 0 |
| Deaths | 0 |
| Video | No-video dual verification (first video still open) |
| Shared ROM SHA-256 | `12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72` |
| Milestone board | [routes/MILESTONES.md](routes/MILESTONES.md) |
| Backlog | [routes/BACKLOG.csv](routes/BACKLOG.csv) |
| Clean track (parallel) | Morph prefix clean @ **26,824f**; bombs/Torizo Clean **GREEN 49,321f** ×2 — [CLEAN_TRACK.md](CLEAN_TRACK.md); does **not** change this gate |

## Current verified tip — Wave Beam (K4.10)

Two matching `--to wave --no-video` runs (2026-08-06) reached ordinary Wave
Room `0xADDE` with Wave Beam collected. Exact frame match both runs.
Integrity green: known transitions (incl. post-Speed reverse), ordered splits,
**0** loads / progression / capacity writes / deaths. Spazer mainline beams
**`0x1005`** (Charge+Spazer+Wave).

| Metric | Value |
|--------|------:|
| Total frames | **136,361** (~37.88 min @ 60 fps), ×2 |
| Speed Hall → Speed split | 129,063 @ `0xAD1B` |
| Speed return → Bubble | 132,057 @ `0xACB3` |
| Bubble → Single | 132,338 @ `0xAD5E` |
| Single → Double | 133,020 @ `0xADAD` |
| Double → Wave | 136,005 @ `0xADDE` |
| Final room | `0xADDE` ordinary gameplay |
| Beams | **`0x1005`** (Charge+Spazer+Wave) |
| Items | **`0x3105`** (includes Speed `0x2000`) |
| Outcome | `wave_collected` |
| Report | `recordings/wave.json` + `wave_dual.json` |

Previous tip (still valid prefix): Speed Booster `--to speed` **130,388f** ×2
(`recordings/speed_spazer.json` + `_dual.json`, 2026-08-06, room `0xAD1B`,
beams `0x1004`). Superseded as default CLI tip — not false.

Side tip (still valid): Frog Save `--to frog` **114,923f** ×2
(`recordings/frog.json` + reverify). Non-Spazer Bat Cave **122,304f** ×2
remains valid history.

★ Next: Ice pure stack — Gate dual GREEN (`rr-fg3` 894f) + Gate→Acid dual GREEN
(`rr-9t4` 370f ×2) + Acid→Snake dual GREEN (`rr-5cf` **652f** ×2) +
**Snake → Ice PLM dual GREEN** (`rr-5if` **1756f** ×2, room `0xA890`, beams
`0x1007`). Outbound pure PLM hops green; parent stack `rr-dbu.11` / continuous
`--to ice` still blocked on compose (`rr-dbu.7`). Work: `bd ready -l super_metroid`.

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
| Bat Cave | `bat_cave` | **122,304** | K4.4 previous default tip (non-Spazer dual) |
| Speed Booster | `speed` | **130,388** | K4.5 previous default (Spazer dual) |
| **Wave Beam** | **`wave`** | **136,361** | **K4.10 default tip** (Spazer dual exact match) |

All listed tips are integrity-green continuous (0 loads / progression / capacity).
Detail and history: [routes/MILESTONES.md](routes/MILESTONES.md).

## Spazer mainline — Warehouse dual (promoted prefix)

K2.2 Spazer on continuous spine through Warehouse. Two integrity-matching runs
(2026-08-05 / 2026-08-06): same tip room, beams, and zero load/prog/capacity/death
flags. Frame delta attributed to Spore combat variance — not a tip mismatch.
Prefix of the Speed tip (beams `0x1004` through Speed dual).

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

Spazer spine extend (historical single, superseded by Speed dual): `--to bat_cave`
with Spazer **127,806f** beams `0x1004` room `0xB07A`
(`recordings/bat_cave_spazer_cwu.json`, 2026-08-06). Folded into Speed dual at
**130,388f**.

## Clean track

Secondary only — does **not** change the program gate above.

| Fact | Value |
|------|-------|
| Morph on clean bombs path | **26,824f** (split; matches assisted morph tip) |
| Clean bombs tip | **GREEN** **49,321f** ×2 (2026-08-06) — parlor `0x92FD`, items `0x1004` |
| Clean integrity | 0 energy/ammo writes; 0 loads/progression/capacity; dual reverify |
| Residual | purged (clean bombs dual GREEN; see CLEAN_TRACK) |
| Next | `SM-CLEAN-STATUS` secondary promote; spore clean still parked |

Contract: [CLEAN_TRACK.md](CLEAN_TRACK.md) · [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md).
No Clean dual / STATUS primary claim.

## Reproduce

```bash
uv run python snes/super_metroid/scripts/record/continuous.py --to wave --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --no-video   # default tip = wave
uv run python snes/super_metroid/scripts/record/continuous.py --to speed --no-video  # previous tip
uv run python snes/super_metroid/scripts/record/continuous.py --to bat_cave --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to frog --no-video   # side tip
uv run python snes/super_metroid/scripts/record/continuous.py --to warehouse --no-video  # Spazer warehouse prefix
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
