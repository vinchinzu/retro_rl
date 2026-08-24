# Status — Super Metroid

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M5 |
| Best verified result | Continuous power-on → **Ice Beam collect** (KPDR K4 Ice) |
| Last verification | 2026-08-10 |
| Runtime class | Bronze |
| Intervention class | Resource-assisted |
| Target | Continuous assisted power-on → ending/credits |
| Current assists | Current energy on Zebes + naturally unlocked current ammo |
| Default CLI tip | `ice` |
| Acceptance | Natural Ice Room entry + Ice Beam PLM (`beams …\|0x0002`) under Spazer mainline |
| Machine report | `recordings/ice.json` + `ice_dual.json` (**148,167f** each) |
| Save-state loads | 0 |
| Progression / capacity writes | 0 |
| Deaths | 0 |
| Video | No-video dual verification (first ice video still open) |
| Shared ROM SHA-256 | `12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72` |
| Milestone board | [routes/MILESTONES.md](routes/MILESTONES.md) |
| Ready work | `bd ready -l super_metroid` |
| Clean track (parallel) | Morph prefix clean @ **26,824f**; bombs/Torizo Clean **GREEN 49,321f** ×2 — [CLEAN_TRACK.md](CLEAN_TRACK.md); does **not** change this gate |

## Current verified tip — Ice Beam (K4 Ice)

Two matching `--to ice --no-video` runs (2026-08-10, `rr-kxge`) reached ordinary
Ice Beam Room `0xA890` with Ice Beam collected. Exact frame match both runs
(**148,167f**). Integrity green: known transitions, ordered splits, **0** loads /
progression / capacity writes / deaths. Spazer mainline beams **`0x1007`**
(Charge+Spazer+Wave+Ice). Business floor climb hardened (cont-tuned 907 runup
ladder + classic warehouse setup preserved).

| Metric | Value |
|--------|------:|
| Total frames | **148,167** (~41.16 min @ 60 fps), ×2 |
| Wave → Double (return) | 136,851 @ `0xADAD` (prefix of ice tip delta) |
| Frog Save → Business | 141,473 @ `0xA7DE` |
| Business → Ice Gate | 145,230 @ `0xA815` |
| Ice Snake → Ice | 147,799 @ `0xA890` |
| Final room | `0xA890` ordinary gameplay |
| Beams | **`0x1007`** (Charge+Spazer+Wave+Ice) |
| Items | **`0x3105`** (includes Speed `0x2000`) |
| Outcome | `ice_collected` |
| Report | `recordings/ice.json` + `ice_dual.json` (from `ice_dual_d` / `ice_dual_e`) |

Previous tip (still valid prefix): Wave Beam `--to wave` **136,361f** ×2
(`recordings/wave.json` + `wave_dual.json`, 2026-08-06, room `0xADDE`, beams
`0x1005`). Superseded as default CLI tip — not false.

Previous tip (still valid): Speed Booster `--to speed` **130,388f** ×2
(`recordings/speed_spazer.json` + `_dual.json`, room `0xAD1B`, beams `0x1004`).

Side tip (still valid): Frog Save `--to frog` **114,923f** ×2
(`recordings/frog.json` + reverify). Non-Spazer Bat Cave **122,304f** ×2
remains valid history.

★ Next: wire `--to ws` over-ocean spark from the power-on Moat leave
(scratch `--to moat` dual **175526f** ×2, not STATUS). Residual:
[`tasks/rr-2r06-residual.md`](tasks/rr-2r06-residual.md). Work:
`bd ready -l super_metroid`.

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
| Wave Beam | `wave` | **136,361** | K4.10 previous default (Spazer dual exact match) |
| **Ice Beam** | **`ice`** | **148,167** | **K4 Ice default tip** (rr-kxge dual exact match) |

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
uv run python snes/super_metroid/scripts/record/continuous.py --to ice --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --no-video   # default tip = ice
uv run python snes/super_metroid/scripts/record/continuous.py --to wave --no-video  # previous tip
uv run python snes/super_metroid/scripts/record/continuous.py --to speed --no-video  # previous tip
uv run python snes/super_metroid/scripts/record/continuous.py --to bat_cave --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to frog --no-video   # side tip
uv run python snes/super_metroid/scripts/record/continuous.py --to warehouse --no-video  # Spazer warehouse prefix
uv run python snes/super_metroid/scripts/record/continuous.py --to morph --clean
```

## Pointers

| Doc | Role |
|-----|------|
| [routes/MILESTONES.md](routes/MILESTONES.md) | Prefix / tip names |
| [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md) | Allowed assists |
| [CLEAN_TRACK.md](CLEAN_TRACK.md) | Clean intervention track |
| [plan.md](plan.md) | Future work only |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layers |
| [routes/ROUTE_KPDR.md](routes/ROUTE_KPDR.md) · [routes/KPDR_TRACKER.md](routes/KPDR_TRACKER.md) | KPDR spine / progress chart |
