# Status — Super Metroid

Glossary: [`CONTEXT.md`](../CONTEXT.md). Working board: [plan.md](plan.md).

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M5 (matrix sticker; rungs are the board) |
| Best verified result | Continuous power-on → **Phantoon defeat + basement leave** (KPDR K6) |
| Last verification | 2026-08-24 (STATUS promote 2026-08-25, `rr-b926`) |
| Runtime class | Bronze |
| Intervention class | Survival (resource-assisted: energy + unlocked ammo) |
| Target | Power-on through end of credits with an RTA (any% KPDR, noob loadout) |
| Current assists | Current energy on Zebes + naturally unlocked current ammo |
| Default CLI tip | `phantoon` |
| Acceptance | Ordinary WS Basement `0xCC6F` `(1240,139)` p10 gs=8 + `$D82B` bit 0 |
| Machine report | `recordings/phantoon.json` + `phantoon_dual.json` (**195,336f** each, exact) |
| Save-state loads | 0 |
| Progression / capacity writes | 0 |
| Deaths | 0 |
| Video | `recordings/phantoon.mp4` (Ceres on tape; frames match) |
| Shared ROM SHA-256 | `12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72` |
| Milestone board | [routes/MILESTONES.md](routes/MILESTONES.md) |
| Ready work | `bd ready -l super_metroid -l spine` |
| Clean track (parallel) | Morph prefix clean @ **26,824f**; bombs/Torizo Clean **GREEN 49,321f** ×2 — [CLEAN_TRACK.md](CLEAN_TRACK.md); does **not** change this gate |

## Current verified tip — Phantoon (K6)

Two matching `--to phantoon` runs (2026-08-24, `rr-8g2u`) reached ordinary
Wrecked Ship Basement `0xCC6F` after doppler kill + loot/exit. Exact frame
match both runs (**195,336f**). Integrity green: known transitions, ordered
splits, **0** loads / progression / capacity writes / deaths. Beams **`0x1007`**
(Charge+Spazer+Wave+Ice). Items **`0x3105`**. Video frames match. Planner
STATUS promote `rr-b926` (2026-08-25): this is the **one living tip**. Ice /
Wave / Speed / Moat / WS are prefix CI, not extra products.

| Metric | Value |
|--------|------:|
| Total frames | **195,336** (~54.26 min @ 60 fps), ×2 exact |
| WO → WS | 175,967 @ `0xCA08` |
| Entrance | 176,402 @ `0xCAF6` |
| Main | 177,636 @ `0xCC6F` |
| Basement → room | 178,300 @ `0xCD13` |
| Fight | 195,000 @ `0xCD13` |
| Loot-exit | 195,168 @ `0xCC6F` |
| Final room | `0xCC6F` ordinary gameplay `(1240,139)` p10 gs=8 |
| Beams | **`0x1007`** (Charge+Spazer+Wave+Ice) |
| Items | **`0x3105`** |
| Outcome | `phantoon_defeated` |
| Report | `recordings/phantoon.json` + `phantoon_dual.json` |
| Video | `recordings/phantoon.mp4` |

Previous living tip (still valid prefix): Ice Beam `--to ice` **148,167f** ×2
(`recordings/ice.json` + `ice_dual.json`, 2026-08-10, room `0xA890`, beams
`0x1007`). Superseded as default CLI tip — not false.

Previous prefix: Wave Beam `--to wave` **136,361f** ×2. Speed Booster
`--to speed` **130,388f** ×2. Frog Save `--to frog` **114,923f** ×2 (side).
Non-Spazer Bat Cave **122,304f** ×2 remains valid history.

★ Next: **Gravity** on this tip (`rr-kw8t`). Pin
`scratch/post_phantoon_leave.state`. Power-on green is the rung. Residual:
[`tasks/rr-kw8t-residual.md`](tasks/rr-kw8t-residual.md).
Work: `bd ready -l super_metroid -l spine`.

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
| Wave Beam | `wave` | **136,361** | K4.10 prefix CI |
| Ice Beam | `ice` | **148,167** | K4 previous living tip |
| **Phantoon** | **`phantoon`** | **195,336** | **K6 living tip** (rr-8g2u dual exact match; STATUS `rr-b926`) |

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
uv run python snes/super_metroid/scripts/record/continuous.py --to phantoon --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --no-video   # default tip = phantoon
uv run python snes/super_metroid/scripts/record/continuous.py --to ice --no-video     # previous living tip
uv run python snes/super_metroid/scripts/record/continuous.py --to wave --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to speed --no-video
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
