# KPDR progress tracker

Machine-readable source: `KPDR_TRACKER.csv` · JSON: `maps/kpdr_tracker.json`.
Regenerate: `uv run python super_metroid/scripts/export/kpdr_tracker.py`.

## Summary

| Metric | Value |
|--------|------:|
| Total segments | 40 |
| Super → Kraid-entry segments | 33 |
| Kraid-entry path progress (weighted) | 81.1% |

### Status counts

| Status | Count |
|--------|------:|
| `continuous` | 14 |
| `controller_dev` | 19 |
| `dev_warp` | 1 |
| `open` | 6 |

### Chart series (status)

```
continuous        14 ##############
controller_dev    19 ####################
dev_warp           1 #
open               6 ######
```

## Segment table (Super → Kraid-entry focus)

| # | Seg | Room | Status | Layer | Item/Boss | Anchor |
|--:|-----|------|--------|-------|-----------|--------|
| 0 | `K0.0` | 0xDF45 Ceres Elevator | **continuous** | continuous | ceres | `` |
| 1 | `K0.1` | 0x91F8 Landing Site | **continuous** | continuous |  | `` |
| 2 | `K0.2` | 0x9E9F Morph Ball Room | **continuous** | continuous | morph | `` |
| 3 | `K0.3` | 0xA107 First Missile Room | **continuous** | continuous | missiles | `` |
| 4 | `K0.4` | 0x9804 Bomb Torizo Room | **continuous** | continuous | bombs+torizo | `` |
| 5 | `K0.5` | 0x9DC7 Spore Spawn Room | **continuous** | continuous | spore_boss | `` |
| 6 | `K0.6` | 0x9B5B Spore Spawn Super Room | **continuous** | continuous | supers_5 | `natural_post_spore_spawn` |
| 10 | `K1.0` | 0xA0A4 Spore Spawn Farming Room | **controller_dev** | controller |  | `dev_b1_farming_entry` |
| 11 | `K1.1` | 0x9D19 Big Pink | **controller_dev** | controller |  | `dev_b1_bigpink_entry` |
| 12 | `K1.2` | 0x9D19 Big Pink | **controller_dev** | controller |  | `dev_b1_bigpink_main_controller` |
| 13 | `K1.3` | 0x9D19 Big Pink (Charge Chozo) | **open** | controller | charge_beam | `` |
| 14 | `K1.4` | 0x9E52 Green Hill Zone | **controller_dev** | controller |  | `dev_b1_bigpink_main_controller` |
| 15 | `K1.5` | 0x9FBA Noob Bridge | **controller_dev** | controller |  | `dev_kpdr_noob` |
| 16 | `K1.6` | 0xA253 Red Tower | **continuous** | continuous |  | `dev_kpdr_red_tower` |
| 20 | `K2.0` | 0xA3DD Bat Room | **continuous** | continuous |  | `` |
| 21 | `K2.1` | 0xA408 Below Spazer | **continuous** | continuous |  | `` |
| 22 | `K2.2` | 0xA447 Spazer Room | **open** | optional | spazer | `` |
| 23 | `K2.3` | 0xCF54 West Tunnel | **continuous** | continuous |  | `` |
| 24 | `K2.4` | 0xCEFB Glass Tunnel | **continuous** | continuous |  | `` |
| 25 | `K2.5` | 0xCF80 East Tunnel | **continuous** | continuous |  | `` |
| 26 | `K2.6` | 0xA6A1 Warehouse Entrance | **continuous** | continuous |  | `dev_kpdr_warehouse` |
| 27 | `K2.7` | 0xA7DE Business Center | **controller_dev** | controller |  | `` |
| 28 | `K2.8` | 0xAA41 Hi-Jump Shaft | **controller_dev** | controller |  | `` |
| 29 | `K2.9` | 0xA9E5 Hi-Jump Room | **controller_dev** | controller |  | `` |
| 30 | `K2.10` | 0xA9E5 Hi-Jump Room | **controller_dev** | controller | hi_jump | `` |
| 31 | `K2.11` | 0xAA41 Hi-Jump Shaft | **controller_dev** | controller |  | `` |
| 32 | `K2.12` | 0xA7DE Business Center | **controller_dev** | controller |  | `` |
| 33 | `K2.13` | 0xA6A1 Warehouse Entrance | **controller_dev** | controller |  | `` |
| 34 | `K2.14` | 0xA471 Warehouse Zeela Room | **controller_dev** | controller |  | `` |
| 35 | `K2.15` | 0xA4DA Warehouse Kihunter Room | **controller_dev** | controller |  | `` |
| 36 | `K2.16` | 0xA521 Baby Kraid Room | **controller_dev** | controller |  | `` |
| 37 | `K2.17` | 0xA56B Kraid's Eye Door | **controller_dev** | controller |  | `` |
| 38 | `K2.18` | 0xA59F Kraid's Room | **controller_dev** | controller |  | `dev_kpdr_kraid_entry` |

## Full route (later KPDR)

| # | Seg | Room | Status | Notes |
|--:|-----|------|--------|-------|
| 40 | `K3.0` | 0xA59F Kraid's Room | controller_dev | Super-spray policy from doorway entry ~1520f boss bit; not continuous yet |
| 41 | `K3.1` | 0xA6E2 Varia Suit Room | controller_dev | rear door + real Varia PLM ~1908f; compose after play_eye_to_kraid on continuous |
| 50 | `K4.0` | 0xAD1B Speed Booster Room | open |  |
| 51 | `K4.1` | 0xADDE Wave Beam Room | open |  |
| 52 | `K4.2` | 0xA890 Ice Beam Room | open |  |
| 60 | `K5.0` | 0xA3AE Alpha Power Bomb Room | open |  |
| 70 | `K6.0` | 0xCD13 Phantoon's Room | dev_warp | after Alpha PB on KPDR |
