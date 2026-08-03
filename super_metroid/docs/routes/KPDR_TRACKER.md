# KPDR progress tracker

Machine-readable source: `KPDR_TRACKER.csv` · JSON: `maps/kpdr_tracker.json`.
Regenerate: `uv run python super_metroid/scripts/export/kpdr_tracker.py`.

## Summary

| Metric | Value |
|--------|------:|
| Total segments | 92 |
| Super → Kraid-entry segments | 33 |
| Kraid-entry path progress (weighted) | 93.9% |

### Status counts

| Status | Count |
|--------|------:|
| `continuous` | 41 |
| `open` | 50 |
| `parked` | 1 |

### Chart series (status)

```
continuous        41 ################
open              50 ####################
parked             1 #
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
| 10 | `K1.0` | 0xA0A4 Spore Spawn Farming Room | **continuous** | continuous |  | `dev_b1_farming_entry` |
| 11 | `K1.1` | 0x9D19 Big Pink | **continuous** | continuous |  | `dev_b1_bigpink_entry` |
| 12 | `K1.2` | 0x9D19 Big Pink | **continuous** | continuous |  | `dev_b1_bigpink_main_controller` |
| 13 | `K1.3` | 0x9D19 Big Pink (Charge Chozo) | **open** | optional | charge_beam | `` |
| 14 | `K1.4` | 0x9E52 Green Hill Zone | **continuous** | continuous |  | `dev_b1_bigpink_main_controller` |
| 15 | `K1.5` | 0x9FBA Noob Bridge | **continuous** | continuous |  | `dev_kpdr_noob` |
| 16 | `K1.6` | 0xA253 Red Tower | **continuous** | continuous |  | `dev_kpdr_red_tower` |
| 20 | `K2.0` | 0xA3DD Bat Room | **continuous** | continuous |  | `` |
| 21 | `K2.1` | 0xA408 Below Spazer | **continuous** | continuous |  | `` |
| 22 | `K2.2` | 0xA447 Spazer Room | **open** | optional → tip/fold | spazer | `SM-SPAZER-*` / SPAZER_EARLY |
| 23 | `K2.3` | 0xCF54 West Tunnel | **continuous** | continuous |  | `` |
| 24 | `K2.4` | 0xCEFB Glass Tunnel | **continuous** | continuous |  | `` |
| 25 | `K2.5` | 0xCF80 East Tunnel | **continuous** | continuous |  | `` |
| 26 | `K2.6` | 0xA6A1 Warehouse Entrance | **continuous** | continuous |  | `dev_kpdr_warehouse` |
| 27 | `K2.7` | 0xA7DE Business Center | **continuous** | continuous |  | `` |
| 28 | `K2.8` | 0xAA41 Hi-Jump Shaft | **continuous** | continuous |  | `` |
| 29 | `K2.9` | 0xA9E5 Hi-Jump Room | **continuous** | continuous |  | `` |
| 30 | `K2.10` | 0xA9E5 Hi-Jump Room | **continuous** | continuous | hi_jump | `` |
| 31 | `K2.11` | 0xAA41 Hi-Jump Shaft | **continuous** | continuous |  | `` |
| 32 | `K2.12` | 0xA7DE Business Center | **continuous** | continuous |  | `` |
| 33 | `K2.13` | 0xA6A1 Warehouse Entrance | **continuous** | continuous |  | `` |
| 34 | `K2.14` | 0xA471 Warehouse Zeela Room | **continuous** | continuous |  | `` |
| 35 | `K2.15` | 0xA4DA Warehouse Kihunter Room | **continuous** | continuous |  | `` |
| 36 | `K2.16` | 0xA521 Baby Kraid Room | **continuous** | continuous |  | `` |
| 37 | `K2.17` | 0xA56B Kraid's Eye Door | **continuous** | continuous |  | `` |
| 38 | `K2.18` | 0xA59F Kraid's Room | **continuous** | continuous |  | `dev_kpdr_kraid_entry` |

## Full route (later KPDR)

| # | Seg | Room | Status | Notes |
|--:|-----|------|--------|-------|
| 40 | `K3.0` | 0xA59F Kraid's Room | continuous | Super-spray on continuous chain; fight segment of K3 |
| 41 | `K3.1` | 0xA6E2 Varia Suit Room | continuous | 101954f continuous K3 tip; 0 loads / 0 progression writes |
| 42 | `K3.2` | 0xA59F Kraid's Room | continuous | integrity-green Business return twice at 113723f; 0 loads/progression/capacity/deaths |
| 43 | `K3.3` | 0xA59F Kraid's Room | continuous | integrity-green Business return twice; elevated jump-enter band |
| 44 | `K3.4` | 0xA56B Kraid's Eye Door | continuous | integrity-green Business return twice |
| 45 | `K3.5` | 0xA521 Baby Kraid Room | continuous | integrity-green Business return twice; gray-door clear |
| 46 | `K3.6` | 0xA4DA Warehouse Kihunter Room | continuous | integrity-green Business return twice; Hi-Jump y-gated landing |
| 47 | `K3.7` | 0xA471 Warehouse Zeela Room | continuous | integrity-green Business return twice; valid upper-left transition |
| 48 | `K3.8` | 0xA6A1 Warehouse Entrance | continuous | right-ledge reverse stack green; lower-lip correction + two-tier Super clear; 113723f twice |
| 50 | `K4.0` | 0xB167 Frog Savestation | continuous | Business elevator descent + blue-door exit; 114923f integrity green twice |
| 51 | `K4.1` | 0xB106 Frog Speedway | open | ★ next pure hop |
| 52 | `K4.2` | 0xAF72 Bubble Mountain approach | open | Speedway corridor exit |
| 53 | `K4.3` | 0xACB3 Bubble Mountain | open |  |
| 54 | `K4.4` | 0xB07A Speed Hall approach | open |  |
| 55 | `K4.5` | 0xACF0 Speed Booster Hall | open |  |
| 56 | `K4.6` | 0xAD1B Speed Booster Room | open | real PLM collect |
| 57 | `K4.7` | 0xACB3 Bubble Mountain | open | return for Wave branch |
| 58 | `K4.8` | 0xADAD Single Chamber | open | Wave path |
| 59 | `K4.9` | 0xAD5E Double Chamber | open |  |
| 60 | `K4.10` | 0xADDE Wave Beam Room | open | real PLM |
| 61 | `K4.11` | 0xA7DE Business Center | open | via Cathedral / Bubble reverse |
| 62 | `K4.12` | 0xA815 Ice Beam Gate Room | open |  |
| 63 | `K4.13` | 0xA865 Ice Beam Snake Room | open |  |
| 64 | `K4.14` | 0xA8B9 Ice Beam Reflection | open |  |
| 65 | `K4.15` | 0xA890 Ice Beam Room | open | real PLM |
| 70 | `K5.0` | 0xA253 Red Tower | open | post-Ice elev to Red |
| 71 | `K5.1` | 0xA3AE Alpha Power Bomb Room | open | first PB capacity |
| 72 | `K5.2` | 0x962A Crateria Kihunter / elev | open | via Red/Green elev |
| 80 | `K6.0` | 0x95FF The Moat | open | Speed/HJ Moat |
| 81 | `K6.1` | 0x93FE West Ocean | open |  |
| 82 | `K6.2` | 0xCA08 Wrecked Ship Entrance | open |  |
| 83 | `K6.3` | 0xCC6F Basement | open | Supers red door |
| 84 | `K6.4` | 0xCD13 Phantoon's Room | open | entry only; fight separate |
| 85 | `K6.5` | 0xCD13 Phantoon's Room | open | BossStrategy |
| 86 | `K6.6` | 0xCA08 Wrecked Ship Entrance | open | ship powered |
| 87 | `K6.7` | 0xCE40 Gravity Suit Room | open | via West Ocean attic |
| 90 | `K7.0` | 0xCEFB Glass Tunnel | open | PB tube break |
| 91 | `K7.1` | 0xD0B9 Main Street | open | Maridia navigation |
| 92 | `K7.2` | 0xD5A7 Aqueduct | open |  |
| 93 | `K7.3` | 0xD95E Botwoon's Room | open |  |
| 94 | `K7.4` | 0xD95E Botwoon's Room | open |  |
| 95 | `K7.5` | 0xD72A The Precious Room / Halfie | open |  |
| 96 | `K7.6` | 0xDA60 Draygon's Room | open |  |
| 97 | `K7.7` | 0xDA60 Draygon's Room | open |  |
| 98 | `K7.8` | 0xD9AA Space Jump Room | open | post-Draygon |
| 100 | `K8.0` | 0xACB3 Bubble Mountain | open | LN approach |
| 101 | `K8.1` | 0xB656 Lower Norfair Elevator | open | Varia+Speed+SJ path |
| 102 | `K8.2` | 0xB510 LN amphitheater path | open |  |
| 103 | `K8.3` | 0xB32E Ridley's Room | open |  |
| 104 | `K8.4` | 0xB32E Ridley's Room | open | BossStrategy |
| 110 | `K9.0` | 0xA66A Statues Room | open | all 4 bosses |
| 111 | `K9.1` | 0xDAAE Tourian Elevator Room | open |  |
| 112 | `K9.2` | 0xDDC4 Metroid Room 4 | open | Ice helpful; assists ok |
| 113 | `K9.3` | 0xDD58 Mother Brain Room | open |  |
| 114 | `K9.4` | 0xDD58 Mother Brain Room | open | multi-phase strategy |
| 115 | `K9.5` | 0xDEDE Tourian Escape Room 4 | open | timer |
| 116 | `K9.6` | 0x91F8 Landing Site | open | endgame |
| 117 | `K9.7` | 0x91F8 Landing Site / credits | open | M8 continuous clear |
| 999 | `OPT-PINK-PB` | 0x9E11 Pink Brinstar Power Bomb Room | parked | not KPDR; optional backfill |
