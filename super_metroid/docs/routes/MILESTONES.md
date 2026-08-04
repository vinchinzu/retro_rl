# Milestones — Super Metroid assisted full clear

Machine source: [`MILESTONES.csv`](MILESTONES.csv) · backlog: [`BACKLOG.csv`](BACKLOG.csv) · spine: [`KPDR_TRACKER.csv`](KPDR_TRACKER.csv).

**Goal:** one continuous power-on → ending/credits run with **only** unlimited energy + ammo ([`ASSIST_CONTRACT.md`](../ASSIST_CONTRACT.md)). Target runtime class: **less-than-Bronze** assisted full clear (M8).

**Current tip (primary / assisted):** Frog Savestation (`--to frog`, **114,923f** ×2 integrity green). Cathedral first-Bubble pure stack **CATH-01…04 GREEN** + **Bubble → Bat pure GREEN R19** (2012f → `0xB07A`). ★ Next assisted pure: Bat → Speed Hall.

**Parallel Clean track:** no energy + no ammo writes — ★ tip target Bomb Torizo (`--to bombs --clean`). Contract: [`CLEAN_TRACK.md`](../CLEAN_TRACK.md). Clean never demotes assisted greens.

## Status legend

| Mark | Status | Meaning |
|------|--------|---------|
| ✅ | `continuous` | Power-on chain integrity green |
| 🔶 | `controller_dev` | Pure controller green; not yet continuous |
| ▶ | `pure_open` | Next pure hop ready to implement |
| 🟨 | `partial` | In progress / partial rollup |
| ⬜ | `open` | Not started |
| ◇ | `optional` | Safety / optional side path |
| ⏸ | `parked` | Explicitly not on KPDR spine |

## Continuous / product milestones

| | ID | Epic | Milestone | Room | Frames / score | CLI / card |
|--:|----|------|-----------|------|----------------|------------|
| ✅ | `M-MORPH` | K0 | Continuous → Morph | `0x9E9F` | 27074 | `morph` |
| ✅ | `M-BOMBS` | K0 | Continuous → Bombs/Torizo | `0x9804` | 47132 | `bombs` |
| ✅ | `M-SPORE` | K0 | Continuous → Spore exit | `0x9DC7` | 73216 | `spore` |
| ✅ | `M-SUPERS` | K0 | Continuous → Spore Supers | `0x9B5B` | 73251 | `supers` |
| ✅ | `M-RED` | K1 | Continuous → Red Tower | `0xA253` | 80445 | `red_tower` |
| ✅ | `M-BAT` | K2 | Continuous → Bat Room | `0xA3DD` | 81652 | `bat` |
| ✅ | `M-BELOW` | K2 | Continuous → Below Spazer | `0xA408` | 82300 | `below_spazer` |
| ✅ | `M-WH` | K2 | Continuous → Warehouse | `0xA6A1` | 83512 | `warehouse` |
| ✅ | `M-HJ` | K2 | Continuous → Hi-Jump Boots | `0xA9E5` | 87696 | `hijump` |
| ✅ | `M-KRAID` | K2 | Continuous → Kraid entry | `0xA59F` | 97170 | `kraid` |
| ✅ | `M-VARIA` | K3 | Continuous → Varia Suit | `0xA6E2` | 101954 | `varia` |
| ✅ | `M-BUSINESS` | K3 | Continuous → Business return | `0xA7DE` | 113723 | `business` |
| ✅ | `M-FROG` | K4 | Continuous → Frog Savestation | `0xB167` | 114923 | `frog` |
| 🔶 | `M-CATH` | K4 | Cathedral climb pure stack (first Bubble) | `0xA7B3`…`0xAFA3` | CATH-01…04 pure green | closed pure |
| ⏸ | `M-SPEEDWAY` | K4 | Pure Frog Save → Speedway (post-Speed) | `0xB106` | pure green ~295f | parked until Speed |
| 🔶 | `M-BUBBLE` | K4 | Pure → Bubble Mountain via Cathedral | `0xACB3` | pure **2609f** | `SM-K4-CATH-04`; not continuous |
| 🔶 | `M-BAT-CAVE` | K4 | Pure Bubble → Bat Cave | `0xB07A` | pure **2012f** | `SM-K4.4-PURE` R19; not continuous |
| ▶ | `M-BAT-SPEED` | K4 | Pure Bat → Speed Hall → Speed | `0xACF0`… | — | ★ next pure |
| ⬜ | `M-SPEED` | K4 | Continuous → Speed Booster | `0xAD1B` | — | `speed` |
| ⬜ | `M-WAVE` | K4 | Continuous → Wave Beam | `0xADDE` | — | `wave` |
| ⬜ | `M-ICE` | K4 | Continuous → Ice Beam | `0xA890` | — | `ice` |
| ⬜ | `M-ALPHAPB` | K5 | Continuous → Alpha Power Bombs | `0xA3AE` | — | `alpha_pb` |
| ⬜ | `M-MOAT` | K6 | Continuous → Moat clear | `0x95FF` | — | `moat` |
| ⬜ | `M-WS` | K6 | Continuous → Wrecked Ship entry | `0xCA08` | — | `ws` |
| ⬜ | `M-PHAN` | K6 | Continuous → Phantoon defeat | `0xCD13` | — | `phantoon` |
| ⬜ | `M-GRAV` | K6 | Continuous → Gravity Suit | `0xCE40` | — | `gravity` |
| ⬜ | `M-TUBE` | K7 | Continuous → Maridia tube break | `0xCEFB` | — | `tube` |
| ⬜ | `M-BOTW` | K7 | Continuous → Botwoon defeat | `0xD95E` | — | `botwoon` |
| ⬜ | `M-DRAY` | K7 | Continuous → Draygon defeat | `0xDA60` | — | `draygon` |
| ⬜ | `M-SJ` | K7 | Continuous → Space Jump | `0xD9AA` | — | `space_jump` |
| ⬜ | `M-LN` | K8 | Continuous → Lower Norfair entry | `0xB656` | — | `ln_entry` |
| ⬜ | `M-RIDLEY` | K8 | Continuous → Ridley defeat | `0xB32E` | — | `ridley` |
| ⬜ | `M-G4` | K9 | Continuous → G4 statues | `0xA66A` | — | `statues` |
| ⬜ | `M-TOURIAN` | K9 | Continuous → Tourian elev | `0xDAAE` | — | `tourian` |
| ⬜ | `M-MB` | K9 | Continuous → Mother Brain defeat | `0xDD58` | — | `mother_brain` |
| ⬜ | `M-ESCAPE` | K9 | Continuous → Escape + Landing | `0x91F8` | — | `escape` |
| ⬜ | `M-CREDITS` | K9 | Continuous → Ending/Credits | `0x91F8` | — | `credits` |

## Clean track (parallel — Bronze / Clean)

Intervention: **no** energy refill, **no** ammo refill. Same Bronze observation
(read-only RAM). Does **not** move the program M5/M8 assisted gate.

| | ID | Milestone | Room | Frames / score | CLI / card |
|--:|----|-----------|------|----------------|------------|
| ✅ | `C-INFRA` | Clean CLI + artifact isolation + integrity | — | unit tests | `SM-CLEAN-*` infra done |
| ✅ | `C-MORPH` | Continuous → Morph (**Clean**) | `0x9E9F` | **27074** | `start_to_morph_clean.json` |
| ▶ | `C-BOMBS` | Continuous → Bombs/Torizo (**Clean**) ★ | `0x9804` | missiles @ 27928/29690 | `SM-CLEAN-BOMBS` — BT = existing model |
| ⏸ | `C-SPORE` | Continuous → Spore (**Clean**) | `0x9DC7` | — | parked until C-BOMBS |
| ⏸ | `C-SUPERS` | Continuous → Supers (**Clean**) | `0x9B5B` | — | parked |

Process: [`CLEAN_TRACK.md`](../CLEAN_TRACK.md). Infra landed — `--clean` uses
`*_clean` stems and zero resource-write integrity.

## Practice + structure rollups

| | ID | Milestone | Score | Notes |
|--:|----|-----------|-------|-------|
| 🟨 | `P-EASY` | Room practice easy+standard ready | 62/108 | dual-track only |
| ⬜ | `P-TOUGH` | Room practice tough queue | 0/117 | after easy/standard |
| ⬜ | `P-LATE` | Room practice late_special+boss | 0/38 |  |
| 🟨 | `P-PATH` | Completion-path rooms continuous | 39/107 | play clearance |
| 🟨 | `A-TIPSPEC` | Data-driven continuous tips complete | — | hop extract open |
| 🟨 | `A-GRAPH` | Graph API typed path summary | — |  |
| 🟨 | `A-PARSE` | Session-scoped parse counters | — |  |
| ⬜ | `C-TRACK` | Clean early tip ladder | 0/3 open | infra → morph → BT |

## Progress (product milestones only)

- Continuous (assisted): **13**
- Next pure ready (assisted): **1** (Cathedral)
- Open (assisted product): **20**
- Clean track: **infra ✅ + morph ✅**; ★ bombs/Torizo ready (missiles clean green); spore+ parked
- KPDR tracker segments: **92** (`continuous`=41, `open`=50)
- Backlog tickets: **~308** (includes CLEAN + Cathedral pure stack)

## How to use

1. Pick the ★ tip from this table / `STATUS.md`.
2. Open the matching **ready** row in `BACKLOG.csv` (or live card under `docs/tasks/`).
3. Pure → graph → compose → stabilize → STATUS (never skip pure-first).
4. Dual-track room practice is parallel only — never continuous evidence.

Regenerate tracker summary: `uv run python super_metroid/scripts/export/kpdr_tracker.py`.

