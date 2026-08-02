# Milestones — Super Metroid assisted full clear

Machine source: [`MILESTONES.csv`](MILESTONES.csv) · backlog: [`BACKLOG.csv`](BACKLOG.csv) · spine: [`KPDR_TRACKER.csv`](KPDR_TRACKER.csv).

**Goal:** one continuous power-on → ending/credits run with **only** unlimited energy + ammo ([`ASSIST_CONTRACT.md`](../ASSIST_CONTRACT.md)). Target runtime class: **less-than-Bronze** assisted full clear (M8).

**Current tip:** Frog Savestation (`--to frog`, **114,923f** ×2 integrity green). ★ Next: Frog Save → Speedway pure.

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
| ✅ | `M-MORPH` | K0 | Continuous → Morph | `0x9E9F` | prefix | `morph` |
| ✅ | `M-BOMBS` | K0 | Continuous → Bombs/Torizo | `0x9804` | prefix | `bombs` |
| ✅ | `M-SPORE` | K0 | Continuous → Spore exit | `0x9DC7` | prefix | `spore` |
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
| ▶ | `M-SPEEDWAY` | K4 | Pure Frog Save → Speedway | `0xB106` | — | `SM-K4-SPEEDWAY-PURE` |
| ⬜ | `M-BUBBLE` | K4 | Pure → Bubble Mountain | `0xACB3` | — | `—` |
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

## Progress (product milestones only)

- Continuous: **13**
- Next pure ready: **1**
- Open: **20**
- KPDR tracker segments: **92** (`continuous`=41, `open`=50)
- Backlog tickets: **288** (open=275, ready=1, done=5)

## How to use

1. Pick the ★ tip from this table / `STATUS.md`.
2. Open the matching **ready** row in `BACKLOG.csv` (or live card under `docs/tasks/`).
3. Pure → graph → compose → stabilize → STATUS (never skip pure-first).
4. Dual-track room practice is parallel only — never continuous evidence.

Regenerate tracker summary: `uv run python super_metroid/scripts/export/kpdr_tracker.py`.

