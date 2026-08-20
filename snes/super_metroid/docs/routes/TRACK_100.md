# 100% Run Board — Super Metroid

**Status:** scaffold (P3 docs, parallel track)  
**Created:** 2026-08-03 via `SM-100-TRACK`  
**Seeded by:** Early Spazer epic (this board)

100% side notes (not the product tip). Tracker: `bd ready -l super_metroid`.  
Primary product: assisted **any% KPDR** full clear (M8 credits).  
100% is a **future track** — no continuous integrity contract exists yet.

## Definition (project 100%)

For this project, **100%** means:

| Category | Criterion |
|----------|-----------|
| **Major items** | All collectible upgrade items (beams, suits, boots, jump, PB, Speed, Charge, Grapple) |
| **E-Tanks** | All 14 Energy Tanks |
| **Ammo packs** | All missile, super missile, power bomb packs |
| **Reserve Tanks** | All 4 Reserve Tanks |
| **Map stations** | All 4 map station downloads (Crateria, Brinstar, Norfair, WS/Maridia) |
| **Bosses** | All 10 major bosses defeated (Spore Spawn, Bomb Torizo, Kraid, Phantoon, Botwoon, Draygon, Ridley, Crocomire, Golden Torizo, Mother Brain) |
| **Escape** | Natural endgame escape → Landing Site → ending/credits |

**Any% vs 100% divergence:** any% KPDR skips Spazer, Charge, Crocomire, Golden Torizo,
Grapple, Spring Ball, Screw Attack, Plasma, most pack collectibles, and map stations.
100% inserts detours for each skipped item and may re-order to minimize backtracking
(e.g. Spazer during K2 Red Brinstar pass, Charge return after Red Tower, Crocomire
during LN).

**Continuous fold policy:** optional secondary tips stay side-tips until pure+stabilize
green; then planner folds them into the default continuous spine. A 100% spine is a
**superset** of the any% KPDR spine — any% tips are not demoted.

## Item checklist

| # | Item | RAM | Continuous status | Epic / card | Notes |
|--:|------|-----|:-----------------:|-------------|-------|
| 1 | Morph Ball | `collected_items` 0x0004 | ✅ done | K0 continuous `--to morph` | |
| 2 | Bombs | `collected_items` 0x1000 | ✅ done | K0 continuous `--to bombs` | Bomb Torizo defeat |
| 3 | Missiles (first) | capacity ≥5 | ✅ done | K0 continuous | First missile pack |
| 4 | Supers (first) | capacity ≥5 | ✅ done | K0 continuous `--to supers` | Spore Super capacity 0→5 |
| 5 | Charge Beam | `beams` 0x10 → `0x1000` | 🔶 controller_dev | K1 optional; parked | Conventional return not route-ready; not on continuous K1 |
| 6 | Varia Suit | `collected_items` 0x0001 | ✅ done | K3 continuous `--to varia` | Natural Kraid fight + Varia PLM |
| 7 | Hi-Jump Boots | `collected_items` 0x0100 | ✅ done | K2 continuous `--to hijump` | Real PLM bit 0x0100; E-Tank collected |
| 8 | E-Tank (Hi-Jump room) | $7E:09C4 | ✅ done | K2 continuous | |
| 9 | E-Tank (Terminator room) | $7E:09C4 | ✅ done | K0 continuous | Terminator ET, spore path |
| 10 | Spazer Beam | `beams` 0x04 | 🔷 in progress | Early Spazer (this board) | Door/collect/return pure green; climb + top→West residual |
| 11 | Speed Booster | `collected_items` 0x2000 | ⬜ open | K4 `--to speed` | After Bubble→Bat→Speed Hall |
| 12 | Wave Beam | `beams` 0x01 | ⬜ open | K4 `--to wave` | After Speed; Single→Double chamber |
| 13 | Ice Beam | `beams` 0x02 | ⬜ open | K4 `--to ice` | Business→Ice Gate→Reflection |
| 14 | Alpha Power Bombs | `collected_items` / capacity | ⬜ open | K5 `--to alpha_pb` | First PB capacity; post-Ice Red Tower climb |
| 15 | Gravity Suit | `collected_items` 0x0020 | ⬜ open | K6 `--to gravity` | Post-Phantoon |
| 16 | Space Jump | `collected_items` 0x0200 | ⬜ open | K7 `--to space_jump` | Post-Draygon |
| 17 | Grapple Beam | `equipment` 0x01 ($7E:09B2) | ⬜ optional | `SM-OPT-GRAPPLE` parked | Skip by default; optional collect on 100% |
| 18 | Plasma Beam | `beams` 0x08 | ⬜ optional | `SM-OPT-PLASMA` parked | Post-Draygon side room; optional |
| 19 | Spring Ball | `collected_items` 0x0002 | ⬜ optional | — | Not on KPDR path; optional 100% |
| 20 | Screw Attack | `collected_items` 0x0008 | ⬜ optional | — | Not on KPDR path; optional 100% |
| 21 | Pink Power Bombs | `collected_items` / capacity | ⏸ parked | `SM-OPT-PINK-PB` | Hard pure maze; not KPDR; backfill for 100% |
| 22 | X-Ray Scope | `equipment` ($7E:09B4) | ⬜ optional | — | Not on KPDR path; optional 100% |

### Ammo packs (expansion count)

| Type | Game total | On KPDR spine | 100% target |
|------|:----------:|:-------------:|:-----------:|
| Missile packs | 46 (230 capacity) | ~5–8 | all |
| Super Missile packs | 10 (50 capacity) | ~2–4 | all |
| Power Bomb packs | 10 (50 capacity) | ~2–4 | all |

Exact counts to be filled after 100% route design. E-Tank total: **14** (2 already
on KPDR continuous: Terminator + Hi-Jump room).

### E-Tank location reference (full 14)

| # | Room | Area | On KPDR? |
|--:|------|------|:--------:|
| 1 | Terminator Room `0x9EAD` | Brinstar | ✅ K0 |
| 2 | Hi-Jump Room `0xA9E5` | Norfair | ✅ K2 |
| 3 | Green Brinstar (main shaft) | Brinstar | possible K1 |
| 4 | Green Brinstar (spike room) | Brinstar | — |
| 5 | Kraid Lair (before boss) | Brinstar | possible K2 |
| 6 | Wrecked Ship (E-Tank room) | WS | possible K6 |
| 7 | Maridia (Mt Everest) | Maridia | possible K7 |
| 8 | Maridia (Botwoon Hall) | Maridia | possible K7 |
| 9 | Maridia (Colosseum) | Maridia | possible K7 |
| 10 | Lower Norfair (Ridley path) | LN | possible K8 |
| 11 | Lower Norfair (acid) | LN | possible K8 |
| 12 | Lower Norfair (entrance) | LN | — |
| 13 | Brinstar (Charge/return path) | Brinstar | — |
| 14 | Crateria (revisit) | Crateria | — |

Full E-Tank location detail deferred to 100% route design.

## Map stations

| # | Station | Area | Room | On KPDR? | Continuous status |
|--:|---------|------|------|:--------:|:-----------------:|
| 1 | Crateria | Crateria | `0x93AA` | possible K1 | ⬜ open |
| 2 | Brinstar | Brinstar | `0x9B9D` | possible K1/early | ⬜ open |
| 3 | Norfair | Norfair | `0xA7A5` | possible K4 | ⬜ open |
| 4 | WS / Maridia | Wrecked Ship | `0xCC6F` | possible K6 | ⬜ open |

On any% KPDR, map stations are not collected. For 100%, each station download sets
a specific event bit; the download animation is ~2–3s per station.

## Boss order

**KPDR any%** (primary spine): Kraid → Phantoon → Draygon → Ridley → Mother Brain + escape.  
**100% extras** are either on-path (Spore Spawn, Bomb Torizo, Botwoon — already on
KPDR) or detours (Crocomire, Golden Torizo).

| # | Boss | Room | On KPDR? | Continuous status |
|--:|------|------|:--------:|:-----------------:|
| 1 | Spore Spawn | `0x9DC7` | ✅ K0 | ✅ continuous |
| 2 | Bomb Torizo | `0x9804` | ✅ K0 | ✅ continuous |
| 3 | Kraid | `0xA59F` | ✅ K2/K3 | ✅ continuous (entry green) |
| 4 | Crocomire | `0xA97D` | ⬜ optional | ⏸ parked (`SM-OPT-CROC`) |
| 5 | Phantoon | `0xCD13` | ✅ K6 | ⬜ open (dev warp only) |
| 6 | Botwoon | `0xD95E` | ✅ K7 | ⬜ open |
| 7 | Draygon | `0xDA60` | ✅ K7 | ⬜ open |
| 8 | Golden Torizo | `0xB3A5` | ⬜ optional | ⏸ parked (`SM-OPT-GT`) |
| 9 | Ridley | `0xB32E` | ✅ K8 | ⬜ open |
| 10 | Mother Brain | `0xDD58` | ✅ K9 | ⬜ open |

All bosses must be defeated (event bit set) for 100%. Crocomire and Golden Torizo
are not on the KPDR path; they require LN detours.

### Boss event reference

| Boss | Event bit | Notes |
|------|-----------|-------|
| Bomb Torizo | event 0x0A | Set when Torizo is defeated |
| Spore Spawn | event 0x0C | Set on defeat |
| Kraid | event 0x01 | Door opens after defeat |
| Phantoon | event 0x02 | Door opens after defeat |
| Botwoon | event 0x04 | Set on defeat |
| Draygon | event 0x05 | Door opens after defeat |
| Crocomire | event 0x06 | Set on defeat |
| Golden Torizo | event 0x07 | Set on defeat |
| Ridley | event 0x03 | Door opens after defeat |
| Mother Brain | event 0x0E | Escape sequence triggers |

## Insertion notes

### Spazer early (in progress)

**Ladder:** this board → `SM-SPAZER-*` work (`bd ready -l super_metroid`).

- First concrete 100% item insertion: Red Brinstar detour during K2.
- From continuous `--to below_spazer` (already green), insert:
  `Below Spazer (0xA408) → Spazer Room (0xA447) collect → Below Spazer (0xA408)`.
- Walljump may be needed for red-room shaft; reuse Bubble walljump patterns.
- **Human record ready:** [`EARLY_SPAZER_HUMAN.md`](../tasks/EARLY_SPAZER_HUMAN.md)
  (`guided_human --from below-spazer --route early-spazer`); guide path on same window.
- After pure green: graph edges → catalog tip `--to spazer` → dual integrity → fold.
- Priority: P2 parallel (does **not** block K4 Bubble Bat serial spine).

### Charge return (parked)

- Charge collect exists `controller_dev` from Big Pink Chozo pedestal.
- Conventional return through Big Pink (climb back up the right shaft) is **not**
  route-ready — parked until 100% route design separates K1 side-trip from the
  main Red Tower descent.
- When 100% activated: insert as a named side-tip after Red Tower, then fold.

### Pink Power Bombs (parked)

- Hard pure maze `0x9E11` — stalled on mid-maze walljump geometry.
- Parked: not on competitive KPDR. Backfill only for 100%.
- When 100% activated: may use later loadout (Speed + Hi-Jump) to simplify.

### E-Tank / pack depth

Only major items listed above. Full E-Tank + missile/super/PB pack detail deferred
to 100% route design — residual note `pack depth later` keeps this scaffold from
bloating to hundreds of rows.

## Continuous fold policy

| Rule | Detail |
|------|--------|
| any% KPDR remains primary | M8 assisted full clear stays the default tip (`--to bat_cave` → … → credits) |
| Secondary tips first | Each 100% insert (Spazer, Charge return, E-Tank detours, Croc, GT) is a **named continuous tip** before fold |
| Pure-first | Pure green from continuous-like source before graph/compose/stabilize |
| On-spine when pure+stab green | After dual integrity, planner folds the detour into the default spine |
| Superset, not replace | 100% spine adds rooms/items to any% KPDR; never removes or re-routes |
| Map stations | Inserted when area is passed; minimal detour |
| Boss extras | Crocomire (LN), Golden Torizo (LN) inserted during LN pass |
| Pack collection | Scattered across the run; not a separate "cleanup" phase |

**100% continuous tip name convention (proposed):** `--to 100_<item>` for side-tips,
then `--to full` / `--to credits_100` for the final folded spine. Exact CLI to be
designed when the first side-tip (Spazer) lands as a continuous tip.

## Relation to other docs

| Doc | Role |
|-----|------|
| [MILESTONES.md](MILESTONES.md) | any% KPDR + Clean milestones; 100% not yet tracked there |
| [ROUTE_KPDR.md](ROUTE_KPDR.md) | Authoritative any% KPDR spine (stays primary) |
| `bd ready -l super_metroid` | Ready / in-flight work |
| [KPDR_TRACKER.csv](KPDR_TRACKER.csv) | any% KPDR per-segment status |

## Status legend

| Mark | Status | Meaning |
|------|--------|---------|
| ✅ | `continuous` | Power-on chain integrity green |
| 🔶 | `controller_dev` | Pure controller green; not yet continuous |
| 🔷 | `in_progress` | Epic / card active (parallel track) |
| ⬜ | `open` | Not started |
| ⏸ | `parked` | Explicitly not on any% KPDR / deferred |
| ⬜ | `optional` | Skip on any%; collect on 100% |

## Checklist (100% track gates)

- [ ] Spazer detour pure green + continuous tip + fold (`SM-SPAZER-*` ladder)
- [ ] Charge return pure green + graph (optional insert)
- [ ] K4 Speed/Wave/Ice continuous (any% already; 100% same path)
- [ ] K5 Alpha PB continuous
- [ ] K6 Phantoon + Gravity continuous
- [ ] K7 Maridia + Draygon + SJ continuous
- [ ] K8 LN + Ridley continuous (+ Crocomire detour + Golden Torizo detour for 100%)
- [ ] K9 Tourian + MB + escape + credits continuous
- [ ] Map station inserts designed and green (4 stations)
- [ ] E-Tank pack depth: spider location for all 14 E-Tanks, all missile/super/PB packs
- [ ] 100% continuous spine assembled and dual integrity green
- [ ] STATUS M8 100% promote
