# Route plan: Any% KPDR (project continuous spine)

Last updated: 2026-07-29.

**Boss order:** Kraid → Phantoon → Draygon → Ridley (then Mother Brain / escape).

This is the **chosen continuous product route** for assisted full clear.
Topology door-warps and the older ship-first skip remain development tools only.

Integrity: [ASSIST_CONTRACT.md](../ASSIST_CONTRACT.md). Facts: [STATUS.md](../STATUS.md).
Plan phases: [plan.md](../plan.md). Room board: [PATH_ROOM_BOARD.md](../research/PATH_ROOM_BOARD.md).

---

## Why KPDR (and why not ship-first)

| Route | Boss order | When to use |
|-------|------------|-------------|
| **KPDR** (this doc) | K → P → D → R | Default continuous spine; easier loadout for heat / ship / Maridia |
| **PRKD** | P → R → K → D | Faster any% meta; harder Moat/WS without Speed; **not** our continuous target |
| Ship-first skip | P before K | Old dev bridge (`phantoon.py skip-to-red` / `ship-route`); **not** continuous evidence |

**Why the project was briefly on Phantoon without Hi-Jump**

- Pure Pink PB mid-maze stalled; a **dev skip** granted PB and warped GHZ→Red Tower→ship topology.
- That is useful for fight/entry probes, but it is **not** KPDR and not continuous.
- On real KPDR, **Hi-Jump comes after Kraid** in competitive routes or
  **before Kraid** for beginner safety. This project now uses the safer
  Hi-Jump-first order. Power Bombs are usually **Alpha PB after Norfair**, not
  Pink Brinstar Mission Impossible.

**Hop table note:** `maps/full_route_hops.json` is a *capability-aware shortest-ish completion path* (early Pink PB, then Kraid, Speed, Ice, ship). It is **not** a human KPDR walkthrough. Prefer this doc for *what to play next*; use hop tables for door topology only.

---

## Project KPDR variant (assisted)

Assists = unlimited **energy + ammo** only (no free items/doors/bosses).

Design goals: no mockball required, no Zeb skip required, strong movement items before hard geometry.

| Choice | Decision | Rationale |
|--------|----------|-----------|
| Supers | **Spore Super** (already continuous) | Avoid Early Supers mockball (`0x9BC8`) — project tests deliberately do not require it |
| Charge Beam | **Collect** (dev → continuous) | Safer bosses; room is on the way through Big Pink |
| Spazer | **Optional safety** after Red Tower | Easier Kraid/Phantoon/MB; skip later if spray is solid |
| First Power Bombs | **Alpha PB `0xA3AE`** after Ice (or after Speed if Ice delayed) | Standard KPDR; **park pure Pink PB** as optional backfill only |
| Hi-Jump | **Before Kraid** | Safer Warehouse climb and Kraid approach; time loss is acceptable for controller reliability |
| Ice | **Collect before Alpha PB / ship** | Safety for Red Tower climb + Phantoon strats; ~10s slower in speedruns, fine for automation |
| Wave | **Collect with Speed loop** | Standard KPDR |
| Grapple / Croc | **Skip** (default) | Not needed with Speed + Hi-Jump + assists; beginner route optional |
| Pink PB `0x9E11` | **Parked** | Not on competitive KPDR; hard pure maze |

```text
[continuous] power-on → Spore Super collect 0x9B5B
       │
       ▼
 Charge (Big Pink) → GHZ → Noob → Red Tower
       │
       ▼
 Warehouse → Business Center → Hi-Jump boots (+ E-Tank)
       │
       ▼
 Back to Warehouse → Spazer? → Kraid → Varia
       │
       ▼
 Bubble Mountain → Speed Booster → Wave Beam → Ice Beam
       │
       ▼
 Elev up → Alpha Power Bombs → Crateria elev
       │
       ▼
 Moat → West Ocean → WS → Phantoon → Gravity
       │
       ▼
 Maridia tube → Botwoon → Draygon → Space Jump (± Plasma)
       │
       ▼
 Lower Norfair → Ridley → G4 statues → Tourian → MB → Escape
```

---

## Segment board (play order)

Labels: **continuous** | **dev controller** | **dev warp** | **open**.

### K0 — Power-on → Spore Super (done)

| Field | Value |
|-------|-------|
| Status | **continuous** |
| Span | Ceres → Morph → Missiles → Bombs/Torizo → Terminator ET → Spore → Super capacity 0→5 |
| Evidence | `recordings/start_to_supers.json` (prefix of red-tower run) |
| Walkthrough | Project: [START_TO_SPORE_SPAWN.md](START_TO_SPORE_SPAWN.md); wiki KPDR Crateria/Blue Brinstar |

---

### K1 — Super exit → main shaft → GHZ → Noob → Red Tower (done continuous)

| Field | Value |
|-------|-------|
| Status | **continuous** (Charge return still open as optional side trip) |
| Rooms | `0x9B5B` → `0xA0A4` → `0x9D19` main → `0x9E52` GHZ → `0x9FBA` Noob → `0xA253` Red Tower |
| Evidence | `recordings/start_to_red_tower.json` (**80,445** frames, integrity green) |
| Code | `run_to("red_tower")` / `play_start_to_red_tower` / `routes/kpdr/` Super→farm→Big Pink main→GHZ→Noob→Red |
| Parked | Pink PB door/maze (`0x9E11`); Charge conventional return |
| Walkthrough | Wiki KPDR “Green/Pink/Red Brinstar & Kraid’s Lair” (through Red Tower) |

```bash
uv run python super_metroid/scripts/probe/post_spore_pb.py --to main
uv run python super_metroid/scripts/probe/kpdr.py pure big-pink-to-ghz \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_bigpink_main_controller.state
uv run python super_metroid/scripts/probe/kpdr.py pure ghz-to-noob \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_kpdr_ghz.state
uv run python super_metroid/scripts/probe/kpdr.py pure noob-to-red \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_kpdr_noob.state
```

---

### K2 — Red Tower → Hi-Jump → Warehouse → Kraid entry

| Field | Value |
|-------|-------|
| Status | **continuous through natural Kraid entry (K2.18)** |
| Path | Red Tower → Bat `0xA3DD` → Below Spazer `0xA408` → tunnels → Warehouse `0xA6A1` → Business `0xA7DE` → Hi-Jump Shaft `0xAA41` → Hi-Jump Room `0xA9E5` → collect → reverse to Warehouse → Zeela → Kihunter → Baby Kraid → Eye Door → Kraid `0xA59F` |
| Continuous evidence | `recordings/start_to_kraid.json` (**97,170** frames, integrity green); prefixes: Hi-Jump 87,696f; Warehouse 83,512f; Bat 81,652f |
| Item evidence | Hi-Jump E-Tank and Boots are collected from their real PLMs; Boots set item bit `0x0100` |
| Return technique | Intended Hi-Jump ledges in the left shaft, then ordinary bombs through the top morph tunnel; **no infinite bomb jumps** |
| Warehouse technique | After returning with Hi-Jump, crouch/stand/tiny-hop Supers open the three-block wall and one Hi-Jump reaches the upper-right Zeela door |
| Walkthrough | [Hi Jump Boots Room](https://wiki.supermetroid.run/Hi_Jump_Boots_Room); Wiki KPDR Kraid section; room pages Warehouse / Baby Kraid / Kraid |

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure red-to-warehouse \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/dev_noob_to_red_controller.state
uv run python super_metroid/scripts/probe/kpdr.py pure warehouse-hijump-kraid \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/red_to_warehouse_controller.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/warehouse_hijump_kraid_composed.state
```

---

### K3 — Kraid fight → Varia

| Field | Value |
|-------|-------|
| Status | **continuous** (KPDR K3 tip) |
| Path | Kraid room `0xA59F` → Super-spray fight → rear door → Varia Room `0xA6E2` → real Varia PLM |
| Continuous evidence | `recordings/start_to_varia.json` (**101,954** frames, integrity green; 0 loads / 0 progression writes) |
| Code | `combat/kraid.py` (`play_kraid_fight_to_varia`); KPDR segment `kraid_entry_to_varia`; `run_to("varia")` |
| Probe | `scripts/probe/kraid_combat.py varia --state entry` → `debug/kraid_varia_run.json` |
| ★ Next | K4 forward: Frog Save → Speedway → Bubble → Speed/Wave/Ice |
| Walkthrough | Wiki Kraid fight and Varia Suit room pages |

---

### K4 — Speed Booster + Wave + Ice

| Field | Value |
|-------|-------|
| Status | **continuous through Frog Savestation**; K4 forward scaffold remains open |
| Path | Varia → Kraid return → … → Business → Frog Save → Speedway / farming → Bubble Mountain → Bat Cave → Speed Hall → Speed Room; Wave branch; Ice branch from Business |
| Graph | `START_TO_SPEED_GRAPH` — Varia return + Business→Frog are **continuous**; Speedway→Speed + Wave/Ice remain `unverified` |
| Continuous evidence | `recordings/start_to_frog_save*.json` (**114,923f** twice; 0 loads / progression / capacity / deaths) |
| First open hop | `play_frog_save_to_speedway` from `post_frog_continuous` |
| Rooms (hop-table subset) | `0xACB3` Bubble, `0xAD1B` Speed, `0xADDE` Wave, `0xA890` Ice |
| Walkthrough | Wiki KPDR “Norfair & Red Brinstar”; Bubble Mountain walljump / Speed / Wave / Ice pages |

---

### K5 — Alpha Power Bombs → Crateria elev

| Field | Value |
|-------|-------|
| Status | **open** |
| Room | Alpha Power Bomb Room `0xA3AE` (Red Brinstar, top of Red Tower climb after elev up from Norfair) |
| Note | Replaces early Pink PB as first PB capacity for continuous KPDR |
| Walkthrough | [Alpha Power Bomb Room](https://wiki.supermetroid.run/Alpha_Power_Bomb_Room) |

---

### K6 — Moat → West Ocean → Wrecked Ship → Phantoon → Gravity

| Field | Value |
|-------|-------|
| Status | Entry **dev warp** only (`dev_phantoon_entry`); fight open; pure rooms open |
| Path | Elev → Crateria Kihunter → Moat `0x95FF` → West Ocean → WS → Phantoon `0xCD13` → Gravity `0xCE40` |
| Dev tools | `phantoon.py ship-route` / `capture-entry` / `fight` (**not** continuous) |
| Walkthrough | [The Moat](https://wiki.supermetroid.run/The_Moat), [Phantoon](https://wiki.supermetroid.run/Phantoon), [Gravity Suit](https://wiki.supermetroid.run/Gravity_Suit) |

With Speed + Hi-Jump, Moat is shinespark or platform jumps (unequip Hi-Jump if platforming). Continuous walljump is PRKD-era; not required on KPDR with Speed.

---

### K7 — Maridia → Botwoon → Draygon → Space Jump

| Field | Value |
|-------|-------|
| Status | **open** |
| Path | Glass tube PB → Main Street → Everest / Crab / Aqueduct → Botwoon → Halfie → Colosseum → Draygon → Space Jump (± Plasma) |
| Walkthrough | Wiki KPDR Maridia section; Botwoon / Draygon room pages |

---

### K8 — Lower Norfair → Ridley

| Field | Value |
|-------|-------|
| Status | **open** (Ridley entry dev state exists on late route) |
| Walkthrough | Wiki KPDR Lower Norfair; [Ridley](https://wiki.supermetroid.run/Ridley) |

---

### K9 — G4 → Tourian → Mother Brain → Escape → ship

| Field | Value |
|-------|-------|
| Status | **open** (MB entry/spray probes only) |
| Walkthrough | Wiki KPDR Tourian & Escape; project endgame notes in [STATUS.md](../STATUS.md) |

---

## Progress snapshot vs KPDR

| Seg | Name | Continuous | Controller | Notes |
|-----|------|:----------:|:----------:|-------|
| K0 | → Spore Super | **yes** | yes | Continuous |
| K1 | Charge / GHZ / Noob / Red Tower | **yes** | yes | Direct Big Pink→Red; Charge return optional |
| K2 | Hi-Jump + natural Kraid entry | **yes** | **yes** | Continuous through Kraid entry |
| K3 | Kraid fight + Varia return → Business | **yes** | yes | Continuous Business tip 113,723f ×2 |
| K4 | Frog / Speed / Wave / Ice | **Frog Save** | partial | First open natural hop is Frog Save→Speedway |
| K5 | Alpha PB | — | — | Preferred first PB |
| K6 | Ship / Phantoon / Gravity | — | — | Warp entry only |
| K7–K9 | Maridia → Ridley → MB | — | — | |

Immediate played-spine queue:

1. **Continuous through Frog Save done** (`start_to_frog_save`, 114,923f integrity green twice).
2. **K4 forward continues:** `START_TO_SPEED_GRAPH`; Frog Save→Speedway, then Bubble → Speed.
3. **Separate K1 gap:** Charge Beam conventional return (optional; not on
   continuous K1).
4. Then K4 continuous tip → K5 Alpha PB → K6 ship / Phantoon.

**Tracker (chartable):** [KPDR_TRACKER.csv](KPDR_TRACKER.csv) ·
[KPDR_TRACKER.md](KPDR_TRACKER.md) · `maps/kpdr_tracker.json`  
Export: `uv run python super_metroid/scripts/export/kpdr_tracker.py`

Do **not** treat ship-first `ship-route` as the continuous next hop.

---

## External walkthroughs (KPDR)

Canonical hub: [Any% (KPDR section)](https://wiki.supermetroid.run/Any%25#KPDR) · [Tutorials index](https://wiki.supermetroid.run/Tutorials)

### Written route outlines

| Resource | Level | Notes |
|----------|-------|-------|
| [Any% → KPDR](https://wiki.supermetroid.run/Any%25#KPDR) | Intermediate | Full room-by-room text; Early Supers mockball + Alpha PB |
| [Beginners Route (UNHchabo)](https://wiki.supermetroid.run/Beginners_Route) | Beginner | Spore Super (no mockball), Hi-Jump **before** Kraid, Grapple/Croc safeties — closest **item philosophy** to assisted automation |
| Room pages on wiki | Per-room | Strat variants (Moat, Bubble Mountain, Phantoon, etc.) |

### Video / playlist tutorials

| Resource | Level | Notes |
|----------|-------|-------|
| [Mishrak](https://www.twitch.tv/mishrak109/v/69192497) | Beginner+ | Compact essential any% KPDR |
| [Scott Falco / Shaving Seconds](https://www.youtube.com/watch?v=8eScs50y110) | Beginner+ | Concise Mishrak-like route |
| [Melonax collection](https://www.twitch.tv/collections/_6FqA5izdhUfQg) | New → sub-hour | Progressive tutorials |
| [Popplars P1](https://www.twitch.tv/videos/753046491) Ceres→Speed · [P2](https://www.twitch.tv/videos/753041899) Speed→Space · [P3](https://www.twitch.tv/videos/753037132) Space→done | Mid | Segmented |
| [ShinyZeni KPDR playlist](https://www.youtube.com/playlist?list=PLdtrbA7NvFa2hvWq6HWvHazr1_woT0lcc) | Advanced | Modern optimal strats |
| [Zoast](https://www.twitch.tv/zoasty/v/46317411) ([timestamps](https://pastebin.com/11RbKe3L)) | Advanced | Strat encyclopedia |
| [Sweetnumb & Oatsngoats](https://www.youtube.com/watch?v=SA9JK6IE24g) | Advanced | Dual commentary |
| [ChTPwner playlist](https://www.youtube.com/playlist?list=PLZDC6DZTqL4w1tZsHH-W8PtXiO4QNG5jN) | FR beginner | Sub-60 minded |

### Project mapping tip

When watching a tutorial that uses **Early Supers mockball**, substitute our **Spore Super continuous prefix** and rejoin at Big Pink / GHZ. When a tutorial uses **Pink PB early**, prefer **Alpha PB after Ice** unless we later need Pink PB for a specific shortcut.

---

## Relation to other route docs

| Doc | Role after this plan |
|-----|----------------------|
| [ROUTE_KPDR.md](ROUTE_KPDR.md) (this file) | **Authoritative continuous spine** |
| [ROUTE_SUPERS_TO_PHANTOON.md](ROUTE_SUPERS_TO_PHANTOON.md) | Historical ship-first / Pink PB work; keep for PB maze notes + GHZ geometry |
| [START_TO_*.md](START_TO_SPORE_SPAWN.md) | Verified early continuous segments |
| `maps/full_route_hops.json` | Door topology for Track A warps only |

---

## Dev anchors to keep vs de-emphasize

| Anchor | Keep for |
|--------|----------|
| `natural_post_spore_spawn` / start_to_supers | Continuous prefix |
| `dev_b1_bigpink_main_controller` | K1 continuation |
| `dev_kraid_*` / `dev_varia_*` | K2 fight/entry |
| `dev_phantoon_entry` | K6 fight prototype only |
| `dev_b1_red_tower_post_pb` / skip-to-red | Optional PB+loadout sandbox — **label developmentOnly** |

---

## Checklist (implementation)

- [x] Choose KPDR as continuous spine (this doc)
- [ ] K1: Charge collect controller
- [x] K1: GHZ + Noob + Red Tower entry by controller-only play
- [x] K1: natural Big Pink approach to GHZ
- [x] K1: continuous power-on → Red Tower
- [x] K2.0: continuous power-on → Bat Room (Red Tower descent)
- [x] K2.1: continuous power-on → Below Spazer (Bat platforms)
- [x] K2: Red Tower → Warehouse Entrance controller-only
- [x] K2: Warehouse → Hi-Jump E-Tank + Boots from real PLMs
- [x] K2: Hi-Jump → Warehouse return with ordinary jumps/bombs; no IBJ
- [x] K2: Warehouse → Kraid natural entry
- [ ] K2: continuous Below Spazer → Warehouse → Hi-Jump → Kraid
- [ ] K3: Kraid fight + rear door + Varia PLM
- [ ] K4: Speed / Wave / Ice
- [ ] K5: Alpha PB collect
- [ ] K6: Moat / Ocean / WS / Phantoon / Gravity by play
- [ ] K7–K9: Maridia → Ridley → Tourian / escape
- [ ] Promote each remaining K2 segment to continuous suffix
- [ ] Optionally regenerate path board notes to label KPDR order vs hop-table order
