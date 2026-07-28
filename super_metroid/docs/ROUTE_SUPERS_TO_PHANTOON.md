# Working route: Super collect → Phantoon entry

Last updated: 2026-07-28.

Living checklist for the **any% ship path** after Spore Super Missiles.
This is the continuous Track B spine toward Phantoon — not the full research
hop table through Kraid/Norfair.

Labels:

| Label | Meaning |
|-------|---------|
| **continuous** | Power-on (or verified continuous prefix) + controller only; 0 state loads after boot; 0 progression writes |
| **dev controller** | Controller-only from a natural/dev save state; no door-warp during the segment |
| **dev warp** | Door-warp / place / loadout grant; topology only |
| **open** | Not working yet |

Integrity contract: [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md). Program facts:
[STATUS.md](STATUS.md). Plan phases: [plan.md](plan.md) (B1–B2).

---

## Target chain

```text
[continuous] power-on → Super collect (0x9B5B capacity 0→5)
       │
       ▼
 0x9B5B Super room ──door 0x8D1E──► 0xA0A4 Farming
       │
       ▼
 0xA0A4 Farming ──door 0x8F82 green Super──► 0x9D19 Big Pink (farm pocket)
       │
       ▼
 0x9D19 Big Pink ──door 0x8E02 bottom──► 0x9E11 Pink PB  → collect PB
       │
       ▼
 0x9E11 ──door 0x8E6E──► 0x9D19 ──door 0x8DEA green Super──► 0x9E52 GHZ
       │
       ▼
 0x9E52 GHZ ──► 0x9FBA Noob Bridge ──► 0xA253 Red Tower
       │
       ▼
 0xA253 ──door 0x901E──► Hellway 0xA2F7 ──► Caterpillar 0xA322
       │
       ▼
 elev 0x962A ──► Crateria Kihunter 0x948C ──► Moat 0x95FF
       │
       ▼
 West Ocean 0x93FE ──► WS Entrance 0xCA08 ──► WS Main 0xCAF6
       │
       ▼
 WS Basement 0xCC6F ──► Phantoon 0xCD13
```

Hop tables:

- Super → PB: `maps/full_route_hops.json` key `spore_spawn_supers__early_power_bombs`
- PB → Kraid research path: `early_power_bombs__kraid` (not this doc’s ship path)
- Red Tower → Phantoon warps: `phantoon_dev.SHIP_ROUTE`

---

## Segment board

### 0 — Continuous prefix (done)

| Field | Value |
|-------|-------|
| Status | **continuous** |
| Span | Power-on → Super Missile capacity 0→5 in `0x9B5B` |
| Frames | ~92,425 (`recordings/start_to_supers.{json,mp4}`) |
| Code | `start_to_supers.py` + `post_spore_controller.play_super_room_collect` |
| Integrity | 0 state loads; 0 progression writes |
| Next | Farm exit + Big Pink |

Reproduce (optional; long):

```bash
uv run python super_metroid/scripts/record_start_to_supers.py --no-video
```

---

### 1 — Super room bottom → Farming `0xA0A4`

| Field | Value |
|-------|-------|
| Status | **dev controller** |
| Entry | After Super collect, bottom of `0x9B5B` (~x=411, y=2187) |
| Exit | Farming settled, game state 8 |
| Door | `0x8D1E` (left blue after bomb gate) |
| Code | `post_spore_controller.play_super_room_to_farming` |
| Dev state | `dev_b1_farming_entry.state`, `dev_b1_supers_natural.state` |
| Continuous? | Not yet re-proven on full power-on suffix |
| Blockers | None known in isolation |

```bash
uv run python super_metroid/scripts/probe_post_spore_pb.py --to farming
```

---

### 2 — Farming → Big Pink farm pocket `0x9D19`

| Field | Value |
|-------|-------|
| Status | **dev controller** |
| Entry | Farming `0xA0A4` with Supers selected |
| Exit | Big Pink ~**(1240, 1413)** near farm door block `[79,87]` |
| Door | `0x8F82` (green, Super required) |
| Code | `post_spore_controller.play_farming_to_big_pink` |
| Dev state | `dev_b1_bigpink_entry.state` |
| Continuous? | Not yet |
| Blockers | None known in isolation |

```bash
uv run python super_metroid/scripts/probe_post_spore_pb.py --to big-pink
```

---

### 3a — Crest farm-pocket lip (partial — **dev controller**)

| Field | Value |
|-------|-------|
| Status | **dev controller** (2026-07-27) |
| Entry | Farm pocket ~x=1240, y=1419 |
| Method | Walk left into lip → run **right** for speed → spin-jump **left** over lip |
| Exit | ~**(1125, 1387)** mid ledge |
| Code | `post_spore_controller.play_big_pink_crest_pocket` |
| Probe | `probe_post_spore_pb.py --to crest` |

```bash
uv run python super_metroid/scripts/probe_post_spore_pb.py --to crest
```

### 3b — Crest ledge → open main shaft (**dev controller**)

| Field | Value |
|-------|-------|
| Status | **dev controller** (2026-07-27) |
| Entry | After crest ~x=1125, y=**1387** standing on raised platform |
| Exit | Open main-shaft volume x≲750 (e.g. ~(746, 1465)) |
| Continuous? | No (not yet on full power-on suffix) |
| Code | `play_big_pink_into_main_shaft` |
| Dev state | `dev_b1_bigpink_main_controller.state` |

**Sequence**

1. Crouch-Super clears permanent Super-only shot **(69, 87)** (`play_big_pink_clear_super_block`).
2. **Double-tap DOWN** morphs into the y87 tunnel (`play_big_pink_morph_to_tunnel`): standing y≈1387 and morph y≈1401 are the **same floor** (pose height). Hold-DOWN alone only crouches and cannot enter the 1-tile tunnel.
3. Morph-roll west + **X** bombs open scroll screen (3,5), clear bomb blocks (62–63, 87), reach x≲750 (`play_big_pink_tunnel_west`).

**Level-data (live WRAM)**

| Tile | Natural crest | After 3b | Meaning |
|------|---------------|----------|---------|
| (69, 87) | type `0xC` BTS `0x0B` | air | Permanent Super-only shot block |
| (63, 87) / (62, 87) | type `0xF` BTS `0x04` | air | Permanent bomb blocks |
| Scroll row y5 | `[0,0,2,0,1]` | `[0,0,2,1,1]` | Screen **(3,5)** via scroll PLM (64, 87) |

**Work items**

- [x] Crest lip x≈1157 (run-right + spin-jump-left)
- [x] Crouch-Super clear + double-tap morph + tunnel-west compose
- [x] `play_big_pink_into_main_shaft` green (no place/WRAM)
- [x] Save `dev_b1_bigpink_main_controller.state` from controller path
- [ ] Re-prove as continuous suffix after power-on → Super

```bash
uv run python super_metroid/scripts/probe_post_spore_pb.py --to super-block
uv run python super_metroid/scripts/probe_post_spore_pb.py --to tunnel-floor
uv run python super_metroid/scripts/probe_post_spore_pb.py --to main --save \
  super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_bigpink_main_controller.state
# Tunnel suffix only (already on floor, Super clear):
uv run python super_metroid/scripts/probe_post_spore_pb.py --to tunnel-west \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_big_pink_open.state
```

---

### 4 — Big Pink shaft → Pink PB door (partial — sill entry green)

| Field | Value |
|-------|-------|
| Status | **sill entry controller green**; approach from main shaft **open** |
| Target door | Bottom PB sill island **x≈574–582, y≈1136** (block `[32,71]`) bank `$83` door **`0x8E02`** → `0x9E11` |
| Alt door | Top doorway **y≈888** door **`0x8DDE`** — entry from place(560,888) works; gray on MI side |
| Geometry | Main platform x≥613 y≈1179; **full-height wall column** x≈590–610; jumps hard-stop min_x=613 (pose 138). Sill is a narrow island, not connected by floor. |
| Entry code | `play_big_pink_enter_pb_door_from_sill` — run+shoot+spin+hold left |
| Dev states | `dev_b1_intercept` (~688,1179 door-height platform); `dev_b1_pb_door_entered` after sill entry |
| Continuous? | No — still need pure path onto sill |
| Blockers | Approach onto sill island without place/WRAM |

**Work items**

- [x] Door sill geometry mapped; blue door opens with shot
- [x] `play_big_pink_enter_pb_door_from_sill` controller (on-sill → `0x9E11`)
- [ ] Pure climb/approach from main (`dev_b1_bigpink_main_controller` / intercept) onto sill
- [ ] Compose `play_big_pink_to_pb_door` without place bridges

```bash
# Sill entry only (expects already on sill — use place for dev):
uv run python super_metroid/scripts/probe_post_spore_pb.py --to pb-door \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_intercept.state
```

---

### 5 — Pink PB room collect `0x9E11` (partial — wall@437 pure; pocket collect green)

| Field | Value |
|-------|-------|
| Status | **wall@437 pure morph-bomb**; **collect green** from x≤225; mid-maze 405→225 **open** |
| Item | Power Bomb PLM ≈ **(100–120, 370–395)** |
| Entry spawn | Bottom door lands ~**(460–472, 395)** facing metal maze |
| Maze wall@437 | **Bombable** with **double-tap morph** (release between DOWN). Crouch-only fails. Missiles/supers no. |
| Wall code | `play_pink_pb_break_maze_wall` → ~**(408, 398)** |
| Collect pocket | x≤**225** y≈395 (x≥230 stuck/solid on this floor) |
| Collect code | `play_pink_pb_morph_bomb_collect` morph-bomb-roll → pb 5/5 |
| Place bridge | reduced **150 → 220** for mid-maze gap only |
| Evidence | `--to pb-maze-wall` / `--to pb-collect` from `dev_b1_pb_door_entered` |
| Continuous? | Needs pure 405→225 then collect; sill approach still open |
| Blockers | Mid-maze after wall (pit y≈457 traps morph; ceiling not IBJ-clearable yet) |

```bash
# Pure wall open (no place once in 0x9E11):
uv run python super_metroid/scripts/probe_post_spore_pb.py --to pb-maze-wall \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_pb_door_entered.state
# Wall pure + place(220,395) + collect:
uv run python super_metroid/scripts/probe_post_spore_pb.py --to pb-collect \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_pb_door_entered.state
uv run python super_metroid/scripts/probe_phantoon.py collect-pb  # warp+place baseline
```

---

### 6 — PB → GHZ → Noob → Red Tower

| Field | Value |
|-------|-------|
| Status | **open** (controller); topology known |
| Path | `0x9E11` → Big Pink → green Super door block `[63,103]` ≈ (1016,1656) → `0x9E52` GHZ → `0x9FBA` Noob Bridge → `0xA253` Red Tower |
| Door (BP→GHZ) | `0x8DEA` (Super) |
| Dev state | `dev_red_tower_stable.state` (Red Tower; may lack PB capacity) |
| Continuous? | No |
| Blockers | Full natural traversal after PB; GHZ enemies/geometry; Noob bridge |

Hop reference: first hops of `early_power_bombs__kraid` through Red Tower, then diverge to ship route (not Kraid).

---

### 7 — Red Tower → Hellway → Caterpillar → elev → Crateria

| Field | Value |
|-------|-------|
| Status | **dev warp** |
| Code | `phantoon_dev.door_warp_ship_route` first hops |
| Doors | `0x901E` Hellway, `0x908A` Caterpillar, `0x90BA` elev, `0x8AF6` Crateria Kihunter |
| Continuous? | No — needs Red Tower climb/nav policy |
| Blockers | Red Tower vertical nav; elevator touch |

---

### 8 — Crateria Kihunter → Moat → West Ocean → WS

| Field | Value |
|-------|-------|
| Status | **dev warp** |
| Doors | `0x8A36` Moat, `0x8AEA` West Ocean, `0x89D6` WS Entrance |
| Continuous? | No |
| Blockers | Moat water gap; West Ocean length; WS entry |

---

### 9 — Wrecked Ship → Phantoon room

| Field | Value |
|-------|-------|
| Status | **dev warp** for entry; **fight open** |
| Path | WS Entrance → Main `0xCAF6` → Basement `0xCC6F` → Phantoon `0xCD13` |
| Doors | `0xA1BC`, `0xA21C`, `0xA2AC` |
| Dev state | `dev_phantoon_entry.state` |
| Fight | `phantoon_dev.run_phantoon_fight` — open-eye damage not reliable |
| Continuous? | Entry no; fight no |

```bash
uv run python super_metroid/scripts/probe_phantoon.py capture-entry
uv run python super_metroid/scripts/probe_phantoon.py ship-route
```

---

## Progress snapshot

| # | Segment | Continuous | Dev controller | Dev warp |
|---|---------|:----------:|:--------------:|:--------:|
| 0 | Power-on → Super collect | **yes** | yes | — |
| 1 | Super bottom → Farming | — | **yes** | — |
| 2 | Farming → Big Pink pocket | — | **yes** | — |
| 3a | Crest pocket lip (→~1125 wall-top) | — | **yes** | — |
| 3b | Super block + morph + tunnel → main | — | **dev controller** | `dev_b1_bigpink_main_controller` |
| 4 | Shaft → PB door | — | sill entry yes; approach **open** | place sill |
| 5 | PB maze wall@437 | — | **pure break** | — |
| 5b | PB mid-maze → collect | — | collect from ≤225; 405→225 **open** | place 220 |
| 6 | PB → Red Tower (GHZ/Noob) | — | **open** | partial states |
| 7 | Red Tower → Crateria elev | — | **open** | **yes** |
| 8 | Moat / Ocean / WS entry | — | **open** | **yes** |
| 9 | WS → Phantoon room | — | **open** | **yes** |
| 10 | Phantoon fight | — | **open** | entry state |

**Bottleneck for continuous growth:** segment **4** (climb/door to Pink PB) then natural collect; topology through Phantoon *room* is door-warp proven after PB.

---

## Immediate next actions (ordered)

1. **Crack 4** — pure approach onto sill (or y1051 ledge → drop): wall@613 blocks
   horizontal; upper ledge exists (`dev_b1_left_upper` ~(597,1051)).
2. **Crack 5b** — pure mid-maze 405 → x≤225 (avoid pit y≈457 / open ceiling).
3. **Compose** `play_post_spore_to_pb` (sill + maze + collect) from natural Super.
4. **Continuous dry** power-on → PB.
5. **Segment 6** GHZ / Noob / Red Tower controller from post-PB Big Pink.

Do **not** polish Phantoon fight until segments 4–6 are controller-proven.

---

## Key coordinates (Big Pink `0x9D19`)

| Landmark | Block | Approx px |
|----------|-------|-----------|
| Farm door (from `0xA0A4`) | [79, 87] | (1272, 1400) |
| Pocket wall (stuck x) | — | x≈**1157**, y≈1419 |
| PB bottom door | [32, 71] | (**520**, **1144**) |
| PB top doorway | [32, 55] | (520, 888) |
| GHZ green Super door | [63, 103] | (1016, 1656) |
| Charge Beam Chozo | [37, 118] | (600, 1896) |

Pink PB room `0x9E11`: item [6, 23] ≈ (104, 376).

---

## Code / state index

| Artifact | Role |
|----------|------|
| `post_spore_controller.py` | Super collect → farming → Big Pink crest → main shaft |
| `phantoon_dev.py` | PB warp-collect, `SHIP_ROUTE`, Phantoon entry/fight |
| `scripts/probe_post_spore_pb.py` | Dev probe through crest / tunnel-floor / main |
| `scripts/probe_phantoon.py` | collect-pb / capture-entry / ship-route |
| `natural_post_spore_spawn.state` | Natural Super-room entry (no Supers) |
| `dev_b1_bigpink_entry.state` | Controller farm→Big Pink landing |
| `dev_b1_crest.state` | Crest standing (~1125,1387) |
| `dev_b1_crest_S_clear.state` | Crest after Super shot block cleared |
| `dev_b1_bigpink_main_controller.state` | Main shaft via pure controller (no place) |
| `dev_big_pink_open.state` | Raised tunnel floor morph (~1125,1401) |
| `dev_big_pink_main.state` | Open shaft reference (blocks already clear) |
| `dev_b1_pink_shaft_low.state` | Low shaft for climb work |
| `dev_power_bombs_collected.state` | Warp-collected PB |
| `dev_phantoon_entry.state` | Warp ship route end |
---

## Update rule

When a segment changes status, edit **this file first**, then one-line
`STATUS.md` “Next milestone” / gap table, and `plan.md` B1 checkboxes.
Keep continuous claims only when a power-on (or full prefix) report exists.
