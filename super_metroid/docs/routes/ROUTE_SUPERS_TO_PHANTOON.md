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

Integrity contract: [ASSIST_CONTRACT.md](../ASSIST_CONTRACT.md). Program facts:
[STATUS.md](../STATUS.md). Plan phases: [plan.md](../plan.md) (B1–B2).

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
| Code | `routes/continuous.py` + `post_spore_controller.play_super_room_collect` |
| Integrity | 0 state loads; 0 progression writes |
| Next | Farm exit + Big Pink |

Reproduce (optional; long):

```bash
uv run python super_metroid/scripts/record/start_to_supers.py --no-video
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
uv run python super_metroid/scripts/probe/post_spore_pb.py --to farming
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
uv run python super_metroid/scripts/probe/post_spore_pb.py --to big-pink
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
uv run python super_metroid/scripts/probe/post_spore_pb.py --to crest
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
uv run python super_metroid/scripts/probe/post_spore_pb.py --to super-block
uv run python super_metroid/scripts/probe/post_spore_pb.py --to tunnel-floor
uv run python super_metroid/scripts/probe/post_spore_pb.py --to main --save \
  super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_bigpink_main_controller.state
# Tunnel suffix only (already on floor, Super clear):
uv run python super_metroid/scripts/probe/post_spore_pb.py --to tunnel-west \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_big_pink_open.state
```

---

### 4 — Big Pink shaft → Pink PB door (partial — entry green; pure approach open)

| Field | Value |
|-------|-------|
| Status | **door entry controller green** (top + bottom place); **pure approach open** |
| Preferred door | **Top** ``0x8DDE`` solid ledge **x≈520–548, y≈907** → ``0x9E11`` top spawn ~y=130 |
| Alt door | Bottom ``0x8E02`` place midair ~(580,1136) → bottom spawn ~y=395 |
| Entry code | `play_big_pink_enter_pb_door_from_sill` / `play_big_pink_enter_pb_door_from_top_ledge` |
| Dev states | `dev_b1_intercept`; `dev_b1_left_upper` ~(597,1051); `dev_b1_upper_floor` ~(798,1051); `dev_b1_pb_top_ledge` (post-land on y907); `dev_b1_pb_door_entered` |
| Continuous? | No — pure path into drop-air / onto y907 still open |
| Probe maps | `debug/post_spore/sill/` (BigPink.png, path crops, landing shots) |

**Geometry locked (2026-07-28/29 probes)**

| Region | Fact |
|--------|------|
| Main platform | x≥613, y≈1179; **full-height wall** min_x=613 (scroll unlock does **not** free it) |
| Upper ledge y≈1051 | Continuous x≈549–613 (`left_upper` / `left_ledge_end` / `upper_floor`); east fall → main, not door |
| y1051 vs bottom door | Upper ledge is corridor **roof** — cannot drop through onto bottom alcove |
| Bottom “sill island” | place/spin **door-zone**, not a hop-to platform from main/upper |
| **Top ledge y907** | Real solid; run-shoot-spin enters; spawn ~y=130 |
| **Drop-air → ledge** | Free-fall from place **x∈[535,555], y∈[850,910]** lands ~(532–544,907) then entry green (saved `dev_b1_pb_top_ledge`) |
| Approach from east | `peak940` / `upper_floor` spin: wall@613 still min_x=613 even at height; upper_floor min_x_air in y900–980 band ≈744 |
| Approach from west | `left_upper` already west of wall; standing jump Δy≈24, spin Δy≈79 — short of drop-air y≤910; naive walljump/bomb-jump did not chain height |
| False platforms | (600,907)/(600,920) place pockets — stuck pose 138, not walkable to door |

**Work items**

- [x] Bottom place-bridge entry → `0x9E11`
- [x] Top solid ledge y907 mapped; entry green (preferred for Mission Impossible)
- [x] Drop-air band mapped (place x535–555 y850–910 → land y907 → entry)
- [x] Upper ledge states + `dev_b1_pb_top_ledge`
- [ ] Pure controller into drop-air (or onto y907) from main / upper_floor / left_upper
- [ ] Compose `play_big_pink_to_pb_door` without place bridges

```bash
# Top door (preferred; solid ledge — place until pure climb exists):
uv run python super_metroid/scripts/probe/post_spore_pb.py --to pb-top-door --allow-place \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_upper_floor.state
# From saved ledge (no place if already on y907):
uv run python super_metroid/scripts/probe/post_spore_pb.py --to pb-top-door \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_pb_top_ledge.state
# Bottom sill place-bridge:
uv run python super_metroid/scripts/probe/post_spore_pb.py --to pb-door --allow-place \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_intercept.state
```

---

### 5 — Pink PB room collect `0x9E11` (partial — wall pure; left-zone→collect green)

| Field | Value |
|-------|-------|
| Status | **wall@437 pure**; **left-zone→collect green**; door→left-volume **open** |
| Alias | “**Mission Impossible Room**” ([wiki.supermetroid.run](https://wiki.supermetroid.run/Pink_Brinstar_Power_Bomb_Room)) — 100% often Quick-Drops a **crumble** from above |
| Item | Power Bomb PLM ≈ **(100–120, 370–395)** (Brinstar PB 5 on that list) |
| Entry spawn | Bottom door ~**(460–472, 395)**; top door ~**(472–493, 139)** |
| Maze wall@437 | **Bombable** morph double-tap; crouch/missiles/supers no → ~**(408, 398)** |
| Free-air topology (after wall open) | Door side x≳410 free; **left volume** x≲220 at y≈310–390 free; **mid x≈230–400 solid** at those y; pit y≈452 continuous but morph headroom ~2px; top y≈171 sealed from shaft below |
| Working suffix | `play_pink_pb_from_left_zone` from ~(180,360) walk/fall → pocket → collect pb 5/5 |
| Collect code | `play_pink_pb_morph_bomb_collect` / `play_pink_pb_from_left_zone` |
| Helpers | `ensure_morph`, `bomb_roll_left_safe`, `wait_until`, `is_morph` |
| Place bridge | Prefer `place(180,360)` left-zone then pure suffix; fallback `place(220,395)` (`--allow-place`) |
| Evidence | `--to pb-maze-wall`; left-zone place + collect; map `debug/post_spore/sill/PinkBrinstarPowerBombRoom.png` |
| Continuous? | Need pure door→left-volume (or pure top→left-volume) |
| Blockers | Mid solid wall; pit dead-end; top floor sealed; sill approach separate |

**External refs**

- [Pink Brinstar Power Bomb Room](https://wiki.supermetroid.run/Pink_Brinstar_Power_Bomb_Room) (Mission Impossible / Quick Drop)
- [Quick Drop](https://wiki.supermetroid.run/Quick_Drop) (crumble pass-through)
- Room map: `debug/post_spore/sill/PinkBrinstarPowerBombRoom.png` (from wiki)
- Big Pink map: `debug/post_spore/sill/BigPink.png`

```bash
# Pure wall open:
uv run python super_metroid/scripts/probe/post_spore_pb.py --to pb-maze-wall \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_pb_door_entered.state
# Mid-maze pure attempt (door→left-volume still open):
uv run python super_metroid/scripts/probe/post_spore_pb.py --to pb-mid-maze \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_pb_door_entered.state
# Collect with place bridge:
uv run python super_metroid/scripts/probe/post_spore_pb.py --to pb-collect --allow-place \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_pb_door_entered.state
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
uv run python super_metroid/scripts/probe/phantoon.py capture-entry
uv run python super_metroid/scripts/probe/phantoon.py ship-route
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
| 4 | Shaft → PB door | — | top y907 + bottom place entry; pure climb **open** | place top/bottom |
| 5 | PB maze wall@437 | — | **pure break** (reactive helpers) | — |
| 5b | PB mid-maze → collect | — | left-zone→collect; mid solid **open** | place 180,360 |
| 6 | PB → Red Tower (GHZ/Noob) | — | **open** | partial states |
| 7 | Red Tower → Crateria elev | — | **open** | **yes** |
| 8 | Moat / Ocean / WS entry | — | **open** | **yes** |
| 9 | WS → Phantoon room | — | **open** | **yes** |
| 10 | Phantoon fight | — | **open** | entry state |

**Bottleneck for continuous growth:** segment **4** (climb/door to Pink PB) then natural collect; topology through Phantoon *room* is door-warp proven after PB.

---

## Immediate next actions (ordered)

1. **Crack 4** — pure into **drop-air x≈535–555 y≈850–910** (or onto y907 ledge).
   Proven: free-fall there lands ledge + entry. Blocked so far: east side wall@613;
   west side (`left_upper`) needs height gain (~spin peak only Δy≈79).
2. **Crack 5b** — pure door→left-volume (or top→crumble→collect). Left-zone
   suffix green; mid y≈395 solid between wall@405 and pocket@225.
3. **Compose** `play_post_spore_to_pb` (door + maze + collect) from natural Super.
4. **Continuous dry** power-on → PB.
5. **Segment 6** GHZ / Noob / Red Tower controller from post-PB Big Pink.

Do **not** polish Phantoon fight until segments 4–6 are controller-proven.

---

## Key coordinates (Big Pink `0x9D19`)

| Landmark | Block | Approx px |
|----------|-------|-----------|
| Farm door (from `0xA0A4`) | [79, 87] | (1272, 1400) |
| Pocket wall (stuck x) | — | x≈**1157**, y≈1419 |
| PB bottom door (place zone) | [32, 71] | midair ~(**580**, **1136**) → alcove ~530,1163 |
| PB top door ledge (solid) | [32, 55-ish] | **x≈520–548, y≈907** (preferred) |
| Drop-air (lands top ledge) | — | place **x∈[535,555], y∈[850,910]** → ~(535,907) |
| Upper ledge (roof of bottom corridor) | — | y≈**1051**, x≈549–613 |
| Main platform west wall | — | min_x=**613**, y≈1179 |
| GHZ green Super door | [63, 103] | (1016, 1656) |
| Charge Beam Chozo | [37, 118] | (600, 1896) |

Pink PB room `0x9E11`: item [6, 23] ≈ (104, 376).

---

## Code / state index

| Artifact | Role |
|----------|------|
| `routes/post_spore_controller.py` | Super collect → farming → Big Pink crest → main shaft |
| `dev/phantoon_dev.py` | PB warp-collect, `SHIP_ROUTE`, Phantoon entry/fight |
| `scripts/probe/post_spore_pb.py` | Dev probe through crest / tunnel-floor / main |
| `scripts/probe/phantoon.py` | collect-pb / capture-entry / ship-route |
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
