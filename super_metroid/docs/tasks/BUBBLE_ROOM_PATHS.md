# Bubble Mountain — room paths (parse catalog)

**Last updated:** 2026-08-03  
**Room:** Bubble Mountain `0xACB3` (2×4 screens, 32×64 blocks)  
**Source graph:** `maps/full_room_graph.json` (node ids / blocks)  
**Human evidence:** `tasks/bubble_missile.json` (+ `_end.state`)

> **Scope:** Additive path / door catalog for guided recording and future
> pure parse. **Does not** edit or replace the in-progress Bubble → Bat spine
> (`routes/kpdr/guide_paths.py` `GUIDE_BUBBLE`, pure R-series, mid climb).
> Bat Super-door path remains separate WIP.

---

## Doors (all nodes — required for any path parse)

Bubble has **7 door nodes**. World pixels are approximate Samus lip centers
(`block * 16` + door mid-height ~+24 for horizontal doors). Prefer graph
`block` for topology; use world for overlays / human pins.

| Node | Name (graph) | Side | Subtype | Block `[x,y]` | ≈ World `(x,y)` | Target room | Target name | KPDR role |
|------|--------------|------|---------|---------------|-----------------|-------------|-------------|-----------|
| **1** | Top-most Left Door (to Bubble Missiles) | left | **green** (Super) | `[0, 7]` | `(16, 136)` | `0xAC83` | Green Bubbles Missile Room | Side missile pack (green door) — **not** the in-room pack |
| **2** | Second Top Left Door (to Save) | left | blue | `[0, 23]` | `(16, 392)` | `0xB0DD` | Bubble Mountain Save Room | **Trap** on first Speed spine (save detour) |
| **3** | Second Bottom Left Door (to Rising Tide) | left | blue | `[0, 39]` | `(16, 648)` | `0xAFA3` | Rising Tide | **First-visit entry** (Cathedral climb) |
| **4** | Bottom-most Left Door (to Farming Room) | left | blue | `[0, 55]` | `(16, 904)` | `0xAF72` | Upper Norfair Farming Room | Bottom-left exit; Speedway farm return |
| **5** | Bottom Door (to Purple Shaft) | down | blue | `[7, 63]` | `(120, 1016)` | `0xAEDF` | Purple Shaft | Floor drop (not first Bubble spine) |
| **6** | Bottom Right Door (to Single Chamber) | right | blue | `[31, 23]` | `(488, 392)` | `0xAD5E` | Single Chamber | Wave branch (post-Speed) |
| **7** | Top Right Door (to Bat Cave) | right | **green** (Super) | `[31, 7]` | `(488, 136)` | `0xB07A` | Bat Cave | **Speed spine** Super door (pure R-series WIP) |

### Connection ids (graph)

| Node | Connection id | Direction |
|------|---------------|-----------|
| 3 | `connection_221_afa3_2_to_acb3_3` | ↔ Rising Tide |
| 4 | `connection_225_af72_3_to_acb3_4` | ↔ Upper Norfair Farm |
| 5 | `connection_226_aedf_1_to_acb3_5` | ↔ Purple Shaft (vertical) |
| 1 | `connection_229_acb3_1_to_ac83_2` | ↔ Green Bubbles Missiles (needs Super) |
| 2 | `connection_230_acb3_2_to_b0dd_1` | ↔ Save |
| 6 | `connection_231_acb3_6_to_ad5e_1` | ↔ Single Chamber |
| 7 | `connection_233_acb3_7_to_b07a_1` | ↔ Bat Cave (needs Super) |

### Wrong-door traps (first visit / Speed spine)

| Node | Door | Why trap |
|------|------|----------|
| 1 | Top-left green → `0xAC83` | Side pack; not Bat / not farm |
| 2 | Save → `0xB0DD` | Save loop; human `cathedral_to_bat_human` fell in |
| 5 | Floor → `0xAEDF` | Drops out of mountain |
| 6 | Right mid → `0xAD5E` | Wave branch only after Speed |

---

## In-room item (not a door)

| Item | Graph | Block `[x,y]` | ≈ World | Human pickup |
|------|-------|---------------|---------|--------------|
| Missile (Visible) | PLM `0xEEDB` | `[20, 60]` | `(328, 968)` | `(340, 956)` @ f=1158 in `bubble_missile` |

Capacity on that recording: **15 → 20** (pack +5). This is the **floor
missile** inside Bubble — **not** the green-door room `0xAC83`.

---

## Path: `bubble-missiles-to-farm` (human-validated)

**Intent:** From first-visit entry (Rising Tide door **3**), collect the
in-room floor missiles, climb back up the right column, then leave through
**bottom-most left door 4** into Upper Norfair Farming Room.

| Field | Value |
|-------|-------|
| Path id | `bubble-missiles-to-farm` |
| Status | human demo only — **not** pure GREEN / not continuous tip |
| Task JSON | `super_metroid/tasks/bubble_missile.json` |
| End state | `tasks/bubble_missile_end.state` |
| Start state | `scratch/post_rising_tide_to_bubble_pure.state` |
| Entry door | node **3** → from `0xAFA3` Rising Tide |
| Exit door | node **4** → to `0xAF72` Upper Norfair Farm |
| Doors used | **3** (in), **4** (out) — no Save / Bat / Single Chamber |
| Frames | ~3931 total; Bubble `f=0..3774`, Farm `f=3775..3930` |
| Missiles | 15 → 20 at `(340, 956)` |

### Room sequence

```text
0xACB3 Bubble Mountain   f=0..3774     entry (56, 641) → exit (19, 907)
0xAF72 Upper Norfair Farm f=3775..3930 entry (19, 907) → end (456, 139)
```

### Bubble metrics (human)

| Metric | Value |
|--------|-------|
| entry | `(56, 641)` pose 25 — door **3** lip |
| missile | `(340, 956)` f=1158 capacity 15→20 |
| climb peak after pack | min_y ≈ 465 (mid shelves; not top band) |
| exit | `(19, 907)` pose 12 — door **4** lip |
| max_x / min_y (in room) | 475 / 465 |
| ordinary `0xB07A` | never |
| door **2** Save | never |

### Guide polyline (world pixels — for future overlay / parse)

Labels are stable ids for a parser. **Do not** merge into `GUIDE_BUBBLE`
(Bat Super spine) while that work is open.

```yaml
path_id: bubble-missiles-to-farm
room_id: 0xACB3
entry_door_node: 3
exit_door_node: 4
item:
  kind: missiles_visible
  block: [20, 60]
  world: [340, 956]   # human pin
points:
  # --- enter via door 3 (Rising Tide) ---
  - {x: 56,  y: 641, label: entry-door3}
  - {x: 120, y: 560, label: shelf-1}
  - {x: 220, y: 535, label: mid-cross}
  - {x: 340, y: 532, label: right-shelf}
  - {x: 470, y: 600, label: right-column-top}
  # --- drop right column to floor missile ---
  - {x: 473, y: 720, label: right-drop-1}
  - {x: 473, y: 860, label: right-drop-2}
  - {x: 450, y: 940, label: floor-approach}
  - {x: 340, y: 956, label: missile-pack}   # capacity +5
  # --- climb back up right column ---
  - {x: 400, y: 920, label: post-pack}
  - {x: 473, y: 820, label: reclimb-1}
  - {x: 470, y: 680, label: reclimb-2}
  - {x: 450, y: 560, label: reclimb-mid}
  - {x: 320, y: 505, label: mid-left-bias}
  - {x: 250, y: 640, label: lower-mid}
  # --- bottom floor run left to door 4 ---
  - {x: 300, y: 800, label: lower-drop}
  - {x: 265, y: 905, label: bottom-floor}
  - {x: 140, y: 930, label: bottom-run}
  - {x: 90,  y: 907, label: door4-approach}
  - {x: 19,  y: 907, label: exit-door4}     # → 0xAF72
```

### Farm after exit (optional handoff)

Not part of the Bubble room parse, recorded for continuity:

| Pin | World | Note |
|-----|-------|------|
| Farm entry settle | `(526, 139)` after transition | door from Bubble |
| Recording end | `(456, 139)` pose 2 | standing mid-Farm |

```yaml
path_id: bubble-missiles-to-farm
room_id: 0xAF72
entry_door: right-to-bubble   # graph: Farm node 3
points:
  - {x: 526, y: 139, label: farm-entry}
  - {x: 456, y: 139, label: farm-stand}
```

---

## Other Bubble paths (index only — do not edit here)

| Path id | Entry door | Exit door | Status | Notes |
|---------|------------|-----------|--------|-------|
| `bubble-to-bat` | **3** Rising Tide | **7** Bat Super | pure WIP (R13 Phase C green; D red) | `GUIDE_BUBBLE` / `SM-K4.4-PURE` — **leave alone** |
| `bubble-missiles-to-farm` | **3** Rising Tide | **4** Farm | human demo | this doc + `tasks/bubble_missile.json` |
| `farm-to-bubble` | **4** Farm | (into room) | scaffold / post-Speed | Speedway shortcut — parked |
| `bubble-to-single-chamber` | any | **6** Single Chamber | open | Wave branch after Speed |
| `bubble-to-green-missiles` | any | **1** green Super | open | side pack room `0xAC83` |

---

## Parse checklist (when wiring a loader)

1. Require every path to name **entry_door_node** and **exit_door_node** from
   the table above (or `null` if stay-in-room practice).
2. Reject paths that omit door nodes or invent ids outside 1–7.
3. Treat nodes **1, 2, 5, 6** as traps unless the path id explicitly opts in.
4. Distinguish **in-room missile** (`block [20,60]`) from **door 1** green
   missile room.
5. Do not overwrite `GUIDE_BUBBLE` points from this path — register a separate
   preset (e.g. `bubble-missiles-to-farm`) when product wants the overlay.

### Re-record command

```bash
uv run python super_metroid/scripts/record/guided_human.py \
  --from bubble --route bubble-only --name bubble_missile_v2
# F5 save → tasks/bubble_missile_v2.json
# Stop only after ordinary 0xAF72 (door 4), not Save / Bat.
```

---

## Evidence refs

| Artifact | Role |
|----------|------|
| `tasks/bubble_missile.json` | Human inputs + trace for this path |
| `tasks/bubble_missile_end.state` | End pin in Farm |
| `tasks/bubble_jump_try.json` | Earlier climb try (no door exit; stay in Bubble) |
| `docs/tasks/HUMAN_CATHEDRAL_TO_BAT_VALIDATE.md` | Cathedral→Bubble human; Save trap notes |
| `maps/full_room_graph.json` | Authoritative door nodes / blocks |
| `routes/kpdr/bubble_mountain_params.py` | Bat-spine geometry (do not edit for this path) |
| `routes/kpdr/guide_paths.py` | Bat overlay WIP — do not merge this path into it yet |
