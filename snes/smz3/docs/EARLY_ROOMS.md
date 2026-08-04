# SMZ3 early rooms and portals

## Start

Combo always boots into **Super Metroid**. After file select on test seed 1337:

| Field | Value |
|-------|--------|
| Room | Landing Site `0x91F8` |
| Area | Crateria (0) |
| Health | 99 |
| Ship settle XY | ~(1152, 1088) |

## First natural segment (M2)

```
Landing Site (0x91F8)
  → bottom-left blue door (no items)
Parlor and Alcatraz (0x92FD)
  entry XY ~(1240, 139) top-right
```

Reproduce:

```bash
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_early_rooms.py --save-png
```

Implementation: `smz3/early_route.py` (room timeout 3× provisional baselines).

### Landing Site notes

- Early controllable frames may still ignore input (pose 0 on ship).
- Door to Parlor is **bottom-left** (graph block `[0, 71]`), not right.
- Open blue door with shot (`LEFT`+`X`), then hold `LEFT` through transition.

## Path to first Zelda portal (natural)

Fixed portal (not randomized). tewtal `sm_teleport_table` comment:
**"Crateria map station → Fortune teller"** — door **`$8976`** → cave **`$0122`**.

On the combo ROM that door pointer is the **Parlor bottom-right red door**
(block `[31, 55]`). Walking through it **is** the SM→Z3 teleport; Pre-Map
Flyway / Crateria Map Room are **not** loaded first. That is the earliest
natural escape from Zebes start (no other early fixed portals).

```
Landing Site (0x91F8)
  → Parlor (0x92FD) top-right entry
  → left-shaft descent (morph-route parlor policy + spin drops)
  → bottom-right RED door [31, 55]  (missiles)  door ptr $8976
  → natural walk-in (no RAM pokes)
ALttP cave $0122  Fortune Teller (light world, OW screen $35 data)
```

| Side | ID |
|------|-----|
| SM door pointer | `$8976` (Parlor red door = combo portal) |
| Z3 cave / exit | `$0122` |
| Z3 OW screen (table) | `$35` (Lake Hylia area) |
| Return SM door | `$8BCE` → Parlor |

### Dev checkpoint: stop **before** teleport

Post-teleport residue is a **black force-blank** under stable-retro — not useful
to "play". Save at the red door still in SM, then walk the portal yourself:

```bash
# Still SM at red door (visible Parlor) + PortalRedDoor.state
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_portal.py --save-png --save-state
uv run python smz3/scripts/play_portal.py
# In window: missiles + RIGHT into red door. F9 dumps. ESC records buttons.

# Optional: auto-walk into portal (usually black hang)
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_portal.py --through-portal --save-state
```

| State | What you see |
|-------|----------------|
| `PortalRedDoor` | SM Parlor, door band, missiles assist — **use this** |
| `PortalResidue` | module `$0F` / black screen after natural walk-in |

Missile grant is only a reachability assist (open red door); the teleport is
natural combo code. Do **not** poke Z3 module/RAM to fake Link control.

Implementation: `smz3/portal_route.py` (`stop=red_door|after_portal`),
`scripts/probe_portal.py`, `scripts/play_portal.py`.

Catalog: `smz3/portals.py`. Other portals (later): Norfair map ↔ Old Man,
Maridia missile refill ↔ DW ice rod, LN refill ↔ Mire fairy.

### Portal status (2026-07-30)

| Step | State |
|------|--------|
| Parlor top → left shaft | done (replay `seg01_parlor` policy) |
| Shaft descent to door Y band (~880) | done (spin-drop script) |
| Red door reach + missile assist | done (`PortalRedDoor` checkpoint) |
| Natural walk-in fires teleport | **done** (`module $0F`, cave `$0122`) |
| Controllable Link (module `$07`/`$09`, sub 0) | **done** (~300 idle frames after `$0F`) |
| Exit Fortune Teller → OW → Link's House (no sword) | **done** (`outdoor_route.py`) |
| Enter Link's House + open chest | **done** (`house_route.py`) |

**Settle detail:** `transition_to_zelda` stores module `$0F`, force-blanks,
then `jml $02b6fb` pre-overworld. Under stable-retro + **JP 1.0** combo this
completes in ~300 frames → module `$09`, screen `$35` (Fortune Teller exterior),
drawn OW frame, D-pad changes facing / DOWN walks. Stopping on first `$0F`
looks like a permanent hang — `open_red_door_portal` now waits a settle budget.

**ROM prerequisite:** ALttP JP 1.0 at `roms/zelda3_jp.sfc` (not USA
`zelda3.sfc`). USA base breaks Z3 handoff.

### Outdoor: Fortune Teller → Link's House (no sword)

```
OW $35 Fortune Teller exterior  (PortalSettled)
  → DOWN to y≥3440 (door is north of spawn; pure UP enters cave)
  → RIGHT to corridor x≈2704
  → UP along corridor → OW $2D
  → UP+LEFT (pure LEFT blocked at entry Y) → OW $2C Link's House
```

No sword on test seed 1337 (uncle not yet). Hostiles are **side-stepped**
only — never reverse the phase goal. Sticky south-clear avoids DOWN/UP
oscillation against the house wall.

```bash
# From PortalSettled + MP4 proof
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_outdoor.py --video --save-png
# Artifacts: recordings/fortune_to_links_house.mp4, m3_links_house_ow.png
```

Implementation: `smz3/outdoor_route.py`, `scripts/probe_outdoor.py`.

### Link's House: enter + chest (map-driven)

Map sources (snes_editor / Yaze — not the in-game minimap):

| Source | Path / fact |
|--------|-------------|
| Yaze warp JSON | `snes_editor/alttp/.../yaze_map_data/hyrule_castle_warps_0x1b.json` — entrance_id `$01` @ **(2224, 2800)** map `$2C` |
| OW asset YAML | `snes_editor/alttp/zelda3/assets/overworld/overworld-44.yaml` — entrance tile (11,15), exit door local (184,232) |
| Feature CSV | `snes_editor/alttp/data/overworld_features.csv` + `overworld_map.overworld_feature_rows(0x2C)` |
| Interior room | `asset_editor/assets/rooms/room_004.json` — Chest (6,16); door spawn measured (2424, 8664) |
| Vanilla open XY | alttp lamp script end **(2491, 8632)** face UP + A |

```
OW $2C arrival ~(2528, 2920)
  → DOWN clear → LEFT to west flank x≈2112
  → UP west ramp to y≈2846 (under house south face; porch is one-way from y≥2936)
  → RIGHT to entrance X≈2224 → UP into door gap
  → interior room $04 spawn ~(2424, 8664)
  → walk to (2491, 8632) UP + A → chest flag $0403 / item (seed-dependent)
```

Test seed 1337: chest grants **heart container** (max HP 24→32), not lamp.

```bash
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_house.py --video --save-png
```

Implementation: `smz3/house_route.py`, `scripts/probe_house.py`.

## Room timeout

| Room key | Provisional standard frames |
|----------|----------------------------|
| `0x91F8` | 90s |
| `0x92FD` | 60s |
| `0x98E2` | 30s |
| `0x9994` | 20s |

Game over at **3×** standard (`smz3.room_timeout`).

## Reuse

While SM is active, call `super_metroid.ram.parse_state` on combo `get_ram()`.
While Z3 is controllable, use `alttp.ram` (pending clean portal settle).
