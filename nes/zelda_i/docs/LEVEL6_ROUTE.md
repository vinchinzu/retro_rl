# Level 6 — The Dragon (route notes)

Status: **assisted-entry** (recon only — not Clean STATUS)

Planning sources:

- [Zelda Dungeon — Level 6: The Dragon](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-6-the-dragon/)
- Local archive: [research/DUNGEON_WALKTHROUGHS.md](research/DUNGEON_WALKTHROUGHS.md)

## Overworld

| Claim | Source | Live |
|-------|--------|------|
| Door west near graveyard | walkthrough | **verified OW screen `0x22`** |
| Enter UP into dungeon | live | UP; south path ~**x112**, mouth band **x≈24–120** |
| Bracelet warp shortcut | walkthrough | optional residual (0x79 rock stairs) |
| Walking path from start | walkthrough | residual (Lost Woods / Death Mountain) |

### Door (live, assisted)

- **Overworld screen:** `0x22`
- **Enter:** push **UP** — south path corridor ~**x112** (spawn on fixture is ~(120,221); shift left then climb); mouth band **x≈24–120**
- **Exit check:** from entry room, **DOWN** returns to OW `0x22`
- Fixture: `custom_integrations/LegendOfZelda-Nes/L6Probe_22.state`
- Checkpoint: `Level6Entrance.state` — `level==6`, mode **5**, room **`0x79`**, xy≈**(120, 205)**

Dev path used for first door hit: screen-teleport to `0x32`, live **UP** transition onto `0x22`, then door hunt. Full Clean walk from sword/start is **not** claimed.

### Bracelet shortcut (source only)

From start `0x77`: right two screens → `0x79`, Power Bracelet push left rock, **middle staircase**. Warp exit then: down, left, up → door. Optional; walking is fine with `--infinite-life`.

### Controller / hops

Scaffold: `level6_overworld.py` (`LEVEL6_DOOR_X`, entry room constant, door-hunt stop predicates). Full hop table from start is **planned** until a live walk path is recorded.

## Interior (live recon + walkthrough)

Assisted (`UnlimitedHealthAssist`). Evidence: `recordings/l6_entry_recon.json`.

```text
OW 0x22 ──UP (south lane x~112)──► 0x79 entry (empty combat)
                       │
                       ├── DOWN → OW 0x22
                       ├── RIGHT (y~157 → x~208 → y~144–149) → 0x7a
                       ├── LEFT sealed
                       └── UP sealed

0x7a east key ──LEFT──► 0x79
                 RIGHT/UP/DOWN sealed at recon (clear/key residual)
```

| Room | Role | Enemies (live) | RoomItemId | Notes |
|------|------|----------------|------------|-------|
| **0x79** | Entry (south mouth) | none at ready | `0x03` | mode 5, xy≈(120,205); fire solids |
| **0x7a** | East of entry | **5× type `0x24`** | **`0x19` key** | walkthrough: orange wizzrobes |

### Entry RIGHT policy (required)

Naive center `y≈141` then RIGHT sticks at **x≈128** (fire-block solids, same class of trap as L2 diamonds).

Correct (no sword-A while aligning — A softlocks the channel):

1. From spawn, **UP** to **y≈157**
2. **RIGHT** to **x≈198–208** (south of fire row)
3. At **x≥206**, nudge **y≈144–149**
4. Push **RIGHT** → room **0x7a** (5× type `0x24`, RoomItemId `0x19` key)

### Walkthrough (not all live)

- RIGHT wizzrobes + key; LEFT locked (do not waste key on Old Man first) — **RIGHT/LEFT match live**
- Compass from Zols; statue/Keese rooms; multi-Wizzrobe + Bubble + Like-Like
- Mid-dungeon **Gleeok (3 heads)** then Map
- Staircase → **Magical Rod** (`ADDR_ROD=0x065F`)
- Vires / Wizzrobes → staircase → **Gohma** (one arrow to open eye)
- Heart → Triforce shard 6 (`triforce & 0x20`)

## Boss / Triforce

| Field | Value |
|-------|--------|
| Item | Magical Rod — `ADDR_ROD = 0x065F` |
| Boss | Gohma — arrow to open eye |
| Triforce bit | `0x20` |

## Checkpoints

| State | Provenance |
|-------|------------|
| `Level6Entrance.state` | Assisted enter from OW `0x22`; room-ready `0x79` |
| `L6Probe_22.state` | OW door screen (dev) |
| `L6Room_7a.state` | East key room after wall-first RIGHT |

## Evidence

- `recordings/l6_entry_recon.json`
- `recordings/l6_entrance_live.png`, `l6_ow_22.png`, `l6_room_7a.png`
- Probe: `uv run python zelda_i/scripts/probe_level6_entry.py --infinite-life --save-state`

## Not claimed

- Clean STATUS / natural-entry from real predecessor TF bits
- Full walk hop table from `0x77` / post-L1
- Bracelet warp live
- Gohma / Rod / triforce bit live
