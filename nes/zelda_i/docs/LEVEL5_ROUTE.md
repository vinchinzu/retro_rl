# Level 5 — The Lizard (route notes)

Status: **assisted-entry** (Survival recon 2026-08-06). Not Clean STATUS.

Source:
[Zelda Dungeon L5](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-5-the-lizard/).

## Overworld

| Field | Live | Notes |
|-------|------|-------|
| Lost Hills screen | **0x1B** | Soft maze: 4× UP |
| Door screen | **0x0B** | UP @x≈112 |
| Entry room | **0x76** | South mouth ~(120, 205) after mode-16 settle |
| Level id | 5 | `ADDR_LEVEL=0x0010` |
| Triforce bit | `0x10` | After Digdogger |
| Item | Whistle / Recorder | `ADDR_WHISTLE=0x065C` |

### Lost Hills (verified)

1. Enter **0x1B** from **0x1C** west @y≈140.
2. East-ledge pocket: arrive ~(240, 141); **DOWN** then **LEFT** into main path
   (x≲100) before climbing.
3. Hold **UP** (align x≈96): three self-wraps stay on 0x1B (reappear south);
   **fourth** UP → **0x0B**.
4. On 0x0B, align x≈112 and UP into the mountain mouth → `level==5`, settle
   room **0x76**.

Only reliable non-wrap exit from Lost Hills is **LEFT** (wiki); wrong
UP/RIGHT/DOWN loops the hill screen.

### Assisted path prefix (from mid-east ~0x4A)

```
0x4A →N→ 0x3A →E→ 0x3B →N→ 0x2B →E→ 0x2C →N→ 0x1C →W@y140→ 0x1B
  → free pocket → UP×4 → 0x0B → door UP → L5 0x76
```

Hops table: `level5_overworld.LEVEL5_PATH_HOPS`. Controller:
`OverworldToLevel5Controller` (pocket free + four-up counter + door hunt).

Can visit L5 without clearing L2–L4 (first quest). Ladder / bracelet / magic
sword are optional for the **door** itself; bracelet warp shortens OW only.

## Interior (source → live)

| Room id | Live enemies / notes | Doors / items |
|---------|----------------------|---------------|
| **0x76** entry | No combat objects at settle; `room_obj_count=3` (statues?) | North open → 0x66; south exits OW; **east not opened** from south spawn (walkthrough: Pols Voice + key to the right — door may be shutter/locked; still open in source map) |
| **0x66** | **3× type 0x30** Gibdo-correlated, **HP=112**, `AliveRule.TYPE_AND_HP` | Pre-clear `doors=0`; after clear **`doors=0x08`** (east free) → **0x67**; DOWN → 0x76; UP/LEFT not free from clear bit alone |
| **0x67** (E of 0x66) | Residual: 1× type **0x4e** (hp0) + 2× type **0x40** (hp240) | Entered from cleared 0x66 RIGHT @y≈141; `doors=0x02` on settle |

### Pure: clear 0x66 (bead `rr-vqw`)

- Spec: `level5_dungeon.ROOM_66_SPEC` / stop `level5_room_66_cleared`
- Controller: `GenericDungeonRoomController` (import-only from `dungeon`)
- Start: `L5_Room_66` (in-room) or chain north from `Level5Entrance` (0x76)
- Track: **Clean** isolated (no health write); ~2k frames in-room, ~4k from entrance
- Object confirm: type **0x30**, spawn HP **112**, expected count **3**
- Doors after clear: **`cur_opened_doors=0x08`** → east open to 0x67; south always back to entry; north/west blocked without further geometry/items
- Whistle path residual: not mapped this bead (need dark rooms / candle / further graph)

```bash
# Isolated pure 2/2 from room-66 checkpoint
uv run python nes/zelda_i/scripts/run_level5_clear66.py --trials 2

# Chain 0x76 → 0x66 clear + save Level5Cleared66
uv run python nes/zelda_i/scripts/run_level5_clear66.py --from-entrance --save-state --trials 1
```

Source route (not all live-mapped yet):

- RIGHT Pols Voice + key (source: east of **entry** 0x76 — still not opened live from south spawn)
- UP Gibdo dark rooms → key; optional bomb skip past Dodongos
- Map; Zol key; Gibdo bombs; Blue Darknuts → staircase
- LEFT Darknuts → staircase → **Whistle**
- Digdogger: Whistle shrinks, sword/bomb finish → heart → TF shard 5

## Boss / Triforce

- Boss: Digdogger (whistle to shrink)
- Triforce bit: **`0x10`**
- Whistle RAM: `0x065C`

## Checkpoints

| State | Provenance |
|-------|------------|
| `Level5Entrance.state` | Assisted settle room 0x76 ~(120,205); sword=1; no inventory/TF poke |
| `OW_1B_LostHills.state` | On 0x1B after pocket free (dev fixture) |
| `OW_0B_L5Door.state` | Door screen before enter (dev fixture) |
| `L5_Room_66.state` | North of entry after walk (assisted) |
| `Level5Cleared66.state` | Clean pure clear of 0x66; doors=0x08; east free → 0x67 |

## Probe

```bash
# From mid-east OW checkpoint
uv run python zelda_i/scripts/probe_level5_entry.py --infinite-life --save-state

# From Lost Hills
uv run python zelda_i/scripts/probe_level5_entry.py --from-state OW_1B_LostHills \
  --infinite-life --save-state --tag l5_from_hills

# From door screen only
uv run python zelda_i/scripts/probe_level5_entry.py --from-state OW_0B_L5Door \
  --infinite-life --save-state --tag l5_from_door
```

## Evidence

- `recordings/l5_entry_recon.json` — door 0x0B, hills 0x1B, entry 0x76, room probes
- `recordings/l5_clear66_isolated.json` / `l5_clear66_entrance.json` — pure clear trials
- `recordings/l5_66_door_probe.json` — N/E/S/W after clear
- `recordings/l5_entrance.png`, `l5_0b_door.png`, `l5_1b_free.png`, `l5_room_66.png`
- Modules: `level5_overworld.py`, `level5_dungeon.py`, `scripts/run_level5_clear66.py`, `scripts/probe_level5_entry.py`

## Next

- Open/verify entry **east** Pols Voice room id live (from 0x76)
- Map north dark rooms from 0x66 (candle); residual 0x67 types 0x4e/0x40
- Whistle item room; Digdogger policy
- Natural-entry from real predecessor (no assist) when tip reaches L5
