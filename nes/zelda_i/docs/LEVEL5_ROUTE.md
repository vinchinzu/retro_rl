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
| **0x76** entry | No combat objects at settle; `room_obj_count=3` (statues) | North open → **0x66**; south exits OW; **east → 0x77** is a **key door**. Mid-room blocks pure RIGHT @ y≈141 (stuck x≈128, tile≈181); y≈149–157 reaches wall x≈208. Clear north **0x66** first (3× Gibdo + fixed key), return south, approach wall y≈157, channel y≈141, RIGHT. Route fixtures: `Level5EntranceFromL4` then `Level5Cleared66`. Old `Level5Entrance` (At4A) lacks Raft/Stepladder/bombs/TF — do not use it for route work |
| **0x66** | **3× type 0x30** Gibdo, **HP=112**, `AliveRule.TYPE_AND_HP` | Pre-clear `doors=0`; after clear **`doors=0x08`** → east free to **0x67**; DOWN → 0x76; UP/LEFT blocked natural (dark-room / west key residual) |
| **0x67** (E of 0x66) | **2× type 0x40** Bubble (HP=240, **sword-immune**) + **1× type 0x4e** (hp0 residual) | Settle `doors=0x02` (LEFT only) → back to 0x66; R/U/D solid. **Dead-end graph node** — no clear pure |
| **0x77** (E of 0x76) | **5× type 0x16** Pols Voice, **HP=160**, `TYPE_AND_HP` | `room_item_id=0x19` small key; doors=0 on settle. Combat pure from `L5_Room_77` **2/2** (~5.3k frames) with backstep controller |
| **0x65** (W of 0x66) | **5× type 0x30** Gibdo HP=112 | PARTIAL: natural west blocked; with door poke settle `doors=0x01` (east back). Clear-only; no extra door bits after clear |
| **0x55** (N of 0x65) | **5× type 0x13** Zol HP=32; item `0x19` | PARTIAL dark-room chain via forced doors from 0x65 UP |

### Door bit convention (live)

`ADDR_CUR_OPENED_DOORS` (`0x00EE`): bit0=R, bit1=L, bit2=D, bit3=U (Data Crystal).

| Room | Observed `doors` | Meaning |
|------|------------------|---------|
| 0x66 post-clear | `0x08` | East free to 0x67 (east arch / shutter) |
| 0x67 settle | `0x02` | West open back to 0x66 |
| 0x65 settle (forced west) | `0x01` | East open back to 0x66 |
| 0x76 entry | `0x00` | North permanent archway still walkable; east closed |

### Pure: clear 0x66 (bead `rr-vqw`)

- Spec: `level5_dungeon.ROOM_66_SPEC` / stop `level5_room_66_cleared`
- Controller: `GenericDungeonRoomController` (import-only from `dungeon`)
- Start: `L5_Room_66` (in-room) or chain north from `Level5EntranceFromL4` (0x76)
- Track: **Clean** isolated (no health write); ~2k frames in-room, ~4k from entrance
- Object confirm: type **0x30**, spawn HP **112**, expected count **3**
- Doors after clear: **`cur_opened_doors=0x08`** → east open to 0x67

```bash
uv run python nes/zelda_i/scripts/run_level5_clear66.py --trials 2
uv run python nes/zelda_i/scripts/run_level5_clear66.py \
  --from-state Level5EntranceFromL4 --infinite-life --save-state --trials 1
```

### Graph pure: 0x66 → 0x67 (bead `rr-87a`)

- Spec: `ROOM_67_SPEC` / stop `level5_room_67_arrived`
- Controller: `Level5East67Controller` (RIGHT @ y≈141; no combat)
- Start: `Level5Cleared66`
- Bubbles **0x40** sword-immune — arrival only, not clear
- Dark-room / west residual **PARTIAL**: natural N/W from 0x66 blocked even with
  candle poke; forced doors open west → 0x65 (5× Gibdo) → north 0x55 (5× Zol)

```bash
uv run python nes/zelda_i/scripts/run_level5_east67.py --trials 2
uv run python nes/zelda_i/scripts/run_level5_east67.py --save-state
```

### Route: 0x66 → 0x76 → east key 0x77

East `0x76→0x77` is a **key door**. Policy: `level5_path.level5_east_key_step`.
Spec/stop stay in `level5_dungeon` (`ROOM_77_SPEC` / `level5_room_77_key_success`).
Do not detour **0x67** (Bubble dead-end residual).

Start: **`Level5Cleared66`** (from `Level5EntranceFromL4` / `Level4Complete`).
`--keep-keys` is optional explicit safety (predecessors keep keys by default).

```bash
uv run python nes/zelda_i/scripts/run_level5_east_key.py \
  --from-state Level5Cleared66 --infinite-life --save-state --trials 1
```

### Isolated combat: 0x77 Pols Voice + key (bead `rr-076`)

- Controller: `Level5PolsVoiceController` (backstep when stuck close)
- Start: **`L5_Room_77`** (room-ready; keys forced 0 **only** for this isolated fixture)
- Object: type **0x16**, HP **160**, count **5**; key RoomItemId **0x19**
- Live 2/2 isolated ~5.3k frames (infinite-life optional for recon)

```bash
uv run python nes/zelda_i/scripts/run_level5_east_key.py --trials 2 --infinite-life
uv run python nes/zelda_i/scripts/run_level5_east_key.py --save-state --infinite-life
```

### Route: 0x77 → bomb-west 0x66 → Recorder 0x04 (bead `rr-4d53.5`)

ROM `0x66` west is a **bomb wall** to **0x65** (Dodongo skip). Policy:
`level5_return_66_step` then `bomb_west_from_66` (south band y=189, stand
(32, 141); pause-menu bombs, no poke). Then the proven 0x65 bomb-west
suffix: 0x64 Blue Darknuts → center stairs 0x07 → 0x06 key-west → 0x05
block-stairs → cellar **0x04** / `ADDR_WHISTLE`.

Start: **`Level5EastKey`** (keys=2 bombs=7 whistle=0). Survival 1/1:
whistle `0→1`, room **0x04** mode 9, keys=1 bombs=5, deaths=0,
progression_writes=0, capacity_writes=0. `route_eligible=false` until
natural-entry from L4 complete is composed.

```bash
uv run python nes/zelda_i/scripts/run_level5_east_to_whistle.py \
    --from-state Level5EastKey --infinite-life --save-state
```

Source route (live through Recorder; Digdogger/TF is a separate suffix):

- RIGHT Pols Voice + key (live room **0x77**; key door after 0x66)
- Return west, bomb-west 0x66 → 0x65 (skip Dodongos)
- 0x65 bomb-west → 0x64 Blue Darknuts → staircase
- Cellar 0x07 other mouth → 0x06 key-west → 0x05 block-stairs → **Whistle**
- Digdogger: Whistle shrinks, sword/bomb finish → heart → TF shard 5

## Boss / Triforce

- Boss: Digdogger (whistle to shrink)
- Triforce bit: **`0x10`**
- Whistle RAM: `0x065C`

## Checkpoints

| State | Provenance |
|-------|------------|
| `Level5Entrance.state` | Old At4A settle 0x76; lacks Raft/Stepladder/bombs/TF — do not use for route |
| `Level5EntranceFromL4.state` | From `Level4Complete`; Raft/Stepladder/bombs/TF `0x0c` preserved |
| `OW_1B_LostHills.state` | On 0x1B after pocket free (dev fixture) |
| `OW_0B_L5Door.state` | Door screen before enter (dev fixture) |
| `L5_Room_66.state` | North of entry after walk (assisted) |
| `Level5Cleared66.state` | Clean pure clear of 0x66; doors=0x08; east free → 0x67 |
| `L5_Room_67.state` | East residual Bubbles; doors=0x02 |
| `L5_Room_77.state` | Isolated Pols Voice room-ready (keys forced 0 OK for this fixture) |
| `Level5EastKey.state` | 0x77 cleared + keys≥1 |
| `Level5Entered65From77.state` | EastKey return + 0x66 bomb-west; room 0x65 |
| `Level5WhistleFrom77.state` | EastKey → natural Recorder in cellar 0x04; whistle=1; `route_eligible=false` |
| `L5_Room_65.state` | West of 0x66 via forced doors (PARTIAL) |
| `L5_Room_55.state` | North of 0x65 via forced doors (PARTIAL) |

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
- `recordings/l5_residual_recon.json` — 0x67 / dark-room / entry-east recon
- `recordings/l5_east67_isolated.json` — graph pure 0x67 arrival
- `recordings/l5_east_key_*.json` / `l5_pols_2of2.json` — Pols Voice key trials
- `recordings/l5_e2w_t2.json` / `l5_e2w_t2_final.png` — EastKey → Recorder 0x04
- `recordings/l5_entrance.png`, `l5_0b_door.png`, `l5_1b_free.png`, `l5_room_66.png`,
  `l5_room67.png`, `l5_room_77.png`, `l5_east_key.png`
- Modules: `level5_overworld.py`, `level5_dungeon.py`, `level5_path.py`
  (facade; `level5_west_path`, `level5_whistle_path`, `level5_cellar_path`,
  `level5_tf_path`),
  `scripts/run_level5_clear66.py`, `scripts/run_level5_east67.py`,
  `scripts/run_level5_east_key.py`, `scripts/run_level5_east_to_whistle.py`,
  `scripts/probe_level5_entry.py`

## Next

- Attach proven Whistle basement `0x04` → Digdogger `0x24` → L5 TF `0x14`
  suffix onto `Level5WhistleFrom77` (still `route_eligible=false`)
- Natural-entry (no assist) after L4-complete → East Key → Recorder is composed
- Natural 0x66 north shutter / 0x56 Dodongos remain an alternate, not required
  for Recorder
