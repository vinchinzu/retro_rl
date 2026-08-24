# Level 6 — The Dragon (route notes)

Status: **assisted pure** through Gleeok enter 0x18 (not Clean STATUS)

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
| Walking path from L5 `0x0B` | walkthrough `↓ ←×7 ↓ ← ↓ ← ↑` | **live** on `--through level6-entry` — 0x0B W/E sealed; 0x1B west is **y=141 LEFT** after south-around the x≈72 rock (not north-edge LEFT / not screenshot-tan x<32); then LEFT×6 at y=141, 0x15 south band, 0x14/0x23 SE blue paths, door UP `0x22`. See `rr-g3c1`. |

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

Scaffold: `level6_overworld.py` (`LEVEL6_DOOR_X`, entry room constant, door-hunt stop predicates). `POST_L5_TO_LEVEL6_HOPS` is **live** on `--through level6-entry` (`l6_entry_continuous_v2` 1/1). Full hop table from start is **planned**.

## Interior (live recon + assisted pure)

Assisted (`UnlimitedHealthAssist`). Evidence: `recordings/l6_entry_recon.json`,
`recordings/level6_east_key_assisted_isolated.json`,
`recordings/level6_west_wizzrobes_assisted_isolated.json`,
`recordings/l6_post_key_graph.json`.

```text
OW 0x22 ──UP (south lane x~112)──► 0x79 entry (empty combat)
                       │
                       ├── DOWN → OW 0x22
                       ├── RIGHT (y~157 → x~208 → y~144–149) → 0x7a ★ east key
                       ├── LEFT key (y~157 → x32 → y~141) → 0x78 ★ west path
                       └── UP sealed (brick)

0x7a east key ──LEFT free──► 0x79
              ──UP KEY──► 0x6a Old Man ⚠ WASTE (do not)
              RIGHT/DOWN sealed post-clear

0x78 west wizzrobes ──clear──► UP (mask 0x09) → 0x68 compass Zols
                    RIGHT free → 0x79

0x68 (5× 0x13 Zols, RoomItemId 0x16 compass) ──UP──► 0x58
                    leftover spine `(120,205)` south mouth; north door open
0x58 (8× Keese 0x1b; key drop live) ──UP──► 0x48
0x48 (blade traps 0x49 — run through) ──UP──► 0x38
0x38 (hard multi-wizzrobe / Like-Like / Bubble) ──left 0x68 UP then west-aisle──► 0x28
0x28 (2× orange 0x24; diamond floor) ──LEFT+UP then RIGHT+UP──► 0x18 Gleeok live
```

| Room | Role | Enemies (live) | RoomItemId | Notes |
|------|------|----------------|------------|-------|
| **0x79** | Entry (south mouth) | none at ready | `0x03` | mode 5, xy≈(120,205); **fire solids** |
| **0x7a** | East of entry | **5× type `0x24`** | **`0x19` key** | assisted pure: keys 0→1 |
| **0x6a** | N of 0x7a | Old Man `0x4d` + fires `0x40` | `0x03` | **key waste** — tip only |
| **0x78** | West of entry | **5× type `0x24`** | `0x03` | key-LEFT from 0x79; clear → UP |
| **0x68** | N of 0x78 | **5× type `0x13`** (Zol) | **`0x16` compass** | spine enter + `ADDR_COMPASS\|0x20` 1/1 |
| **0x58** | N of 0x68 | **8× Keese `0x1b`** | rupee drop live | spine clear 1/1; **north still sealed** |
| **0x48** | N of 0x58 | blade traps `0x49` | — | spine enter 1/1; run UP residual |
| **0x38** | N of 0x48 | **7×** wizzrobe `0x23`/`0x24` + Like-Like `0x17` + Bubble `0x40` | — | spine clear 1/1; Bubble residual; left 0x68 `(96,144)` UP live |
| **0x28** | N of 0x38 | **2× orange `0x24`** | — | spine clear 1/1 leftover `(120,181)`; diamond floor not solid |
| **0x18** | N of 0x28 | **Gleeok 3-head** (live PNG) | — | spine enter 1/1 leftover `(120,189)`; fight residual |

### Entry RIGHT policy (required)

Naive center `y≈141` then RIGHT sticks at **x≈128** (fire-block solids, same class of trap as L2 diamonds).

Correct (no sword-A while aligning — A softlocks the channel):

1. From spawn, **UP** to **y≈157**
2. **RIGHT** to **x≈198–208** (south of fire row)
3. At **x≥206**, nudge **y≈144–149**
4. Push **RIGHT** → room **0x7a** (5× type `0x24`, RoomItemId `0x19` key)

Controller: `level6_overworld.Level6EntryRightController` (~374f from spawn).

### ⚠ Old Man key trap (0x7a UP → 0x6a)

After collecting the east key, **do not** push **UP** in 0x7a.

| Field | Live |
|-------|------|
| Edge | `0x7a` **UP** with keys≥1 |
| Destination | **`0x6a`** |
| Cost | **keys 1→0** |
| Content | Old Man object `0x4d` + two `0x40` (fireball statues); tip *AIM AT THE EYES OF GOHMA* |
| Exit | DOWN free back to 0x7a (keys still 0) |

Walkthrough warning matches live: spending the only key here softlocks progress until a shop key.

### Entry LEFT key policy (required for Rod path)

Correct key spend is **LEFT from 0x79** (not UP from 0x7a).

Fire solids block naive `y≈141` LEFT from east return (~x224) or mid-room (~x112). Same wall-first band as RIGHT:

1. From east return **(224,141)**: **LEFT** to **x≈208**
2. **UP/DOWN** to **y≈157**
3. **LEFT** along wall y to **x≈32**
4. Slide **y≈141** at west wall
5. Push **LEFT** → consumes 1 key → room **0x78** (5× type `0x24`)

From south spawn **(120,205)** with keys≥1: **UP** to y≈157 → **LEFT** to x≈32 → y≈141 → LEFT.

Controller: `level6_overworld.Level6WestKeyDoorController`.

### East key pure (0x79 → 0x7a) — **assisted 2/2**

| Field | Live |
|-------|------|
| Start | `Level6Entrance` (0x79) or `L6Room_7a` |
| Specs | `level6_dungeon.ROOM_79_SPEC` / `ROOM_7A_SPEC` |
| Combat | `Level6EastKeyController` (GenericDungeonRoomController + backstep) |
| Stop | `level6_room_7a_key_success` — keys≥1, no live 0x24 |
| Checkpoint | `Level6EastKey.state` — room **0x7a**, keys **1**, xy≈(120,141) |
| Runner | `scripts/run_level6_east_key.py --infinite-life --trials 2 --save-state` |
| Track | **assisted pure** (Survival health writes; Clean dies to beams) |

Combat notes:

- Type **0x24**, HP starts 64 (wood sword chips); `AliveRule.TYPE_AND_HP`
- Overlap at west door (~16,141) stalls kills — backstep when dist under 16 without progress
- Key collect near center after clear (~136,141); FIXED_INVENTORY like L2 0x6c
- Post-clear: `cur_opened_doors=0`, `open_doorway_mask=0` — no kill-doors open;
  LEFT still returns to **0x79**

```bash
uv run python nes/zelda_i/scripts/run_level6_east_key.py --infinite-life --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level6_east_key.py --from-state L6Room_7a --infinite-life --trials 2
```

### West wizzrobes pure (Level6EastKey → 0x78 clear) — **assisted 2/2**

| Field | Live |
|-------|------|
| Start | `Level6EastKey` (0x7a keys≥1) or `L6Room_79_keys1` |
| Path | free LEFT → 0x79 → key LEFT (fire-bypass) → 0x78 |
| Spec | `level6_dungeon.ROOM_78_SPEC` |
| Combat | `Level6WestWizzrobeController` (same backstep as east) |
| Stop | `level6_room_78_clear_success` — room 0x78, no live 0x24 |
| Checkpoint | `Level6WestWizzrobes.state` — room **0x78**, cleared |
| Runner | `scripts/run_level6_west_wizzrobes.py --infinite-life --trials 2 --save-state` |
| Track | **assisted pure** |
| Post-clear | `doors=0x01` (RIGHT), `mask=0x09` (R+U) — **UP → 0x68** |

```bash
uv run python nes/zelda_i/scripts/run_level6_west_wizzrobes.py --infinite-life --trials 2 --save-state
```

### Compass room enter (0x78 leftover → 0x68) — **spine 1/1**

| Field | Live |
|-------|------|
| Start | Survival leftover `0x78` `(144,141)` after west clear |
| Path | OccupancyWalker to north door `(120,93)`; UP on band y≤109 |
| Stop | `--through level6-compass` play-ready `0x68` |
| Leftover | `(120,205)` south mouth; 5× Zol `0x13` live; north door open |
| Controller | `level6_path.Level6North68Controller` |
| Track | **assisted Survival** |
| Notes | 8 miss-blocks UP at x=144 (statue column); no door poke |

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-compass --no-video --trials 1
```

### Compass pickup (0x68 Zol clear) — **spine 1/1**

| Field | Live |
|-------|------|
| Start | Survival leftover `0x68` `(120,205)` |
| Combat | occupancy-patrol GenericDungeonRoomController; wooden-sword splits |
| Stop | `--through level6-clear68` — no live Zol/gel + `ADDR_COMPASS & 0x20` |
| Leftover | `(120,149)` center; compass HUD; north door open |
| Ignore | invuln `0x2b` / block `0x68` |
| Track | **assisted Survival** |

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-clear68 --no-video --trials 1
```

### Keese room enter (0x68 leftover → 0x58) — **spine 1/1**

| Field | Live |
|-------|------|
| Start | Survival leftover `0x68` `(120,149)` after compass |
| Path | OccupancyWalker to north door; 10 miss-blocks on x=120 column |
| Stop | `--through level6-keese` play-ready `0x58` |
| Leftover | `(120,205)` south mouth; 8× Keese live; north door sealed |
| Track | **assisted Survival** |

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-keese --no-video --trials 1
```

### Keese clear (0x58) — **spine 1/1**

| Field | Live |
|-------|------|
| Start | Survival leftover `0x58` `(120,205)` |
| Combat | occupancy-patrol, TYPE-only 8× Keese `0x1b` |
| Stop | `--through level6-clear58` no live Keese |
| Leftover | `(112,167)`; keys=5 (no key); green rupee SW; **north sealed** |
| Ignore | invuln `0x2b` / block `0x68` / corner fires |
| Track | **assisted Survival** |

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-clear58 --no-video --trials 1
```

### Blade-trap enter (0x58 leftover → 0x48) — **spine 1/1**

| Field | Live |
|-------|------|
| Start | Survival leftover `0x58` `(112,167)` after Keese clear |
| Path | OccupancyWalker long-UP; boxed `(112,167)` then north door |
| Stop | `--through level6-room48` play-ready `0x48` |
| Leftover | `(120,205)` south mouth; 4× blade trap `0x49`; north open |
| Keys | 5→5 (**free**, not a key door) |
| Track | **assisted Survival** |

Clear58 leftover PNG shutter is walkable. Do not treat that as sealed.

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-room48 --no-video --trials 1
```

### Hard-room enter (0x48 leftover → 0x38) — **spine 1/1**

| Field | Live |
|-------|------|
| Start | Survival leftover `0x48` `(120,205)` |
| Path | OccupancyWalker run-UP; do not fight blade traps `0x49` |
| Stop | `--through level6-room38` play-ready `0x38` |
| Leftover | `(120,189)`; Like-Like + wizzrobe live; north open |
| Track | **assisted Survival** |

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-room38 --no-video --trials 1
```

### Hard-room clear (0x38) — **spine 1/1**

| Field | Live |
|-------|------|
| Start | Survival leftover `0x38` `(120,189)` |
| Combat | occupancy-patrol GenericDungeonRoomController; 7 live (2 orange `0x24` + 2 blue `0x23` + 3 Like-Like `0x17`) |
| Stop | `--through level6-clear38` no live wizzrobe/Like-Like |
| Leftover | `(32,149)` west; Bubble `0x40` residual; two center blocks unpushed; north **looks** open, is sealed until left 0x68 moves |
| Ignore | invuln `0x2b` / block `0x68` / Bubble (sword-immune) |
| Track | **assisted Survival** |

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-clear38 --no-video --trials 1
```

### Hard-room north (0x38 leftover → 0x28) — **spine 1/1**

| Field | Live |
|-------|------|
| Start | Survival leftover `0x38` `(32,149)` |
| Path | west-clip inland; live left 0x68 slot11 `(96,144)` south-face UP until y moves; west aisle x=64 then north door `(120,93)` |
| Stop | `--through level6-room28` play-ready `0x28` |
| Leftover | `(120,189)` south mouth; orange wizzrobes live |
| Track | **assisted Survival** |
| Notes | v2 UP at north shutter never left (sealed until 0x68 y changes). v3 200f UP is not a push. v4 occupancy from the push plane boxed. v5 center-UP @ x=120 hits the pair. |

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-room28 --no-video --trials 1
```

### Wizzrobe clear (0x28) — **spine 1/1**

| Field | Live |
|-------|------|
| Start | Survival leftover `0x28` `(120,189)` |
| Combat | occupancy-patrol GenericDungeonRoomController; 2 orange `0x24` |
| Stop | `--through level6-clear28` no live orange wizzrobes |
| Leftover | `(120,181)` south-inland; diamond floor walkable (0 occupancy misses) |
| Ignore | invuln `0x2b` / block `0x68` / Bubble `0x40` |
| Track | **assisted Survival** |

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-clear28 --no-video --trials 1
```

### Gleeok enter (0x28 leftover → 0x18) — **spine 1/1**

| Field | Live |
|-------|------|
| Start | Survival leftover `0x28` `(120,181)` after orange clear |
| Path | LEFT+UP at y=181 (cardinal UP solid); hold UP from `(96,173)`; RIGHT+UP at y=109 (cardinal RIGHT solid) |
| Stop | `--through level6-room18` play-ready `0x18` |
| Leftover | `(120,189)` south mouth; Gleeok 3-head live |
| Keys | 5 (no spend) |
| Track | **assisted Survival** |
| Notes | v1 occupancy boxed leftover. v2 hold-UP y stays 181. v3 LEFT to x=80 works; UP at y=181 solid. v4 peel y=189 then aisle UP walks to 181 then solid. v5 LEFT+UP clips to `(96,173)`; RIGHT solid. v6 hold-UP to `(96,109)`; RIGHT solid. v7 RIGHT+UP at y=109 enters. Do not fight Gleeok this hop. |

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-room18 --no-video --trials 1
```

### Post-east-key graph (live recon)

Probe: `scripts/probe_level6_past_east_key.py --infinite-life --try-old-man`.

| Edge | Type | Keys | Live |
|------|------|------|------|
| 0x7a → LEFT → 0x79 | free | 0 | **yes** |
| 0x7a → UP → 0x6a | **key** | −1 | **yes — Old Man trap** |
| 0x7a → RIGHT/DOWN | sealed | — | yes |
| 0x79 → LEFT → 0x78 | **key** | −1 | **yes** (fire-bypass) |
| 0x79 → RIGHT → 0x7a | free | 0 | yes |
| 0x79 → UP | sealed brick | — | yes |
| 0x79 → DOWN | OW exit | — | residual (south mouth) |
| 0x78 → UP → 0x68 | kill-door after clear | 0 | **yes** |
| 0x78 → RIGHT → 0x79 | free | 0 | yes |
| 0x68 → UP → 0x58 | free/after clear | 0 | recon |
| 0x58 → UP → 0x48 | **free** (PNG shutter is walkable) | 0 | spine 1/1 |
| 0x48 → UP → 0x38 | **free** (run traps, no clear) | 0 | spine 1/1 |
| 0x38 → UP → 0x28 | **left 0x68 UP** then west-aisle (shutter sealed until y moves) | 0 | spine 1/1 |
| 0x28 → UP → 0x18 | LEFT+UP y=181, hold UP, RIGHT+UP y=109 | 0 | spine 1/1 |

### Walkthrough (not all live)

- RIGHT wizzrobes + key; LEFT locked (do not waste key on Old Man first) — **RIGHT/LEFT/Old Man match live**
- Compass from Zols — **0x68 live** (RoomItemId `0x16`; inventory pickup residual)
- statue/Keese rooms — **0x58 live**
- multi-Wizzrobe + Bubble + Like-Like — **0x38 live clear** (Bubble residual)
- Mid-dungeon **Gleeok (3 heads)** then Map — **0x18 enter live**; fight residual
- Staircase → **Magical Rod** (`ADDR_ROD=0x065F`) — residual
- Vires / Wizzrobes → staircase → **Gohma** (one arrow to open eye) — residual (rr-d6v)
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
| `L6Room_7a.state` | East key room after wall-first RIGHT (enemies live) |
| `Level6EastKey.state` | Assisted pure: 0x7a cleared + keys 0→1 |
| `L6Room_79_keys1.state` | 0x79 with keys≥1 after free return from 0x7a |
| `L6Room_78.state` / `Level6WestWizzrobes.state` | West wizzrobe room enter / cleared |
| `L6Cleared78.state` | 0x78 clear dev fixture |

## Evidence

- `recordings/l6_entry_recon.json`
- `recordings/level6_east_key_assisted_isolated.json` — 2/2 from `Level6Entrance`
- `recordings/level6_west_wizzrobes_assisted_isolated.json` — 2/2 from `Level6EastKey`
- `recordings/l6_post_key_graph.json` — door map + north chain recon
- `recordings/l6_entry_continuous_v2.json` — power-on → L6 `0x79` 1/1
- `recordings/l6_east_key_continuous_v1.json` — 0x7a keys 5→6 1/1
- `recordings/l6_west_continuous_v1.json` — 0x78 clear 1/1
- `recordings/l6_compass_continuous_v1.json` — 0x68 enter 1/1 occupancy UP 221f
- `recordings/l6_clear68_continuous_v1.json` — 0x68 compass 1/1 occupancy-patrol 4,939f
- `recordings/l6_keese_continuous_v1.json` — 0x58 enter 1/1 occupancy UP 209f
- `recordings/l6_clear58_continuous_v1.json` — 0x58 Keese clear 1/1 882f
- `recordings/l6_room48_continuous_v1.json` — 0x48 enter 1/1 occupancy UP 341f; keys 5
- `recordings/l6_room38_continuous_v1.json` — 0x38 enter 1/1 occupancy run-UP 261f
- `recordings/l6_clear38_continuous_v1.json` — 0x38 occupancy-patrol clear 1/1 5,487f
- `recordings/l6_room28_continuous_v6.json` — 0x28 enter 1/1 left-0x68 UP + west-aisle 3,207f
- `recordings/l6_clear28_continuous_v1.json` — 0x28 occupancy-patrol 2× orange 0x24 1/1 2,587f
- `recordings/l6_room18_continuous_v7.json` — 0x18 Gleeok enter 1/1 280f leftover `(120,189)`
- `recordings/l6_entrance_live.png`, `l6_ow_22.png`, `l6_room_7a.png`, `l6_0x6a.png`
- Spine: `uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-room18 --no-video --trials 1`
- Probe: `uv run python zelda_i/scripts/probe_level6_entry.py --infinite-life --save-state`
- Graph: `uv run python nes/zelda_i/scripts/probe_level6_past_east_key.py --infinite-life --try-old-man`
- Pure: `uv run python nes/zelda_i/scripts/run_level6_east_key.py --infinite-life --trials 2`
- Pure: `uv run python nes/zelda_i/scripts/run_level6_west_wizzrobes.py --infinite-life --trials 2`

## Residual to Rod / Gohma (rr-d6v)

Not claimed live as pure segments:

1. **0x68 compass** pickup (`ADDR_COMPASS|0x20`) after Zol clear — **live**
2. **0x58** Keese clear — **live**; key inventory no
3. **0x48** enter + run-UP — **live**
4. **0x38** clear — **live**
5. **0x28** enter + orange-wizzrobe clear — **live**
6. **Gleeok (3 heads)** enter `0x18` — **live**; fight + Map residual
7. Staircase → **Magical Rod** (`ADDR_ROD`)
8. Vire / wizzrobe path → Gohma arrow → Heart → TF `0x20`

## Not claimed

- Clean STATUS / natural-entry from real predecessor TF bits
- Clean east/west wizzrobe combat (beams kill without assist)
- Full walk hop table from `0x77` / post-L1 (post-L5 `0x0B` → `0x79` is live)
- Bracelet warp live
- Gleeok fight / Rod / Gohma / triforce bit live
