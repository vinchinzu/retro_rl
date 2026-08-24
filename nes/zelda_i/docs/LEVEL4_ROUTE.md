# Level 4 — The Snake (route notes)

**Status:** continuous power-on Survival is live through play `0x31` with
`ADDR_LADDER` (`l4_west31_continuous_v1`, leftover `(208,141)`). South-U
around the pushed 0x68 from 0x32 SE stairs. Isolated BFS is still not this
tape. Next: reverse 0x31 maze → `0x30` KEY-UP `0x20`. Do not claim Clean
STATUS — Survival assist only for this segment. Do not close `.6` until
TF `0x08`.

**Beads:** `rr-0fx` Z4.1 live entry (done); `rr-5lu` interior residual;
epic `rr-q3n`.

Planning sources (external, not emulator facts):

- [Zelda Dungeon — Level 4: The Snake](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-4-the-snake/)
- Local archive: [research/DUNGEON_WALKTHROUGHS.md](research/DUNGEON_WALKTHROUGHS.md)
- RAM: `zelda_i/ram.py` (`ADDR_RAFT`, `ADDR_LADDER`, `ADDR_TRIFORCE`)

Every screen id and room id is **source-hypothesized** unless marked
**(live)**.

---

## Gates / required capabilities

| Cap | RAM | Source role |
|-----|-----|-------------|
| **Raft** | `ADDR_RAFT` (`0x0660`) ≠ 0 | Hard gate: dock → island only with Raft from L3 |
| Sword | `ADDR_SWORD` ≥ 1 | Combat |
| Bombs (helpful) | `ADDR_BOMBS` | Optional wall skips / Manhandla |
| Blue Candle (helpful) | `ADDR_CANDLE` | Dark rooms |
| **Stepladder** (dungeon item) | `ADDR_LADDER` (`0x0663`) | Cross water tiles inside L4 (and later OW) |
| Triforce shard 4 | `ADDR_TRIFORCE & 0x08` | Clear stop |

**Predecessor:** Level 3 Manji drops Raft. **Do not** poke `0x0660` for
Clean STATUS or published pure-first evidence.

**Optional OW prep (source):** east-coast Raft Heart Container — from start
east 8, north 4 to dock, walk onto dock (Raft carries north into cave).
Choose Heart over potion.

---

## Overworld

### Live path (assisted 2026-08-08, rr-0fx)

Start: checkpoint **`Level3Complete`** (mode 18, room 0x3d, `raft=1`,
`tf&0x04`). Fanfare settles to OW **`0x74`** ~(128,125).

```
0x74 W@y141 → 0x73 free mid x≈128 → N → 0x63 free south
  → E@y≈145–155 → 0x64 E@y141 → 0x65 N@x112 → dock 0x55
  → N@x≈128 (Raft) → island 0x45 → door UP @x128 → level 4 room 0x71
```

| Landmark | Id | Live? | Notes |
|----------|-----|-------|-------|
| Post-L3 return | **`0x74`** | **live** | Same as L3 door mouth |
| Raft dock | **`0x55`** | **live** | South entry y=221; raft only x≈128 |
| Island door | **`0x45`** | **live** | UP into dungeon |
| Entry room | **`0x71`** | **live** | South mouth ~(120,205) mode 5 |
| East-coast Raft heart dock | `0x3F` | no | Source only |
| Raft heart cave | `0x2F` | no | Source only |

**Traps (live):**

- `0x73`: arrive east edge from 0x74 — free mid before UP.
- `0x63` east: **y∈[145,155]** only. y=141 sticks in bush near x≈144.
- Dock `0x55`: UP at x≤112 never boards; align **x≈128** then UP.
- Do not poke Raft. Not Clean STATUS.

**Module:** `level4_overworld.py` — `LEVEL4_HOPS_FROM_POST_L3`,
`OverworldToLevel4Controller`, `PostL3TriforceSettleController`.
`SOURCE_HYPOTHESIS = False`.

### Runner

```bash
# 2/2 assisted entry + Level4Entrance.state
uv run python nes/zelda_i/scripts/run_level4_entry.py --infinite-life --trials 2 --save-state

# Dock only → OW_L4Dock.state
uv run python nes/zelda_i/scripts/run_level4_entry.py --infinite-life --dock-only --save-state

# Plan dry-run
uv run python nes/zelda_i/scripts/run_level4_entry.py --plan-only
```

Evidence: `recordings/l4_entry_recon.json` (**2/2 assisted**, ~2173f/trial).
Checkpoints: **`Level3ExitOverworld`**, **`OW_L4Dock`**, **`Level4Entrance`**.

---

## Interior (live pure dual-green — rr-5lu / rr-2ysf 2026-08-09/10)

Module: `level4_dungeon.py`. Runner:
`scripts/run_level4_rooms.py`. Evidence:
`recordings/l4_chain_key_pure_chain_to_key.json` (**2/2 pure** ~1278f),
`recordings/l4_clear50_pure_clear_50.json` (**2/2 pure** ~2478f),
`recordings/l4_keyright62_pure_key_right_62.json` (**2/2 pure** ~1133f),
`recordings/l4_clear62_pure_clear_62.json` (**2/2 pure** ~11536f),
`recordings/l4_compass62_pure_compass_62.json` (**2/2 pure** ~471f,
ADDR_COMPASS|0x08 + return 0x61). Not Clean STATUS promote.

### Live graph (from `Level4Entrance`)

```
0x71 entry (empty combat, item 0x03)
  --UP @ x≈120--> 0x61
0x61: 3× Vire type **0x12** (HP 64) → sword split type **0x1c** (slots 10–12)
  --BOMB_UP stand≈(120,105) face UP--> 0x51
0x51: 8× Keese type **0x1b** (TYPE-only) + RoomItemId **0x19** key (keys 0→1 @~136,149)
  --LEFT @ y≈141--> 0x50
0x50: 5× Vire **0x12**  (RIGHT→0x51; scripted N→0x40)
0x51 --DOWN @ x≈120--> 0x61
0x61 --KEY-RIGHT @ y≈141 (keys 1→0)--> 0x62
0x62: 5× Vire + RoomItemId **0x16** Compass (dark maze)
0x50 --coordinate-gated N via observed bands + long UP--> **0x40**
0x40: 5× Zol **0x13** → gel **0x14** + key **0x19** (east-corridor path)
0x40 --free UP @x≈120--> **0x30**: 3× Vire + 2× invuln **0x2b**
0x30 --clear (ignore 0x2b; north-band y≥128)--> KEY-RIGHT @y141 --> **0x31**
0x31 --clear maze Vires --> free RIGHT --> **0x32**
0x32 --clear Zol+LikeLike --> push left block --> stairs **0x60** --> **ADDR_LADDER**
0x60 --reverse dock waypoints --> **0x32** play (ladder set)
```

| Room | Live? | Enemies | Item / notes | Segment bead |
|------|-------|---------|--------------|--------------|
| **0x71** | **live pure 2/2** | none | Empty mouth; free UP only | `rr-zchy` |
| **0x61** | **live pure 2/2** | 3× `0x12` → split `0x1c` | Clear ~295f; bomb N → 0x51; KEY-RIGHT → 0x62 | `rr-yr77` / `rr-h278` |
| **0x51** | **live pure 2/2** | 8× `0x1b` Keese | Key `0x19` pickup ~ (136,149) | `rr-wqdu` |
| **0x50** | **live pure 2/2; continuous 1/1** | 5× `0x12` Vire | North via coordinate gates → 0x40 (not dead-end) | `rr-2ysf` / `rr-xc3x` |
| **0x62** | **live pure enter+clear+compass 2/2** | 5× `0x12` Vire | Compass `0x16` dark maze; pickup ~(136,132); return LEFT→0x61 | `rr-2ysf` / `rr-9so0` |
| **0x40** | **live pure clear+key 2/2; continuous 1/1** | 5× `0x13` → `0x14` | Key path hold6 east corridor; free UP → 0x30 | `rr-xc3x` / `rr-q8eq` |
| **0x30** | **live pure clear+KEY-R 2/2; continuous clear+KEY-R 1/1** | 3× `0x12` + 2× `0x2b` | Walkable y≥128; clear from (120,205); KEY-RIGHT @y141 → 0x31 | `rr-q8eq` / `rr-n1wn` |
| **0x31** | **live pure clear+RIGHT 2/2; continuous clear+RIGHT 1/1** | 5× `0x12` Vire | Maze; leftover (112,141); clear opens R; free RIGHT → 0x32 at (16,141) | `rr-n1wn` / `rr-resv` |
| **0x32** | **live pure clear 2/2; continuous enter 1/1** | 2× `0x13` + 2× `0x17` | Ignore 0x2b/0x68; leftover (16,141); push left → stairs | `rr-tib8` |
| **0x60** | **live pure 2/2; continuous 1/1** | 4× `0x1b` Keese | mode-9 basement; east-dock waypoints → `ADDR_LADDER` at (136,141); reverse dock → 0x32 play (192,189) | `rr-tib8` |

### Post-compass expand (rr-o0nn / rr-xc3x live 2026-08-10)

Start: **`Level4Compass`** (0x61, `ADDR_COMPASS|0x08`, keys=0, doors=1 RIGHT).

Early component was closed at `{0x71, 0x61, 0x51, 0x50, 0x62}` until **0x50 north** opened:

| From | Exit | Dest | Notes |
|------|------|------|-------|
| 0x61 | free/BOMB UP | 0x51 | hole still open post-compass |
| 0x61 | RIGHT | 0x62 | re-enter without key (door bit stays) |
| 0x61 | DOWN | 0x71 | free |
| 0x51 | LEFT @y141 | 0x50 | free |
| 0x51 | DOWN | 0x61 | free |
| 0x51 | **UP** | **sealed** | not a key door (keys poke does not consume) |
| 0x51 | **RIGHT** | **sealed** | same |
| 0x50 | RIGHT | 0x51 | free |
| 0x50 | **UP scripted** | **0x40** | coordinate gates `(160,181)→(112,181)→(112,120)→(128,100)`; UP to y≈93, LEFT to x≈120, then long UP |
| 0x62 | LEFT | 0x61 | only durable exit; bomb stands no open |
| 0x40 | DOWN | 0x50 | free return |
| 0x40 | **UP free** | **0x30** | after clear; x≈120 (rr-q8eq) |
| 0x40 | LEFT/RIGHT | **sealed** | live probe |
| 0x30 | DOWN | 0x40 | free return |
| 0x30 | **KEY-RIGHT @y141** | **0x31** | keys 1→0; 5× Vire (rr-n1wn) |
| 0x30 | UP / LEFT / free RIGHT | **sealed** | live probe |
| 0x31 | LEFT | 0x30 | free return after key door |
| 0x31 | **RIGHT after clear** | **0x32** | doors 2→3; continuous: UP y=113, RIGHT+DOWN clip, south-U waypoints (not state-BFS) |
| 0x32 | LEFT | 0x31 | free return |
| 0x32 | **push left block** | **0x60** stairs | mode-9 basement (rr-tib8) |
| 0x60 | **reverse dock waypoints** | **0x32** play | continuous v2 leftover (192,189); ladder set |
| 0x32 | N/E/W free | **sealed** | live probe |

Also live-negative: Vire re-clear key farm (8 cycles) **no drops**.

**ADDR_LADDER = 1** after 0x60 pickup (rr-tib8 pure 2/2). Evidence:
`recordings/l4_tib8_clear32_clear_32.json`,
`recordings/l4_tib8_stepladder_stepladder.json`,
`recordings/l4_resv_room32_recon.json`.

### Runner

```bash
# Continuous power-on verified through 0x32 Zol+LikeLike clear
uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level4-clear32 --no-video --trials 1
# Stepladder through-stop verified v34 (ADDR_LADDER at (136,141))
uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level4-stepladder --no-video --trials 1 \
  --tag l4_stepladder_continuous_v34
# Exit 0x60 → 0x32 play verified v2 (ladder leftover (192,189))
uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level4-exit60 --no-video --trials 1 \
  --tag l4_exit60_continuous_v2
# West 0x32 → 0x31 verified v1 (leftover (208,141))
uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level4-west31 --no-video --trials 1 \
  --tag l4_west31_continuous_v1
# Pure dual-green room segments (no --infinite-life)
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment entry_up --trials 2
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment clear_61 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment bomb_61 --from-state Level4Entrance --trials 2
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment key_51 --from-state Level4Entrance --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment chain_to_key --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment clear_50 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment key_right_62 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment chain_to_62 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment clear_62 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment compass_62 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment north_40 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment key_40 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment north_30 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment clear_30 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment key_right_31 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment clear_31 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment east_32 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment clear_32 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment stepladder --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment exit_60 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment west_31 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment map_21 --infinite-life --trials 2 --save-state
# Natural key (no recon poke) from skip-compass checkpoint
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment map_21 \
  --from-state Level4Room31PostLadderNaturalKey --infinite-life --no-key-poke --trials 2
# Continuous natural PostLadder → TF 0x08 (assisted first-pass)
uv run python nes/zelda_i/scripts/run_level4_continuous_tf.py \
  --from-state Level4Room31PostLadderNaturalKey --infinite-life --trials 2 --save-state
# Clean continuous (rr-vdnc dual-green; no assist)
uv run python nes/zelda_i/scripts/run_level4_continuous_tf.py \
  --from-state Level4Room31PostLadderNaturalKey --trials 2 --tag l4_vdnc_clean_cont_tf
# Clean Gleeok-only smoke
uv run python nes/zelda_i/scripts/run_level4_gleeok.py --trials 2 --tag l4_vdnc_gleeok_clean
```

### Post-ladder (rr-05fz pure + natural continuous 2026-08-10)

Start: **`Level4Stepladder`** (mode 9 room **0x60**, `ADDR_LADDER=1`, pedestal
~(136,141)).

```
0x60 settle ~150f idle (item freeze) → clear 4× Keese 0x1b
  → hold4 BFS exit → 0x32 play  (Level4PostLadder)
0x32 free LEFT (BFS around pushed 0x68) → 0x31
0x31 LEFT → 0x30
0x30 KEY-UP (ladder + keys≥1) → 0x20 clear Vires
0x20 state-BFS RIGHT → 0x21 gels + map 0x17 → ADDR_MAP|0x08
```

| Segment | Evidence | Frames (typ.) | Checkpoint |
|---------|----------|---------------|------------|
| `exit_60` pure 2/2 | `l4_05fz_exit60_exit_60.json` | ~765 | `Level4PostLadder` |
| `west_31` pure 2/2 | `l4_05fz_west31_west_31.json` | ~372 | `Level4Room31PostLadder` |
| `map_21` assisted 2/2 | `l4_rvae_map21_map_21.json` | ~17872 | `Level4Map` |

### Gleeok approach from Level4Map (rr-rvae; enter + TF dual-green 2026-08-10)

Live graph (maze BFS + bomb stands; assisted). **Gleeok enter dual-green;
melee + HC + TF `0x08` dual-green from `Level4GleeokEnter`.**

```
0x21 map (Level4Map)
  --maze BFS LEFT--> 0x20 --free UP--> 0x10 Manhandla 0x3c --UP--> 0x00 bubbles (dead-end)
  --BOMB_UP @(120,105)--> 0x11 type 0x35 cluster
       --UP--> 0x01 8× Keese + key 0x19  (natural key for map KEY-UP residual)
       --RIGHT--> 0x12 5× Vire + block 0x68
            --UP--> 0x02 blade traps 0x49 (dead-end)
            --clear + push 0x68 LEFT @(112,144)--> doors 2→3
            --hold4 PATH_12_TO_GLEEOK plen31 RIGHT--> 0x13 Gleeok 0x43 + HC 0x1a
       --LEFT--> 0x10 Manhandla
```

| Room | Live? | Notes |
|------|-------|-------|
| **0x11** | **live** | BOMB_UP from map; type `0x35`; checkpoint `Level4Room11` |
| **0x01** | **live** | Keese + key `0x19` (keys 0→1 after clear) |
| **0x12** | **live dual** | 5× Vire; push block LEFT opens R; `Level4Room12Cleared` |
| **0x02** | **live** | traps only; DOWN→0x12 |
| **0x13** | **live dual enter + kill** | Gleeok `0x43` HP≈160 + head `0x46` + HC; UP → 0x03 TF |
| **0x10** | **live** | Manhandla side path; UP→0x00 dead-end |
| **0x00** | **live** | bubbles; only DOWN |

**Live dual TF (rr-rvae 2026-08-10):** from `Level4GleeokEnter` melee A-spam
(prefer head `0x46` when present) → boss dead ~3.6k f → HC containers+1 → free
UP → **0x03** → mid walk → **`tf&0x08`**. Evidence:
`recordings/l4_rvae_gleeok_tf_dual.json` (2/2 dual exact ~4.3k f). Runner:
`scripts/run_level4_gleeok.py --infinite-life --trials 2 --save-state`.
Checkpoint **`Level4Complete`**. Module: `level4_boss_combat.py`.

**Residual closed (rr-05fz assisted dual-green 2026-08-10):** natural KEY-UP
via **skip-compass** spare key (`Level4Room31PostLadderNaturalKey`, keys≥1,
`map_21 --no-key-poke` 2/2) + continuous PostLadder→map→Gleeok→TF 2/2
(~34.7k f). Evidence: `l4_05fz_map_natural_map_21.json`,
`l4_05fz_postladder_cont_tf.json`, `l4_05fz_map_to_tf.json`. 0x01 Keese key
still available after map BOMB_UP. **Not Clean STATUS** (assist still on).

**Clean residual closed (rr-vdnc 2026-08-10):** continuous
`Level4Room31PostLadderNaturalKey` → map no-poke → Gleeok → TF `0x08` **2/2
Clean** (no `--infinite-life`, `key_poke=false`) ~33.9k f/trial. Gleeok south-stand
policy: approach south y≥165, hold `(body.x, body.y+22)` face UP+A, fireball
horizontal dodge only dist≤14; **do not chase** detached heads while body residual
remains (head kite dies Clean; south stand clears faster/safer). Evidence:
`l4_vdnc_clean_cont_tf.json` (dual_green, track=clean),
`l4_vdnc_gleeok_clean_dual.json` (GleeokEnter-only Clean dual ~1.6k f). Runner:
`run_level4_continuous_tf.py` **without** `--infinite-life`;
`run_level4_gleeok.py` without assist. Module: `level4_boss_combat.py`
(`STAND_DY=22`). **Not full-game Clean STATUS** (lab checkpoint continuous).

**Natural-entry compose PARTIAL (rr-zavx 2026-08-10):** Clean dual-green
**Entrance → skip-compass NaturalKey** (no compass KEY-RIGHT; keys≥1 for
map KEY-UP) **2/2** ~45.8k f/trial. Runner:
`scripts/run_level4_entrance_tf.py --to-natural-key-only`. Evidence:
`recordings/l4_zavx_natkey_dual.json`. Spine segments pure:
`chain_to_key` → `clear_50` → `north_40` → `key_40` (ALIGN to path anchor
before maze; skip-compass clear pose ~(72,125) was missing key) →
`north_30`…`west_31` → `Level4Room31PostLadderNaturalKey` (ladder=1, keys=1,
health≈103).

**Continuous Entrance→TF Clean residual (rr-gjey PARTIAL 2026-08-10):** full
compose runner wired (`run_level4_entrance_tf.py` without
`--to-natural-key-only`) but **not dual-green** from natural health.

Lab cliffs (7 containers, full health byte `0x6F=111`):

| Start | Path | Floor | Notes |
|-------|------|-------|-------|
| `Level4GleeokEnter` poke | fight only | **≥106** dual TF | stock mid-fight; **post-boss residual fireball** must 2D-flee (idle/goto died at 106) |
| `Level4Map` poke → path | continuous | **≥108** dual TF | approach costs more than GleeokEnter vestibule |
| Natural spine | Entrance→NaturalKey | ends **~103** | peels: clear_30−2, clear_32−2, chain/key_40/clear_31/west −1 each |
| Natural map_21 | PostLadder→Map | peels **~3** | `L4:0x31/0x30/0x20` one each (assist heatmap) |
| Natural enter Gleeok | map→0x13 | **~98–100** | clear12 often −0..2; fight death mid-approach |

**rr-gjey pins:** post-boss residual fireball care in
`level4_boss_combat.py` (lateral flee while any `0x56` present; no long
unprotected idle/goto). Evidence:

- `l4_gjey_gleeok_clean_dual.json` — GleeokEnter full Clean dual (reg)
- `l4_gjey_gleeok_hp106_dual.json` — GleeokEnter **poke 106** dual TF (floor drop)
- `l4_gjey_map108_floor_dual.json` — Map **poke 108** → path → TF dual (continuous floor)
- Continuous enter settle shortened (no 40f idle on vestibule; fireballs
  inflate approach cost)

Pure spine still ends ~103; natural enter ~98–100 **below continuous floor
108**. Gel thrash does **not** reliably drop hearts. Heart-safe spine polish
(clear_30/32/map peels) or mid-fight damage cut still open. Epic **rr-q3n**
residual; **not Clean STATUS**. Assisted continuous PostLadder→TF remains
dual-green (rr-05fz). GleeokEnter-only Clean dual remains green (rr-vdnc).

```bash
# Clean dual skip-compass NaturalKey from Entrance (rr-zavx pin)
uv run python nes/zelda_i/scripts/run_level4_entrance_tf.py \
  --to-natural-key-only --trials 2 --save-state --tag l4_zavx_natkey_dual
# Full compose (spine + continuous TF; Clean dual still residual rr-gjey)
uv run python nes/zelda_i/scripts/run_level4_entrance_tf.py \
  --trials 2 --save-state --tag l4_gjey_entrance_tf
# Gleeok Clean reg + floor poke pins
uv run python nes/zelda_i/scripts/run_level4_gleeok.py --trials 2 --tag l4_gjey_gleeok_clean
```

**Traps (0x12→0x13):** after clear doors often L-only (raw=2); **bomb RIGHT and
KEY-RIGHT do not open 0x13**; push block 0x68 LEFT first; naive y141 hold-RIGHT
fails (maze) — use `PATH_12_TO_GLEEOK` hold4; if path sticks at east wall
y≈149, align y141 + live BFS exit (rr-zavx).

Map → Gleeok compose: `scripts/run_level4_continuous_tf.py` / `run_level4_gleeok.py`.

**Traps (post-ladder live):**

- Pedestal freeze: **~100–150 idle** after loading `Level4Stepladder` before
  any movement (1–50 idle = stuck).
- Exit BFS must **settle through mode 4/6/7** (~400f) — 180f leaves mode 4 on
  dest room and false-negatives the exit.
- Pushed block **0x68** blocks naive west door; use hold4 BFS path.
- 0x30 free N sealed; **KEY-UP** with ladder + key → **0x20** (recon key poke
  when checkpoint keys=0).
- KEY-UP from **0x31** enters **0x21 south pocket** only (isolated; not map).
- Map room **0x21**: RoomItemId **0x17**, 5× Gel **0x15**; thrash expands maze
  then hold6 BFS pickup ~(208,181). 0x20 east needs state-BFS (door bit R=0).

**Traps (live):**

- Source “entry LEFT Keese key” is **wrong** on this seed/path — entry is empty; first key is bomb-N of Vires.
- **0x50 is NOT a dead-end** — north exit to **0x40** uses the live coordinate
  gates through `(128,101)`, then UP to y≈93, LEFT to x≈120, and long UP;
  naive center+UP fails on interior blocks (rr-xc3x).
  Compass remains KEY-RIGHT 0x61→0x62.
- Vire split is type **`0x1c`**, not standard Keese `0x1b`; HP stays 0 (type-only) and lands in slots **10–12**.
- Free doorways often show `cur_opened_doors=0` / `open_doorway_mask=0` — do not require door bits for UP 0x71→0x61 or LEFT 0x51→0x50.
- Bomb stand on 0x61: **(120, ~105)** face UP + B; wait blast then push UP.
- Key item id is **0x19 from room entry** (not drop-after-clear); walk mid-room after Keese clear.
- KEY-RIGHT 0x61: hold **y≈141** RIGHT; keys 1→0; vestibule enter ~(16,141).
- **0x30** walkable band **y∈[128,208]** only (solid north wall). Clear Vires
  from north-band patrol face **UP**; ignore invuln **0x2b**. Free N/E/W sealed;
  KEY-RIGHT @y141 → **0x31** (rr-n1wn).
- KEY-RIGHT 0x30: hold **y≈141** RIGHT; keys 1→0; enter 0x31 ~(16,141).
- **0x32** clear Zol `0x13` + LikeLike `0x17` (ignore `0x2b`/`0x68`). Push stand
  detour around center statues → LEFT push → NE approach ~(208,96) UP into
  stairs. **0x60** mode-9: settle NW ~(48,77); multi-grid BFS + goal-state
  restore for `ADDR_LADDER`. Stepladder segment needs **5 idle** frames before
  clear (combat RNG).
- 0x62 **dark maze**: open seek fails; use scripted holds
  (`MAZE_62_TO_COMPASS` hold6 → pickup ~(136,132) sets `ADDR_COMPASS|0x08`,
  then `MAZE_62_RETURN_WEST` hold4 → LEFT scroll to 0x61). Center y=141 is
  wall-blocked from the west door — must follow the return corridor.
- Post-compass: 0x62 only durable exit is **LEFT→0x61**. Progress is **0x50 N→0x40**,
  not 0x51 UP (sealed).
- From `Level4Compass`: KEY-RIGHT door stays open (RIGHT re-enter 0x62, no key).
- **0x40 Zol→Gel**: wooden sword splits `0x13`→`0x14` (HP=0 type-only). Include gels
  in live set; `settle_all_dead=0`. Key is **not** mid-room-naive after clear —
  south pocket walls force east-corridor path `MAZE_40_TO_KEY` hold6 → ~(136,117)
  then free UP@x120 → **0x30**.

Checkpoints (dev): `Level4Room61`, `Level4Room61Cleared`, `Level4FirstKey`,
`Level4Room50Cleared`, `Level4Room62`, `Level4Room62Cleared`, `Level4Compass`,
`Level4Room40`, `Level4Room40Cleared`, `Level4Room30`, `Level4Room30Cleared`,
`Level4Room31`, `Level4Room31Cleared`, `Level4Room32`.

### Source speed route (planning only past compass — not emulator facts)

Room IDs **beyond 0x40** remain source-hypothesized until probed.

| Step | Action (source) | Notes |
|------|-----------------|-------|
| Past first key | KEY-RIGHT dark maze | **live** 0x62 Compass |
| Dark chain N | 0x50 N → 0x40 Zols+key | **live** rr-xc3x |
| Dark chain N cont. | water block | needs Stepladder |
| Like-Like + Zol | push left block → stairs | **Stepladder** |
| Gleeok (2 heads) | fireballs | E → TF `0x08` |

**Key item:** Stepladder (`ADDR_LADDER`).
**Boss:** Gleeok (2-head). Object type id **`0x43` live** (room **0x13**);
detached head **`0x46`**; fireball **`0x56`**.
**Triforce bit:** `0x08` **live dual-green** from `Level4GleeokEnter`.

### Policy notes

- Vire: wooden sword splits → `0x1c`; clear both generations.
- Keese 0x51: TYPE-only liveness (HP stays 0).
- Like-Like (later): stay out of contact (Magical Shield loss).
- Water tiles: after Stepladder, automatic on single-tile gaps.
- Gleeok: melee A-spam; prefer head `0x46` when present; body HP≈160 then
  TYPE-only residual; no bomb requirement (unlike Dodongo). UP after clear →
  TF room `0x03`.

---

## Boss / Triforce stop predicates (stubs)

```text
level4_boss_cleared  — body type 0x43 absent on 0x13 (heads/fireball residual ok)
level4_complete      — ADDR_TRIFORCE & 0x08  (mode 18 fanfare settle live)
```

`level4_triforce_stop(snap)` / `level4_complete_success(ram)`: inventory bit
`0x08` only (not a continuous natural-entry claim). Live dual-green from
`Level4GleeokEnter` via `Level4GleeokFightController`.

---

## Checkpoints

| State | When | Status |
|-------|------|--------|
| `Level3ExitOverworld` | Post-L3 fanfare settle OW 0x74 raft=1 | **live** |
| `OW_L4Dock` | Dock screen 0x55 | **live** |
| `Level4Entrance` | `level==4`, play mode, room 0x71 | **live** |
| `Level4FirstKey` | 0x51 keys≥1 after Keese clear | **live** |
| `Level4Room50Cleared` | 0x50 Vires clear (dead-end) | **live** |
| `Level4Room62` | KEY-RIGHT enter 0x62 vestibule | **live** |
| `Level4Room62Cleared` | 0x62 Vires clear (compass residual) | **live** |
| `Level4Compass` | after `ADDR_COMPASS & 0x08` | partial / residual |
| `Level4Stepladder` | after `ADDR_LADDER` (mode-9 0x60) | **live** |
| `Level4PostLadder` | exit 0x60 → 0x32 play ladder=1 | **live** |
| `Level4Room31PostLadder` | west of PostLadder on 0x31 | **live** |
| `Level4Map` | `ADDR_MAP & 0x08` on 0x21 | **live** (assisted) |
| `Level4Room11` | BOMB_UP from map 0x21 | **live** (recon) |
| `Level4Room12` | east of 0x11 Vires | **live** (recon) |
| `Level4Room01` | north of 0x11 Keese+key | **live** (recon) |
| `Level4Boss` / `Level4GleeokEnter` | 0x13 Gleeok vestibule | **live** (enter dual) |
| `Level4BossCleared` | after Gleeok (pre/post HC) | **live** (assisted) |
| `Level4Complete` | `triforce & 0x08` mode 18 | **live** dual-green 2/2 |

---

## Evidence

- `recordings/l4_entry_recon.json` — **2/2 assisted** entry from `Level3Complete`
- Checkpoints `Level4Entrance`, `OW_L4Dock`, `Level3ExitOverworld` (+ provenance)
- Related RAM: `ADDR_RAFT`, `ADDR_LADDER`, TF `0x08`
- **Not Clean STATUS**
