# L7 sitting leftover (rr-8t4.2, 2026-09-04)

Did not STATUS-promote. Did not edit `STATUS.md`. Bead `rr-8t4.2` stays
`in_progress`. Residual is this file, not `rr-tne2-residual.md`.
Did not run `bd`/`git` writes.

**Pin:** `Level7Entrance` — L7 play `0x79` `(120,205)` south mouth, mode 5,
whistle=1 (poked), food=0, TF=0, keys=0, bombs=0, 3 HC. Not the L6-leave
packet. Do not start at Hungry Goriya. Do not poke Food/Whistle/doors/TF.

## This sitting (2026-09-04): recon fixture + `0x6B` east GREEN 2/2

**New disclosed recon fixture — `Level7InteriorReconFixture`.**
`scratch/build_level7_interior_recon_fixture.py` **walks** the fixture-live
`0x79 → 0x69 → 0x6A → 0x6B` chain from `Level7Entrance`
(`EntryNorthDoorController` → `Room69EastController` → `Room6AEastController`,
no `set_state`/teleport), clears the six `0x6B` goriya `0x05`, then discloses
exactly three fixture writes: `ADDR_FOOD` `$065D` 0→1, `ADDR_BOMBS` `$0658`
0→8 (`min(8, max_bombs)`; `ADDR_MAX_BOMBS` read never written), `ADDR_KEYS`
`$066E` 0→4. Settled leftover: L7 play `0x6B` `(136,109)` mode 5, TF 0,
Candle 0, Whistle 1, `deaths=0`, `progression_writes=capacity_writes=0`.
Provenance `Level7InteriorReconFixture.provenance.json`: `development_only:
true`, `natural_entry: false`, `route_eligible: false`, `fixture_only: true`,
every write recorded with before/from/to. Traverse ran under the standard
`UnlimitedHealthAssist` (disclosed in notes as a traversal aid, not a
fixture write; final health byte unchanged at `0x22`). **Not** on any spine.

**`0x6B` (`GORIYA_HINT`) east is GREEN 2/2.** From the fixture,
`scratch/probe_l7_room6b_onward.py --dir RIGHT`: `0x6B` `(16→136,109)` →
ride the `y=109` band east past the central X of diamond blocks → drop the
east column to `y=141` → push the OPEN east doorway → **live dest `$EB=0x6C`
`(16,141)` west mouth, mode 5**, arrived at frame 283, `deaths=0`,
`progression/capacity writes=0`, byte-identical on both trials
(`recordings/6b_right_v1.json` / `6b_right_v2.json`). `0x6C` census:
`0x38` (digdogger-family) + `0x55` (statue/fireball projectile) — this is
`DIGDOGGER_1` (source: RIGHT → `DIGDOGGER_1`, the skippable whistle-split
spur). Graph: `DIGDOGGER_1` promoted `ram_id=0x6C` `evidence=fixture-live`;
`GORIYA_HINT` RIGHT → `DIGDOGGER_1` `verification=fixture-live`.

**`0x6B` LEFT backtrack GREEN 2/2.** `--dir LEFT` → `0x6A` `(224,141)` east
mouth, mode 5, keese `0x1b` present, byte-identical
(`recordings/6b_left_v1/v2.json`). Graph: `GORIYA_HINT` LEFT → `KEESE`
`GateKind.OPEN` `verification=fixture-live`.

**`0x6B` UP is BLOCKED 2/2.** `--dir UP` → Link pins at `(128,93)` on the
`y=93` band, no room change, on both trials
(`recordings/6b_up_v1/v2.json`). A straight centre-x UP push does **not**
reach `OLD_MAN_NOSE` even with the six goriya cleared — the north exit (if
any) is not on the `y=93` centre band. `OLD_MAN_NOSE` stays hypothesis.

**Wired (fixture-live, `route_eligible=false`, NOT on the executable
chapter chain):**
- `level7/path.py`: `room_6b_east_step` + `Room6BEastController`
  (`spec_id="level7_room6b_east"`), `east_of_room6b_ram_id()`, `ROOM_6B*`
  geometry constants. Mirrors `Room6AEastController`.
- `level7/hops.py`: `make_room6b_east_controller` (+ back-filled
  `make_room69_east_controller` / `make_room6a_east_controller` for the
  earlier already-green legs). None are in `level7_red_candle_chapter_stages`
  yet — that chain is still `entry_first_door → hungry_goriya (fail-closed)
  → tip_stairs → red_candle`.
- `level7/graph.py`: `DIGDOGGER_1` `ram_id=0x6C`; `GORIYA_HINT` exits
  re-annotated (LEFT/RIGHT `fixture-live`, UP dead-belief note).
- `tests/test_level7_dungeon.py`: live-prefix test extended with
  `DIGDOGGER_1: 0x6C`.

## 2026-09-05 — EAST mainline extended: `0x6C`→`0x6D`, `0x6B`→`0x5B`, `0x69` west bomb

**Row-6 corridor is `0x69 – 0x6A – 0x6B – 0x6C – 0x6D` (W→E, OPEN doors).**
`0x6D` is the east **dead-end** (STALFOS_KEY). The candle mainline branches
**west of `0x6A`**: `0x69` has a **BOMB wall on its west side → `0x68`**.
(The earlier "LEFT + bomb-UP through `0x6A`" belief is dead — `0x6A` has NO
north exit; the `0x6A` top wall *is* the `y=93` band. `0x69` is the source
`GORIYA_BOMB_HUB`.)

Verified this sitting (all from `Level7InteriorReconFixture`, `deaths=0`,
`progression/capacity writes=0`, `route_eligible=false`):

| walk | dest `$EB` | entry | census / reward | evidence |
|------|-----------|-------|-----------------|----------|
| `0x6B` RIGHT | `0x6C` | `(16,141)` W mouth | digdogger `0x38` + statue `0x55` | **2/2** (`6b_right_v1/v2`) |
| `0x6C` RIGHT | `0x6D` | `(16,141)` W mouth | stalfos `0x2a`, reward `room_item_id 0x19` small_key | **2/2** (`6c_right_v1/v2`) |
| `0x6B` UP (x≈118, y=93) | `0x5B` | `(120,205)` S mouth | bubble `0x40` + statue `0x50` | **2/2** (`6b_north_dest_v1/v2`, frame 189) |
| `0x69` LEFT **bomb** (stand ~`(44,141)` face LEFT) | `0x68` | mid-transition `(188,141)` | interior unobserved; `cur_opened_doors` LEFT bit sets | **2/2** (`69_branch_v2/v3`) |

Dead-ends confirmed 2/2:
- `0x6D` (STALFOS_KEY): RIGHT/UP/DOWN blocked, only LEFT → `0x6C` (`6d_v1/v2`).
  *(Walking over the key in `0x6D` bumped recon `keys` 4→5 — natural pickup.)*
- `0x5B` (OLD_MAN_NOSE): N/E/W/S all walled at `y=141` from the south entry
  (`5b_v1`); the "secret in the tip of the nose" hint room, NOT the mainline.
- `0x69` NORTH: precise x-sweep `104..156` on the `y=93` band — all solid,
  no notch (`69_branch_v2/v3`).

Wired (fixture-live, `route_eligible=false`, NOT on the executable chain):
- `level7/path.py`: `Room6BNorthController` / `room_6b_north_step`
  (`level7_room6b_north`), `Room6CEastController` / `room_6c_east_step`
  (`level7_room6c_east`), `Level7BombWall` + `L7_ROOM69_WEST_BOMB`
  (`room=0x69 stand=(44,141) face=LEFT opens_to=0x68`),
  `north_of_room6b_ram_id()` / `east_of_room6c_ram_id()`.
- `level7/hops.py`: `make_room6b_north_controller`,
  `make_room6c_east_controller`, `make_room69_west_bomb_controller`
  (returns `dungeon.bomb_wall.BombWallController(wall=L7_ROOM69_WEST_BOMB,
  level=7)`).
- `level7/graph.py`: `OLD_MAN_NOSE ram_id=0x5B`, `STALFOS_KEY ram_id=0x6D`,
  `KEESE_TRAPS ram_id=0x68`, all `evidence=fixture-live`; `MOLDORMS`
  (== `GORIYA_BOMB_HUB`) gains LEFT-bomb→`KEESE_TRAPS` and DOWN→`ENTRY`
  exits; `DIGDOGGER_1` RIGHT→`STALFOS_KEY` and `GORIYA_HINT` UP→`OLD_MAN_NOSE`
  promoted `fixture-live`.
- `tests/test_level7_dungeon.py`: live-prefix test + `OLD_MAN_NOSE:0x5B`,
  `STALFOS_KEY:0x6D`, `KEESE_TRAPS:0x68`.

## 2026-09-05 (cont.) — branch chain `0x69`→`0x68`→`0x58`; two more recon fixtures

Continuing the branch. All from the recon-fixture chain, `deaths=0`,
`route_eligible=false`:

| walk | dest `$EB` | census / notes | evidence |
|------|-----------|----------------|----------|
| `0x69` west **bomb** → `0x68` | `0x68` KEESE_TRAPS | dark; **4 blade traps `0x49`** (corners) + 4 keese `0x1b`; `open_doorway_mask=13` | **2/2** (`69_branch_v2/v3`) |
| `0x68` **UP** (align x=120) → `0x58` | `0x58` DODONGOS_UPGRADE | dodongo-family `0x31`, `room_item_id 0x0f`, dark; Link spawns bottom `(120,205)` | **2/2** (`68_up_v1/v2`) |

New disclosed recon fixtures (both `development_only`/`fixture_only`/
`route_eligible:false`/`natural_entry:false`, provenance records the WALK
chain; only a `poke_bombs(8)` count top-up before the `0x69` bomb, no
`max_bombs`/Candle/TF/door/health/capacity writes):
- **`Level7Interior68ReconFixture`** — settled 0x68 `~(208,93)`, keys 4,
  bombs 7, Food 1.
- **`Level7Interior58ReconFixture`** — settled 0x58 `(120,205)`, keys 4,
  bombs 7, Food 1.

**`0x58` layout (blind y-band sweep, `58_map_v1`):** dark, dodongo `0x31`.
North of `y≈88` is a **narrow x=120 corridor** (the door channel toward
`0x48`). `y=93` clear x=32..179; `y=109..189` open on the **east half**
(x≈92/116 → 208), west half walled. **No RIGHT transition on any band** —
Link reaches x=208 everywhere but does not cross. `cur_opened_doors`
flipped to `8` (RIGHT) and `keys` 4→3 during the sweep → there is a
**locked/kill-gated EAST door** (mainline → GORIYA_COMPASS) that needs the
dodongo `0x31` killed and/or a key. `0x58` also reaches **`0x48`** (bubble
`0x40` + `0x4f`, key-gated — the optional BOMB_UPGRADE `0x48`, NOT
mainline; the earlier probes' "LEFT→0x48" was really the north x=120
channel + knockback).

Wired: `level7/path.py` `Room68NorthController` / `room_68_north_step`
(`level7_room68_north`), `north_of_room68_ram_id()`, `ROOM_68*` consts;
`level7/hops.py` `make_room68_north_controller`; `level7/graph.py`
`DODONGOS_UPGRADE ram_id=0x58` + `KEESE_TRAPS` UP→`DODONGOS_UPGRADE`
`fixture-live`; test live-prefix += `DODONGOS_UPGRADE:0x58`.

## 2026-09-05 (cont. 2) — `0x58`→`0x59` (GORIYA_COMPASS)

**`0x58` EAST → live `$EB=0x59` (GORIYA_COMPASS)** — **2/2 byte-identical**
(`58_east_v2/v3`). `0x59` `(16,141)` W mouth, mode 5, census **goriya
`0x05` + `0x06`**. **OPEN door, keys unchanged** (the earlier "gated"
belief was wrong — the 58-map key-drop was a *north* door to `0x48`).

Key `0x58` facts:
- The 3× `0x31` (hp 240) are **invulnerable roamers** — sword + 7 bombs did
  nothing (`58_clear_v1`). Do **not** try to clear the room; dodge them.
- `0x58` has a **central structure** walling `y=141` west of `x~129`. The
  east door route climbs the east-open column: `(120,165) → (200,165) →
  (200,141) → push RIGHT`. `room_item_id 0x0f` stays uncollected (behind
  the block / not on the critical path).
- New fixture **`Level7Interior59ReconFixture`** (settled `0x59` `(16,141)`,
  keys 4 / bombs 7 / Food 1; disclosed: bombs count top-up only).

Wired: `level7/path.py` `Room58EastController` (`level7_room58_east`),
`east_of_room58_ram_id()`, `ROOM_58*` consts; `level7/hops.py`
`make_room58_east_controller`; `level7/graph.py` `GORIYA_COMPASS ram_id=0x59`
+ `DODONGOS_UPGRADE` RIGHT→`GORIYA_COMPASS` and DOWN→`KEESE_TRAPS`
`fixture-live`; test live-prefix += `GORIYA_COMPASS:0x59`.

## 2026-09-05 (cont. 3) — `0x59`→`0x49` (GORIYA_BUBBLE); **LADDER blocker**

**`0x59` UP → live `$EB=0x49` (GORIYA_BUBBLE)** — **2/2 byte-identical**
(`59_up_v2/v3.json`, arrived frame 2329; `59_ctl_v3/v4` = the wired
`Room59UpController` 2/2, arrived frame 1618). From
`Level7Interior59ReconFixture`: kill-clear the goriya `0x05`/`0x06` (sets
`cur_opened_doors` bit 3 = **UP**, `open_doorway_mask`→10; boxes Link at
`(48,125)`). The route around the central mass (fills ~`x100..190` /
`y118..165`) is a **perimeter waypoint micro**: rise the west side to the
`y~100` open band → west to `x~44` → rise to the `y~93` top band → cross
east to `x=120` → push UP (pre-push `(118,93)`). `0x49` `(120,205)` S
mouth, mode 5, census **goriya `0x05` + keese `0x1b` + bubble residual
`0x2b`** (+ transient boomerang `0x5c`). keys 4 / bombs 7 / Food 1
unchanged, `deaths=0`, `progression/capacity writes=0`.

New disclosed recon fixture **`Level7Interior49ReconFixture`** — settled
`0x49` `(120,205)`, keys 4 / bombs 7 / Food 1. **No fixture writes at all**
(counts already fine); `development_only`/`fixture_only`/
`route_eligible:false`/`natural_entry:false`.

**`0x49` (GORIYA_BUBBLE) is fully mapped — and the mainline is BLOCKED here
without the Stepladder.**
- Exactly two doors: **DOWN → `0x59`** (back, 1/1) and **UP → DIGDOGGER_2**
  (source). LEFT and RIGHT are **walled 2/2** (Link pins at `x=32` / `x=208`
  on `y=141`).
- The UP door **bit opens** on the goriya kill-clear, but a **full-width
  horizontal water moat** (~`y120`, colliding tile **`0xF4`**) walls the
  entire room — fine x-sweep `x=16..224` step 4, **every column blocked at
  `y=133`**, no land bridge anywhere.
- **Diagnostic (one-off, no fixture saved):** poking `ADDR_LADDER 0x0663`
  →1 lets Link walk straight north across the moat to the `x=120` door
  threshold (tile 118) and into the top half. **The moat is a Stepladder
  gate.**
- The recon-fixture chain descends from the `Level7Entrance` poke pin
  (Whistle poked, Food/TF/keys/bombs minimal) and **carries no Ladder**
  (`ADDR_LADDER`=0). The disclosed-write budget is Food/bombs/keys *count*
  top-ups only — a Ladder poke is a **capability write, out of scope**.

Wired (fixture-live, `route_eligible=false`, NOT on the executable chain):
- `level7/path.py`: `Room59UpController` / `room` consts `ROOM_59*`,
  `north_of_room59_ram_id()`. Phases clear→rise1→west→rise2→cross→push,
  mirrors `Room58EastController`.
- `level7/hops.py`: `make_room59_up_controller`.
- `level7/graph.py`: `GORIYA_BUBBLE ram_id=0x49 evidence=fixture-live`;
  `GORIYA_COMPASS` UP→`GORIYA_BUBBLE` `KILL_CLEAR` `verification=fixture-live`;
  `GORIYA_BUBBLE` DOWN→`GORIYA_COMPASS` `fixture-live`, UP→`DIGDOGGER_2`
  annotated with the moat/Ladder note (dest still hypothesis).
- `tests/test_level7_dungeon.py`: live-prefix += `GORIYA_BUBBLE:0x49`.

## Onward toward Red Candle — resume point

**BLOCKED at `0x49` (GORIYA_BUBBLE) pending a Ladder decision.** The
candle mainline (`GORIYA_BUBBLE → DIGDOGGER_2 → GORIYA_PRE_HUNGRY →
HUNGRY_GORIYA → MAP → … → CANDLE_PUSH → RED_CANDLE_CELLAR`) is gated by
the `0x49` water moat, which needs the **Stepladder** (an L4 item Link
carries in a real run but the recon-fixture chain does not). Options for
the orchestrator:
1. Allow a **disclosed `ADDR_LADDER`→1 poke** in a `Level7Interior49*` (or
   earlier) recon fixture — analogous to the already-disclosed Whistle/Food
   pokes on this chain — then continue the room-by-room recon from `0x49`
   UP.
2. Rebuild the recon chain from a fuller predecessor that already owns the
   Ladder (a real post-L4/L6 loadout).
3. Accept the L7-B recon stops at `0x49` until a natural-entry tape exists.

Also add **Stepladder** to `LEVEL7_ROUTE.md` required capabilities (it was
omitted from the L7 gate list).

If continuing from `0x49` (with a Ladder): push UP through the moat
(`x=120`, door plane `y≈93`), then recon `DIGDOGGER_2` and onward. All
rooms past `0x49` are **hypothesis**. `0x49` fixture: keys 4 / bombs 7 /
Food 1. Do **not** poke `ADDR_CANDLE`.

Fixture chain to regenerate `.state` files (all gitignored): parent
`Level7InteriorReconFixture` (`build_level7_interior_recon_fixture.py`) →
`Level7Interior68ReconFixture` (`probe_l7_room68_onward.py --save-fixture`)
→ `Level7Interior58ReconFixture` (`probe_l7_room58_onward.py --save-fixture`)
→ `Level7Interior59ReconFixture` (`probe_l7_room58_east.py --save-fixture`)
→ `Level7Interior49ReconFixture` (`probe_l7_room59_up.py --save-fixture`).
Optional dest pins: `Level7Interior78ReconFixture` (`probe_l7_room68_down.py
--save-fixture`), `Level7Interior48ReconFixture` (`probe_l7_room58_north.py
--save-fixture`).

## 2026-09-03 — L7-B side rooms: `0x68` DOWN + `0x58` north

Did not walk `0x49` UP. Did not poke `ADDR_LADDER` / `ADDR_CANDLE` /
`ADDR_MAX_BOMBS`. Candle-mainline narrative above is unchanged.

| walk | dest `$EB` | entry | census / reward | evidence |
|------|-----------|-------|-----------------|----------|
| `0x68` DOWN | `0x78` ROPES_KEY | `(120,77)` N mouth | ropes `0x28` + floor `small_key 0x19`; keys 4 (not picked); dead-end | **2/2** (`68_down_v3/v4`, `68_ctl_v1/v2` frame 297/298) |
| `0x58` UP KEY | `0x48` BOMB_UPGRADE | `(120,205)` S mouth | bubble `0x40` + `0x4f`; old-man "I BET YOU'D LIKE TO HAVE -100"; keys 4→3; `max_bombs` stays 8 | **2/2** (`58_north_v2/v3`, `58_ctl_v1/v2` frame 337/338) |

`0x68` DOWN: blade traps `0x49` in the four corners. OccupancyWalker
poisoned the grid on knockback (v2 stood `(174,149)`, 12 misses). Waypoint
micro: peel west to `x=160` (off the east trap column), drop `y=141`
(between trap rows `y~93`/`y~189`), align `x=120`, push DOWN. If knocked
onto the `y~189` trap row off-x, rise first.

`0x58` north: 3× invuln `0x31` (hp 240) — dodge, do not kill. A central
2-block mass walls the `x=120` column around `y=141`. OccupancyWalker to
`(120,93)` boxed at `(122,165)` (25 misses). East-around: climb `y=165`,
RIGHT `x=160`, UP `y=93`, align `x=120`, push the KEY door (natural key
spend). `0x48` is a **dead-end** 100-rupee bomb-capacity old-man room;
do **not** write `max_bombs` (`capacity_writes=0`).

Wired (fixture-live, `route_eligible=false`, NOT on the executable chain):
- `level7/path.py`: `Room68DownController` / `room_68_down_step`
  (`level7_room68_down`), `Room58NorthController` / `room_58_north_step`
  (`level7_room58_north`), `south_of_room68_ram_id()` /
  `north_of_room58_ram_id()`.
- `level7/hops.py`: `make_room68_down_controller`,
  `make_room58_north_controller`.
- `level7/graph.py`: `ROPES_KEY ram_id=0x78`, `BOMB_UPGRADE ram_id=0x48`,
  both `evidence=fixture-live`; `KEESE_TRAPS` DOWN and `DODONGOS_UPGRADE`
  UP KEY `verification=fixture-live`.
- `tests/test_level7_dungeon.py`: live-prefix += `ROPES_KEY:0x78`,
  `BOMB_UPGRADE:0x48`.

New dest fixtures (disclosed writes: none; `development_only` /
`fixture_only` / `route_eligible:false` / `natural_entry:false`):
- `Level7Interior78ReconFixture` — settled `0x78` `(120,77)`, keys 4 /
  bombs 7 / Food 1 / `max_bombs` 8.
- `Level7Interior48ReconFixture` — settled `0x48` `(120,205)`, keys 3 /
  bombs 7 / Food 1 / `max_bombs` 8.

## Dead beliefs

1. Dead: `0x6B` UP at centre-x (x=128) reaches `OLD_MAN_NOSE` — the notch
   is at **x≈118**; centre-x is solid (blocked 2/2 at `(128,93)`).
2. Dead: `0x6C` (`DIGDOGGER_1`) is a "skippable spur" off the mainline —
   the room **is** on the mainline (only the digdogger *fight* is
   whistle-skippable); it leads to the STALFOS_KEY dead-end.
3. Dead: the candle mainline is "LEFT + bomb-UP through `0x6A`". `0x6A` has
   no north exit. The branch is the **`0x69` west bomb wall → `0x68`**.
4. Dead: `0x6A` north has a bombable wall (9 bomb attempts x=64..192 on the
   `y=93` band, which is the north wall — nothing opened; those attempts
   also mis-positioned Link, but the `0x69` west-bomb branch is now the
   confirmed route, so `0x6A`-north is retired).
5. Dead: OccupancyWalker to the `0x68` south / `0x58` north door — blade
   traps and `0x31` knockback poison the grid (stood `(174,149)` / boxed
   `(122,165)`). Waypoint micros. Also dead: `0x58` north is a straight
   `x=120` walk from the south mouth — a central 2-block mass walls that
   column around `y=141`; east-around first.

## Prior sittings (still standing)

- `0x6A` east GREEN 2/2 (`Room6AEastController`, `recordings/
  l7_room6a_east_room6a_east_v2/v3.json`): rise west column `y=93`, cross,
  drop east column `x=200`→`y=141`, push OPEN doorway plane `x=224`. Room
  unlit (Candle 0); keese `0x1b` never block. Dead: `0x6A` `y=141` centre
  band (walled `x=48` tile `0xB1`); dead: `0x6A` east is KEY/KILL_CLEAR
  (it is OPEN).
- `0x69` east GREEN 2/2 (`east_route_step`: rise `y=109`, RIGHT `x=204`,
  DOWN `y=141`, push `x=208`). Dead: `cur_opened_doors & RIGHT` as a
  walkability test — it never sets on an OPEN L7 doorway; per-pixel
  occupancy boxes Link in after four graded misses (use waypoints).

---

## L7 chapter handoff (integrator reference)

`route_eligible=false` on every fixture. Shared spine still fail-closed at
`level7_pond_drain_entry`. Public `--through` targets unchanged:
`level7-entry`, `level7-red-candle`, `level7`.

---

## L7-A — topology + entry preparation

- **chapter id:** `rr-8t4.1` / `level7-entry`
- **evidence label:** mixed. Offline graph + Bait/post-L6 interface =
  **hypothesis**. Start-based `0x53→0x52` inland-left micro =
  **fixture-live**. Post-L6 bait prefix `0x22→0x32→0x33→0x23→0x24→0x25` =
  **fixture-live**. Pond `0x42` overworld screen **reached 2/2 geometry-only**
  (`OverworldToLevel7PondController` from `PostSwordStart`). Recon drain
  with an `ADDR_WHISTLE` poke enters L7 play **`0x79` `(120,205)`**
  (`Level7Entrance` pin). Natural drain from the L6 leave is still
  unobserved (`rr-8t4.4`). Stop at fixture-live; integrator owns
  natural-segment / spine-green.

- **exact predecessor (updated 2026-09-02, Phase 1):** the L6 fanfare leave is
  **measured and verified** — OW `0x22` `(112,125)` TF `0x3F` keys 2 bombs 8
  rupees 42, `selected_item=2` (arrows), Whistle 1 Food 0 Candle 0, 8 HC full
  (`--through level6-exit` 2/2, `l6_exit_ow.json` + `l7p1_l6exit.json`,
  byte-identical). Carried as `MEASURED_POST_L6_EXIT`, a shared
  `zelda_i.overworld.stitch.OverworldHandoff`, **`verified=True`**
  (`route_eligible=false`). The deleted `(120,221)` / 80R / White-Sword poke
  loadout was never a fanfare. The live L6 interior residual play `0x09`
  `(56,109)` Rod=0 is not an L7 start.

- **required inventory/capabilities:** TF `0x3F`, Whistle ≥1, Rod ≥1, Bow ≥1,
  sword. Food 0 until natural 60R Bait at shop hyp `0x34`. Candle remains 1
  until Red Candle. Geometry-only pond mapping may run without Whistle.

- **ordered internal stage names and controller factories:**
  1. `level7_post_l6_overworld` — `make_post_l6_overworld_controller(handoff, hops)`
     (`OverworldHandoff` gate + default `hops=POST_L6_TO_BAIT_HOPS`). Handoff
     `verified=True` since Phase 1, so it **walks `0x22→0x25` green**
     (`path_complete`, phase DONE, 1577f, `writes=0`) on a continuous power-on.
     The `_left_mouth` latch keeps the `0x22` mouth-tile spawn from tripping
     the re-entry refusal on frame 1.
  2. `level7_bait_purchase` — **Survival:** `make_survival_bait_purchase_controller`
     (`l7_hops(survival=True)`) — one disclosed `ADDR_FOOD` write (`$065D`→1)
     in place of the natural 60R buy, since the natural L6→shop OW route is a
     mountain-locked pocket (bead `rr-8t4.4`); `SPINE_L7_RUPEE_RETOPUP` still
     tops the owned rupee **count** 42→60 for the cost paid; passes green.
     **Clean:** `make_bait_purchase_controller` still fails closed at
     `bait_shop_geometry_unobserved` (shop `0x34` geometry / cave xy / buy
     policy all still unobserved). Disclosed in `docs/ASSIST_CONTRACT.md`.
  3. `level7_pond_drain_entry` — `make_pond_entry_controller()`
     (still fail-closed on the spine). Recon drain from `OW_L7Pond` with a
     disclosed `ADDR_WHISTLE` poke is **green** (`drain_v2`, L7 play `0x79`
     `(120,205)`). A natural-whistle controller from the L6 leave is `rr-8t4.4`.

  Recon-only (not a spine stage): `OverworldToBaitShopController` from
  `Level6ExitOverworld`; `OverworldToLevel7PondController` from
  `PostSwordStart` (no Whistle).

- **exact endpoint predicate:** `level7_entry_stop` — live L7 play in
  **`0x79`**, TF `0x3F`, Whistle and Food owned, `route_eligible` and
  `evidence` in `{natural-segment, spine-green}`. Room id is live
  (`fixture-live`) but evidence is not in that set, so the predicate
  still fails closed.

- **expected inventory deltas:** Food 0→1. No TF change. Keys/bombs unchanged
  on OW. Candle stays 0 (Blue Candle not on the mainline; Red Candle is the L7
  dungeon item). Disclosed **Survival** writes at `level7_bait_purchase`:
  rupee **count** 42→60 (`SPINE_L7_RUPEE_RETOPUP`) + one `ADDR_FOOD`→1
  (`SurvivalBaitPurchaseController`, bead `rr-8t4.4`). No rupee-grant /
  Whistle / door / TF writes. Clean does the Food gain via a natural buy
  (still fail-closed until `rr-8t4.4`).

- **known dead beliefs / first missed RAM claim:**
  - Dead: `hop10_ay` DOWN from `0x53` `(224,173)` (v9). Inland-left first.
  - Dead: start-`0x77` pond walk as the spine post-L6 path.
  - Dead: OW `0x22` as proven L6 leave; live L6 prefix `0x09` as L7 start.
  - Dead: `(112,125)` on `0x22` is a dead spot — it *is* the measured leave
    (the mouth tile). Re-entry only fires on a fresh UP into it / mode 16;
    the `_left_mouth` latch handles the spawn-on-mouth case.
  - Dead: `0x32` `(120,61)` `off_north` DOWN (`l7_bait_from_l6`). Corridor is x=112.
  - Dead: `0x33` RIGHT @ y=141 → `0x34` (`l7_bait_32ax` leftover `(208,141)`).
  - Dead: `0x24` DOWN at `(16,189)` (`l7_bait_33up`), `(160,189)`
    (`l7_bait_24belt`, north-ladder x), `(208,189)` (`l7_bait_24se` SE).
  - Dead: occupancy xmin=14 west pocket `(0,141)`; SW occupancy box `(25,181)`.
  - Dead: `0x24↓0x34` (south wall sealed at x=16 / 160 / 208).
  - Current leftover (Phase 1, continuous power-on): play `0x25` `(0,141)`
    west mouth, `level7_post_l6_overworld` **green** through it
    (`l7p1_entry_v2.json`).
  - **2026-09-02 recon (`scratch/{sweep_25_armos,probe_24_to_shop,
    probe_33_south_to_shop}.py`, `--from-state Level6ExitOverworld`):**
    `0x33`/`0x23`/`0x24`/`0x25` are a mountain-bounded desert pocket with **no
    south exit** — `0x25` south walled (only hidden exit north x≈208→`0x15`,
    wrong way); `0x24` south walled at every x, 10-Armos sweep reveals no shop
    stair (top-right = Bracelet); `0x33` south walled at x∈{120,160,208},
    `0x33→0x34` RIGHT still walled. External grid confirms L6=C3=`0x22`,
    Bracelet=E3=`0x24`, Bait shop=E4=`0x34`. Source `U L×3 U×3` enters `0x34`
    walking **north from `0x44`**; the shop stair is the top-row-middle Armos
    *on `0x34`*.
  - `0x32` south edge also solid (x∈{56,80,96,192}); `0x32` exits are only
    NORTH→`0x22` and EAST→`0x33`. **The whole `0x22/0x32/0x33/0x23/0x24/0x25`
    region is a mountain-locked pocket** — no southward route to the row-4/5
    band with pond `0x42` / shop `0x34`.
  - **Retire** the `0x25→0x35→0x34` plan and the `0x23/0x24/0x25` detour
    entirely. The shop `0x34` and pond `0x42` are both entered walking north
    out of the **western forest band** (`LEVEL7_POND_APPROACH_HOPS`
    territory, reached from *start*): `…0x64→0x54→0x44↑0x34`,
    `…0x54→0x53→0x52→0x42↑`. Route-owner decision: buy Bait before L6, or
    loop the post-L6 pocket exit north/west around the mountains, or take a
    longer post-L6 leg. The spine's `0x22→…→0x25` "green" walk is a dead spur.
    **Tracked as bead `rr-8t4.4`** (natural L6→shop OW route). Until it lands,
    the Survival spine sets Food directly (`SurvivalBaitPurchaseController`,
    disclosed in `docs/ASSIST_CONTRACT.md`).

- **L7 start pin — captured (recon).** `Level7Entrance.state`: L7 play
  **`0x79` `(120,205)`** south mouth, whistle=1 (poked), food=0, TF=0,
  3 HC. Drain recipe: `OW_L7Pond` → poke `$065C` + B-slot 5 → 12×B →
  idle ~240f → walk dry bed; stairs `(96,132)` tile 114. Harness:
  `scratch/probe_l7_pond_drain.py`. `LEVEL7_ENTRY_STOP.screen=0x79`
  `evidence=fixture-live` `route_eligible=false` — spine still fail-closed.
  Not natural-entry (PostSwordStart + whistle poke, no L5/L6 TF). Natural
  whistle-carrying pond leftover is still blocked by the `0x22` mountain
  pocket (`rr-8t4.4`) and the missing post-L5 OW checkpoint.

- **Why the pond leftover had whistle=0 (not a failed L5 pickup).** The
  geometry walk is `PostSwordStart` → `LEVEL7_POND_HOPS` (`0x77→…→0x42`).
  It never enters L5, so `$065C` stays 0 and TF stays 0. On the Survival
  spine the Recorder **is** earned: `attach_level5_whistle_suffix` /
  `--through level5-whistle` 1/1, and `--through level6-exit`
  (`l7p1_l6exit.json` `final.whistle=1`, `MEASURED_POST_L6_EXIT`). There is
  no `Level5ExitOverworld`; whistle-owning pins are L5-interior
  (`Level5WhistleFrom77` cellar `0x04`). Post-L5 OW settle is Lost Hills
  `0x0B`, then L6. Post-L6 fanfare dumps Link in the `0x22` mountain
  pocket with Whistle 1 but **no south path** to pond `0x42`. So: pickup
  works; the pond walk that exists does not go through L5; the tape that
  owns Whistle cannot walk to the pond. Natural leftover = leave L5 to
  OW and take the west-forest hops to `0x42` **before** L6.

- **fixture provenance:** Phase 1 is a **continuous power-on** tape, not a
  save-state fixture. `--through level7-entry` (`recordings/l7p1_foodfix.json`)
  drives power-on → measured L6 fanfare exit → `level7_post_l6_overworld`
  green `0x22→0x25` → `level7_bait_purchase` green (Survival Food fixture) →
  fail-closed at `level7_pond_drain_entry`. `set_state=0`, `deaths=0`. Pond
  recon remains `recordings/l7_dnp_pond_53.json` leftover play `0x52` `(112,181)`.
  Entry pin: `Level7Entrance` / `OW_L7Pond` (`l7_pond_drain_drain_v2.json`).

- **files changed (Phase 1) / public target:** `level7/entry.py` (verified
  handoff + `_left_mouth` latch + `SurvivalBaitPurchaseController`),
  `level7/hops.py` (`survival=` swap), `level7/spine.py`
  (`SPINE_L7_RUPEE_RETOPUP`, `l7_hops(survival=True)`),
  `spine/survival.py` (`spine_final_fields(ram)`, `topup_owned_rupees`,
  `rupee_retopup`), `dungeon/ops.py` (`poke_rupees`, `poke_food`,
  `apply_owned_inventory(rupees=)`), `assist.py` (`poke_food` re-export),
  `scripts/run_survival_spine.py`, `tests/test_level7_hops.py`,
  `tests/test_dungeon_ops.py`, `docs/{LEVEL7_ROUTE,ASSIST_CONTRACT}.md`,
  `docs/tasks/{l7-handoff,ow-handoff,rr-tne2-residual}.md`.
  Public target: **`level7-entry`** (green through `level7_bait_purchase` on
  the Survival spine).

---

## L7-B — entry through Red Candle

- **chapter id:** `rr-8t4.2` / `level7-red-candle`
- **evidence label:** **fixture-live** row-6 corridor
  `0x79→0x69→0x6A→0x6B→0x6C→0x6D` + `0x6B`→`0x5B` spur + branch
  `0x69`─bomb→`0x68`─UP→`0x58`─EAST→`0x59` + side rooms `0x68` DOWN
  `0x78` and `0x58` KEY-UP `0x48`; rooms past `0x59` (candle chain)
  still hypothesis
- **predecessor:** `Level7Entrance` pin — L7 play `0x79` `(120,205)`. Inventory
  is the poke loadout (Whistle 1, Food 0, TF 0), **not** the L6-leave packet.
  Hungry Goriya still needs Food; isolate with the recon fixture or `rr-8t4.4`.
- **required:** Food ≥1 (Hungry Goriya), Whistle, bombs for wall skips, keys
  for three locks (skip fifth via bombs)
- **recon fixture:** `Level7InteriorReconFixture` (`scratch/
  build_level7_interior_recon_fixture.py`) — WALKS `0x79→0x6B` from
  `Level7Entrance`, clears the six `0x6B` goriya, then discloses `ADDR_FOOD`
  0→1, `ADDR_BOMBS`→8, `ADDR_KEYS`→4. Settled L7 play `0x6B` `(136,109)`
  TF 0 Candle 0 Whistle 1. `development_only`/`fixture_only`/
  `route_eligible:false`/`natural_entry:false`; no Candle/Whistle/TF/door/
  room/health/capacity writes. NOT on any spine.
- **stages / factories:**
  1. `level7_entry_first_door` — `make_entry_first_door_controller()`
     (live north OPEN → `$EB=0x69`; `route_eligible=false`).
  1a. `level7_room69_east` — `Room69EastController` /
     `make_room69_east_controller()` clears the `0x69` goriyas, walks the
     **OPEN** east doorway to live `$EB=0x6A` `(16,141)` (2/2, 3,809f,
     `writes=0`). Not a spine stage.
  1b. `level7_room6a_east` — `Room6AEastController` /
     `make_room6a_east_controller()` walks the unlit `0x6A` KEESE room west
     mouth → **OPEN** east doorway to live `$EB=0x6B` `(16,141)` (2/2, 482f,
     `writes=0`, `deaths=0`). Waypoint: rise `y=93`, cross, drop
     `x=200`→`y=141`, push `x=224`. Not a spine stage.
  1c. `level7_room6b_east` — `Room6BEastController` /
     `make_room6b_east_controller()` (2/2) walks `0x6B` west mouth → OPEN
     east doorway to live `$EB=0x6C` (`DIGDOGGER_1`, `(16,141)`, 283f).
     Ride `y=109` east past the central X, drop `x=200`→`y=141`, push
     `x=224`. **Assumes `0x6B` goriya cleared upstream.**
  1d. `level7_room6c_east` — `Room6CEastController` /
     `make_room6c_east_controller()` **(new 2026-09-05, 2/2)** walks `0x6C`
     west mouth → east door to live `$EB=0x6D` (`STALFOS_KEY`: stalfos
     `0x2a` + small_key `0x19`, a **dead-end**). Ride `y=141`; a digdogger
     bump nudges Link through.
  1e. `level7_room6b_north` — `Room6BNorthController` /
     `make_room6b_north_controller()` **(new 2026-09-05, 2/2)** walks `0x6B`
     west mouth → OPEN north notch at **x≈118** → live `$EB=0x5B`
     (`OLD_MAN_NOSE`: bubble `0x40` + `0x50`, a **dead-end** hint spur).
  1f. `level7_room69_west_bomb` — `make_room69_west_bomb_controller()`
     **(new 2026-09-05, 2/2)** = `dungeon.bomb_wall.BombWallController(wall=
     L7_ROOM69_WEST_BOMB, level=7)`: `0x69` west BOMB wall (stand
     `(44,141)` face LEFT) → live `$EB=0x68` (`KEESE_TRAPS`: dark, 4 blade
     traps `0x49` + 4 keese). **This is the candle-path branch.** Needs
     bombs + bomb on B. Not a spine stage.
  1g. `level7_room68_north` — `Room68NorthController` /
     `make_room68_north_controller()` **(new 2026-09-05, 2/2)** walks `0x68`
     (align x=120 on top band) → OPEN north door → live `$EB=0x58`
     (`DODONGOS_UPGRADE`: 3× invuln `0x31`, `room_item_id 0x0f`). Not a
     spine stage.
  1h. `level7_room58_east` — `Room58EastController` /
     `make_room58_east_controller()` **(new 2026-09-05, 2/2)** climbs the
     `0x58` east-open column `(120,165)→(200,165)→(200,141)` → OPEN east
     door → live `$EB=0x59` (`GORIYA_COMPASS`: goriya `0x05`/`0x06`).
     Dodges the 3× invuln `0x31`. Not a spine stage. Recon resumes at
     `0x59` (its UP door → GORIYA_BUBBLE, mainline).
  1i. `level7_room68_down` — `Room68DownController` /
     `make_room68_down_controller()` **(new 2026-09-03, 2/2)** walks `0x68`
     OPEN south door → live `$EB=0x78` (`ROPES_KEY`: ropes `0x28` + floor
     `small_key 0x19`, a **dead-end**). Peel `x=160`, drop `y=141`, push
     DOWN. Not a spine stage.
  1j. `level7_room58_north` — `Room58NorthController` /
     `make_room58_north_controller()` **(new 2026-09-03, 2/2)** east-arounds
     the `0x58` central mass, KEY north door (keys 4→3) → live `$EB=0x48`
     (`BOMB_UPGRADE`: bubble `0x40` + `0x4f`, 100-rupee old-man
     bomb-capacity **dead-end**). Do not write `max_bombs`. Not a spine stage.
  2. `level7_entry_to_hungry_goriya` — `make_entry_to_goriya_controller()`
     (fails `hungry_goriya_requires_food` if Food=0; else room unobserved)
  3. `level7_tip_of_nose_stairs` — `make_tip_stairs_controller()` (blocker + ledger notes)
  4. `level7_red_candle_pickup` — `make_red_candle_controller()`
     (`ADDR_CANDLE` 1→2 natural; room unobserved)
  Executable chapter chain (`level7_red_candle_chapter_stages`) is unchanged:
  `entry_first_door → entry_to_hungry_goriya (fail-closed) → tip_stairs →
  red_candle_pickup`. Stages 1a–1j are recon-wired only.
- **endpoint:** `level7_red_candle_stop` — Candle==2, TF `0x3F`, Whistle
  retained, Food==0, exact live room. Room id `None` → fail closed.
- **resume point:** from `Level7InteriorReconFixture`, `0x6B` LEFT → `0x6A`
  LEFT → `0x69` (west-traverse = Room6AEastController mirror), then
  `make_room69_west_bomb_controller()` into **`0x68`**, then recon `0x68`
  (KEESE_TRAPS) and the source candle chain past it. See the
  "2026-09-05 — EAST mainline extended" section above.
- **expected deltas:** Food 1→0 at Hungry Goriya; Candle 1→2 at cellar.
  Ledger hyp net (dungeon): keys +1+1−1−1+1−1, bombs several −1 wall skips.
  Prefer bomb north of Map over locked east (fifth lock).
- **dead beliefs:** fifth lock required; source RAM room ids as stop specs;
  N path is Moldorms (live dest `0x69` is goriya `0x05`); `0x69`/`0x6A`/`0x6B`
  east are key/kill-clear doors (all OPEN — `cur_opened_doors` stays 0 on
  every live L7 doorway, but a **bombed** door DOES set the bit — `0x69`
  west); per-pixel occupancy boxes Link in — use waypoints; `0x6B` UP at
  centre-x (x=128) reaches `OLD_MAN_NOSE` (notch is x≈118); `0x6C`/`0x6D`
  are on the mainline as far as the STALFOS_KEY dead-end (`0x6C` fight is
  whistle-skippable, room is not); candle mainline is "bomb-UP through
  `0x6A`" (`0x6A` has no north exit — the branch is `0x69` **west** bomb →
  `0x68`).
- **fixture:** `Level7InteriorReconFixture` (recon only). `route_eligible=false`.
- **public target:** **`level7-red-candle`**.

---

## L7-C — Red Candle through shard leave

- **chapter id:** `rr-8t4` clear / `level7`
- **evidence label:** **hypothesis**
- **predecessor:** L7-B (Candle 2, Food 0, TF `0x3F`, Whistle owned)
- **required:** Whistle (forced Digdogger), Candle 2, incoming heart
  container count recorded
- **stages / factories:**
  1. `level7_forced_digdogger` — `make_forced_digdogger_controller()`
  2. `level7_aquamentus_heart` — `make_aquamentus_heart_controller()`
  3. `level7_shard_and_settled_leave` — `make_level7_shard_leave_controller()`
- **endpoint:** `level7_complete_stop` — TF `0x7F`, Candle 2, Whistle ≥1,
  hearts +1 and full, measured settled OW leave. Leave screen `None` → fail closed.
- **expected deltas:** TF `0x3F→0x7F` (`0x40`), heart containers +1, full
  hearts, deaths 0. Post-fanfare OW leftover **UNMEASURED**.
- **dead beliefs:** none live. Boss type Aquamentus hypothesized (verify).
- **fixture:** none. `route_eligible=false`.
- **public target:** **`level7`**.

---

## Stitch notes for L8

Expected L7 leave (still unmeasured, do not invent):

- TF `0x7F` (bits through shard 7)
- Candle 2 (Red)
- heart containers +1 vs L7 incoming, full `lo==hi`
- Whistle retained; Food 0 after Hungry Goriya
- post-fanfare OW leftover still **UNMEASURED** (screen, x/y, keys, bombs,
  rupees, selected). L8 must keep `PostLevel7Handoff.verified=false` until
  this packet is measured.

Seam for integrator: `L7_THROUGH`, `L7_STOPS`, `continue_level7_spine`.
`spine/survival.py` is yours. Wave A did not attach it.

Did not STATUS-promote.
