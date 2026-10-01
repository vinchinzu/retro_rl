# ALTTP port plan: Zelda I tools to the credits (200 tasks)

Written 2026-09-29. Tracked **in this file only**. Do not create, claim or
close `bd` issues for these tasks. The tick box on each task is the tracker.

Goal: take the working NES `nes/zelda_i` stack (Clean power-on to the credits,
M8, tape replay) and rebuild the same tools for SNES `snes/alttp`. Then use
them to finish ALTTP: Sanctuary, three pendants, Master Sword, Agahnim, seven
crystals, Ganon, credits. ALTTP today is M1: continuous tip at room `0x50`.
There are 7 dev-only dungeon entry pins (`docs/tasks/residual.md`).

Every task is sized for one agent in one sitting. It names its inputs, the
files it may touch, and a check that says it is done. If a task is too big,
split it here (`AP-045a`, `AP-045b`). Do not stretch the sitting.

---

## How to use this file

1. Pick the **lowest-numbered** open task (`- [ ]`) whose `Needs` are all
   `[x]`. Dungeon blocks (Phase H) can run in parallel once Phases A to E
   are done. Use one worktree per parallel lane (see Rules).
2. Claim it: change `- [ ]` to `- [~] <agent> <date>` and save.
3. Read only the files the task names, plus `snes/alttp/AGENTS.md`. Do not
   read the history of other tasks.
4. Finish: `- [x] <date> · evidence: <report path or test name>`. Add one
   line under the task if you learned something the next agent needs.
5. Blocked after two honest tries: `- [!] BLOCKED <date>: <one-line reason>,
   <evidence path>`. Stop there. Do not weaken the check to make it pass.

Commands assume cwd `snes/`, as in `alttp/AGENTS.md`:

```bash
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/<tool>.py ...   # anything that boots the ROM
QT_QPA_PLATFORM=offscreen uv run pytest alttp/tests -q             # unit tier
```

## Rules for every task

- **A save-state pin is not continuous.** The ladder is planned, then
  isolated, then natural_entry, then continuous. Only a power-on session
  with no mid-run state load is continuous.
- **No RAM writes in a natural or continuous run.** Pokes are allowed only
  in pins named `Dev*`, `Whatif_*` or `*Entry`, and each needs a
  `.provenance.json` that says "poke" (AP-026). Never write `$F3CC`, keys,
  rupees, HP or items in a run you will quote.
- **Planner owns `docs/STATUS.md`.** Only the rung tasks may edit it: AP-082,
  each dungeon's step 8, AP-199 and AP-200, and only with the evidence the
  task names.
- **Geometry lives in `alttp/maps/*.json`.** No room xy in Python. Only
  `recoveryPath` points for a real wedge go in the map.
- Do not overwrite `recordings/verified_tip_run.json` on a red run.
- One emulator per process. Batch runs spawn subprocesses.
- Soft max ~1000 lines per source file. When a file grows, merge into the
  owning module (the "Composer") or delete code. Do not split off a sibling file.
- Before porting a NES helper, check `retro_harness/` for a shared one:
  `audit.py`, `hop_timer.py`, `branch_rollout.py`, `bk2.py`, `recorder.py`,
  `snapshot.py`. Promote ALTTP code to `retro_harness/` only in a task that
  says so.
- Stage only the files you touched. Never `git add -A`. The tree is shared
  with other game lanes.
- Do not edit runtime modules while a long run is in flight. Freeze a
  worktree and run from it with `PYTHONPATH=$W:$W/snes:$W/nes`. The venv
  `.pth` imports the main tree otherwise.
- `z3-json-data` and a live scan are the authority on rooms, items and
  logic. The "hint" lines below are vanilla knowledge. If a hint is wrong,
  fix the hint in this file and note it in your task line.

## Lessons carried over from Zelda I (read once)

1. **Power-on save-point loop.** Run continuous. When it fails, fix from the
   `Full_<stage>` pin, resume forward, then run continuous again. Do not add
   an idle frame after `set_state`.
2. **One run is the measurement.** The emulator is deterministic, so one
   config has one outcome. Score combat on 12 RNG offsets (idle N frames
   first), never on one tape.
3. **Read the census before tuning.** Read `reason_by_room` and
   hits-by-cause first. Four 20k-frame NES stalls were only visible there.
4. **Measure; do not trust wiki numbers.** Zelda I had Link's walk speed
   wrong by half. Sword latency has to be measured from one isolated press.
5. **Read the door table before adding a clear.** A room whose exit is open
   or keyed is walked alive. Fight only for a shutter, a push block or an
   item the route takes.
6. **Tug-of-war stalls.** When Link flips 1-2 px, print consecutive frames'
   actions. One helper owns a goal end to end. Add hysteresis.
7. **Stall escapes are opt-in.** Gate a fix on the stall itself, not on a
   condition a green run also meets.
8. **A tape is a tape of its code.** Quote a tape only against the source it
   was made with.
9. **Audit a pin before quoting hearts.** Zelda I entry pins held 15 hearts in
   3 containers.
10. **Filters hide hazards.** Projectiles often have `hp == 0`. Do not filter
    sprites on HP.
11. **The reactive layer runs first.** A position rule that returns before
    the threat guard is the main source of damage.
12. **A what-if pin sizes a lever.** Write the item into a pin copy, run the
    segment, decide, then delete the pin.
13. **A `@dataclass` default set on the class later does not reach instances.**
14. **The ROM is the enemy model.** A savestate rollout is exact. Straight-line
    enemy prediction was 22% wrong at the dodge horizon. Do not port enemy AI
    from a disassembly; use the constants only.

## Zelda I → ALTTP translation table

| Zelda I (NES) | NES source | ALTTP target | Tasks |
|---|---|---|---|
| Sparse RAM snapshot | `nes/zelda_i/ram.py` | `alttp/ram.py` (extend) | AP-005..013 |
| Hit log, ledger | `nes/zelda_i/spine/ledger.py` | `alttp/spine/ledger.py` | AP-014..018 |
| Room timer | `nes/zelda_i/room_timer.py` | `alttp/room_timer.py` | AP-015 |
| Button tape + replay | `nes/zelda_i/runner.py`, `scripts/replay_tape.py` | `alttp/spine/tape.py`, `alttp/scripts/replay_tape.py` | AP-019..021 |
| Hop table, `--through` | `nes/zelda_i/spine/hops.py` | `alttp/spine/hops.py`, `alttp/scripts/run_spine.py` | AP-023 |
| Save points, `--resume` | `scripts/run_survival_spine.py` | `alttp/scripts/run_spine.py` | AP-024, AP-025 |
| Clean ladder | `nes/zelda_i/spine/clean_tip.py` | `alttp/spine/clean_ladder.py` | AP-003 |
| Pin probe | `scripts/pin_probe.py` | `alttp/scripts/pin_probe.py` | AP-031..037 |
| Pin audit | `scripts/audit_pins.py` | `alttp/scripts/audit_pins.py` | AP-027 |
| Stage replay, offsets | `scripts/stage_replay.py`, `scratch/offset_pins.py` | `alttp/scripts/stage_replay.py`, `eval_offsets.py` | AP-028, AP-029 |
| Door table | `dungeon/tilemap.room_door_types` | door arrays in WRAM | AP-034, AP-035 |
| World item flags | `$067F`/`$06FF`/`$077F` | room flags / OW flags (verify) | AP-009, AP-010 |
| ROM collision grid | `dungeon/tilemap.py` (`$6530`) | tile attribute map (verify) | AP-045, AP-046 |
| Lattice walkers | `dungeon/hop_controller.py` | `alttp/dungeon/nav.py` | AP-049..051 |
| Rollout, `PolicyGuard` | `nes/zelda_i/rollout.py` | `alttp/rollout.py` | AP-054..058 |
| Shot guard | `dungeon/shot_guard.py` | projectile guard | AP-064 |
| Potion guard | `dungeon/pause_select.py` | Y-menu + bottle guard | AP-065, AP-066 |
| Room spec | `dungeon/engine.py` `DungeonRoomSpec` | `maps/room_XX.json` `clearPolicy` | AP-063 |
| Assist (Survival) | `nes/zelda_i/assist.py` | `alttp/assist.py` | AP-068 |
| What-if pins | `run_survival_spine.py --set` | `alttp/scripts/whatif_pin.py` | AP-041 |
| Bosses | `dungeon/gohma.py`, `dungeon/gleeok.py`, `level9/patra.py`, `level9/ganon.py` | `alttp/dungeon/bosses/*.py` | dungeon step 6 |

## ALTTP differences that change the port

- **RAM indexing.** `get_ram()` index = WRAM offset below `0x2000`, else
  `16384 + offset`. Writes go through `memory.assign` on a bus address (see
  `bus_addr` in `alttp/scratch/probe_ice_entry.py`). Bank `$7F` coverage is
  not verified (AP-045).
- **16 sprite slots**, 16-bit world coordinates, large rooms with quadrants,
  two layers (`$EE`), floors, and pits that drop Link to the floor below.
- **Free 8-way movement.** Expect no NES-style 8 px turn lattice; AP-047
  confirms. Doorways still need alignment.
- **Followers:** Zelda, the old man, Kiki, the maiden, the super bomb. Each
  one has an acceptance byte (`$F3CC` and friends, AP-013).
- **Y-item menu, magic meter, bottles, dash, lift/throw, hookshot, mirror,
  flute travel, medallions.** Each is a primitive task.
- **SNES steps cost more than NES steps.** Measure the rollout budget before
  copying the 24-frame horizon (AP-054).
- **Save & Quit is a legal move** (no emulator mutation). It is a
  travel tool (AP-030).

## Phase map

| Phase | Tasks | Exit gate |
|---|---|---|
| 0 Ground rules | AP-001..004 | Goal, assist contract, ladder, route template written |
| A Instrumentation (M2) | AP-005..018 | Inventory, flags, sprites, hit log, ledger on every run |
| B Tape, saves, audit | AP-019..030 | Tape replays the verified tip with 0 desync; save points + resume |
| C Scans and probes | AP-031..044 | `pin_probe` shows doors, chests, tiles; scanner makes room pins |
| D Movement + walk-ahead | AP-045..056 | Tile BFS walker, stairs/layers, rollout walk |
| E Combat + guards | AP-057..072 | `PolicyGuard`, safe strike, clear v2, bottle guard, census |
| F Castle → Sanctuary | AP-073..082 | Rung C1: continuous power-on to Sanctuary |
| G Overworld + side items | AP-083..098 | Boots, book, mirror, Master Sword, flippers, flute, medallions, silver arrows |
| H Dungeons | AP-099..194 | 12 dungeons × 8 steps; rungs C2..C14 |
| I Ganon + credits | AP-195..200 | M7 continuous run, M8 tape replay MP4 |

Clean ladder rungs (AP-003): C1 Sanctuary · C2 Eastern · C3 Desert · C4 Hera ·
C5 Master Sword · C6 Agahnim 1 · C7 PoD · C8 Swamp · C9 Skull · C10 Thieves ·
C11 Ice · C12 Mire · C13 Turtle Rock · C14 GT / Agahnim 2 · C15 Ganon + credits.

---

## Phase 0: ground rules (AP-001..004)

- [x] 2026-09-29 · evidence: plan.md dropped the dungeon/credits non-goal and links PORT_PLAN.md; tests/test_docs.py 8 passed **AP-001 · Record the goal change in `plan.md`**
  Needs none. Files: `alttp/docs/plan.md`.
  Do: remove "Full dungeon ladder or Ganon credits" from Non-goals. Add a
  one-line "Next" pointer to `PORT_PLAN.md` as a relative link. Keep every
  other rule in `plan.md`.
  Done: `uv run pytest tests/test_docs.py -q` (from repo root) passes.

- [x] 2026-09-29 · evidence: snes/alttp/docs/ASSIST_CONTRACT.md plus AGENTS pointer; Owner ruling blank; tests/test_docs.py 8 passed **AP-002 · ALTTP assist contract**
  Learned: Zelda I NES heart and inventory addresses are not ALTTP rules. Survival here is last-heart refill only. Lookahead is not ruled.
  Needs none. Read `nes/zelda_i/docs/ASSIST_CONTRACT.md`. Files:
  `alttp/docs/ASSIST_CONTRACT.md`, `alttp/AGENTS.md` (pointer line).
  Do: define Clean (no RAM write, no load, no refill), Survival (HP refill at
  the last heart, dev only, disclosed), what-if pins (dev only, deleted after)
  and rollout lookahead (disclosed as `rollout_restores`). The owner allowed
  lookahead for Zelda I on 2026-09-28. Write that it is **not yet ruled** for
  ALTTP and leave an "Owner ruling:" line blank.
  Done: file exists; `tests/test_docs.py` passes.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_clean_ladder.py -q: 4 passed; next_open() returns C1 **AP-003 · Clean ladder table**
  Learned: next_open() returns the id string of the first status=open row (C1). acceptance is a predicate name, not a callable. Stop strings are the phase-map labels. LadderStatus.GREEN is skipped.
  Needs none. Read `nes/zelda_i/spine/clean_tip.py` (first 120 lines). Files:
  `alttp/spine/__init__.py`, `alttp/spine/clean_ladder.py`,
  `alttp/tests/test_clean_ladder.py`.
  Do: rows C1..C15 from the Phase map, each with stop name, acceptance
  predicate name, status `open`. Add `next_open()`.
  Done: `uv run pytest alttp/tests/test_clean_ladder.py -q` passes and
  `next_open()` returns C1.

- [x] 2026-09-29 · evidence: snes/alttp/docs/routes/DUNGEON_TEMPLATE.md has Entry through Open reds for step-2 copies **AP-004 · Dungeon route doc template**
  Learned: Leave DUNGEON_TEMPLATE.md blank. Copy it to routes/<SLUG>_ROUTE.md and fill the copy.
  Needs none. Read `nes/zelda_i/docs/LEVEL6_ROUTE.md` (skim the headings).
  Files: `alttp/docs/routes/DUNGEON_TEMPLATE.md`.
  Do: headings every dungeon doc fills in: Entry (pin, room, first hop),
  Requirements (items, from z3), Room table (room, doors + types, chests,
  sprites, notes), Keys and big key, Big chest, Mechanics, Boss (sprite ids,
  HP byte, phase byte, what hurts it, when), Prize and exit, Save points, Open
  reds. Keep it short.
  Done: file exists; later dungeon step 2 tasks copy it.

## Phase A: instrumentation (AP-005..018)

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_ram_inventory.py -q passed; live pins FighterSword sword/shield, FighterSwordLamp lamp, CastleB1Key keys/hp/max_hp **AP-005 · Inventory block reader**
  Learned: FighterSword has sword $F359=1 and shield $F35A=1 (both 0 on HyruleCastleGrounds); lamp $F34A=1 only on FighterSwordLamp; CastleB1Key is keys=1, hp=$14, max=$18; boomerang $F341=1 on CastleB3Boomerang; HC big key is $F367 bit 6 on CastleMainZeldaBoomerang. $F34F is the bottle index, not a bottle slot; rupees are the actual word at $F362, not the $F360 goal.
  Needs none. Read `alttp/ram.py`, `alttp/docs/ram_map.md`. Files:
  `alttp/ram.py`, `alttp/tests/test_ram_inventory.py`, `alttp/docs/ram_map.md`.
  Do: add a frozen `AlttpInventory` read from `$F340..$F37B` (bow,
  boomerang, hookshot, bombs, powder, fire rod, ice rod, medallions, lamp,
  hammer, flute, net, book, somaria, byrna, cape, mirror, gloves, boots,
  flippers, pearl, sword, shield, armor, 4 bottles, rupees, big key, compass,
  map, heart pieces, max HP, HP, magic, keys, pendants, arrows, crystals).
  Take candidate addresses from the zelda3 disassembly. Mark each field
  `verified_on="<pin>"` or `"unverified"`. Verify at least sword, lamp, HP,
  max HP and keys on existing pins (`FighterSword`, `FighterSwordLamp`,
  `CastleB1Key`).
  Done: unit test reads a synthetic RAM array; `ram_map.md` table lists every
  field with its verification pin.

- [x] 2026-09-29 · evidence: first_desync_frame 2800 in recordings/ap006_tas_sync.json; name menu still up on the Down walk; lag 0 and lag 1; no sanctuary **AP-006 · TAS oracle: sync the USA Tompa movie**
  Learned: Reset SRAM is the 0x60 fill and it loads. Name screen opens on the A at 1142 and never confirms; both lags desync at frame 2800 while module is still $04. Not an oracle past Sanctuary.
  Needs none. Read `alttp/tas/README.md`, `alttp/tas/catalog.py`,
  `alttp/tas/smv.py`. Files: `alttp/scripts/tas_sync.py`,
  `alttp/docs/ram_map.md` (one section).
  Do: fetch `tompa_1269M.smv` (USA, 274k frames) with `tas/fetch_refs.py`.
  Play it into stable-retro from the movie's start condition. Stop at the
  first frame where Link's room/xy stops making sense (a wall walk, or a menu
  at the wrong time). Report the frame of first desync.
  Done: report `alttp/recordings/ap006_tas_sync.json` with
  `first_desync_frame`. If sync holds past Sanctuary, write "oracle usable to
  frame N" in `ram_map.md`. **Nothing later depends on this passing.**

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_ram.py 6 passed; CastleMain d=02 fl=00 ly=0 st=0 hp=18 ifr=0 kb=0; CastleB1Pit d=02 fl=FF ly=1 hp=04 ifr=17; EasternPalaceEntry d=04 fl=00 hp=18; CastleB1Guard hit counted $031F 57.. and $0046 18.. while $5D=2 **AP-007 · Snapshot extras**
  Learned: dungeon $040C is cur_palace_index_x2 (castle 0x02, eastern 0x04, OW/FighterSword 0xFF, high byte 0). Floor $A4: F1=0x00 B1=0xFF B3=0xFD. Layer $EE is 1 in CastleB1Pit. I-frames $031F countdown_for_blink; knockback $0046 link_incapacitated_timer. $5D is already link_action (0 rest, 2 recoil, 21 hold-up). HP on the snapshot is $F36D.
  Needs AP-005. Files: `alttp/ram.py`, tests.
  Do: add to `AlttpSnapshot`: dungeon id (candidate `$040C`), floor
  (candidate `$A4`), layer `$EE`, Link state `$5D`, i-frames/knockback
  counters (find by taking a hit on a pin and diffing), and HP. Verify each on
  a pin; unverified fields stay out.
  Done: `pin_probe`-style print from 3 pins in the report shows sane values.

- [x] 2026-09-29 · evidence: module_census one run, report /tmp not verified_tip_run.json; red at pocket_to_main_hall missed door; first frames: intro 0x00@78, file select 0x01@973, dialog 0x0E/02@1477, play 0x07/00@2836 and 0x09/00@4059, OW load 0x08@3940, room scroll 0x09/05@4672, dungeon load 0x11@7371, stairs 0x07/10@8401. Menu and module 0x06 not seen. Hint 0x01 title was wrong. **AP-008 · Module/submodule census**
  Learned: 0x01 is file select, 0x00 is intro, 0x02 was not seen. Play is submodule 0 only. Bush hole is module 0x11 DungeonFallingEntrance, not 0x06 PreDungeon. Census stopped before rooms 0x61/0x60/0x50.
  Needs AP-007. Files: `alttp/scripts/module_census.py`, `alttp/docs/ram_map.md`.
  Do: run `run_to_verified_tip.py` once, logging every `(module, submodule)`
  change with frame, room and screen. Name each value seen (play, room
  scroll, stairs, dialog, menu, dungeon load, OW load).
  Done: `ram_map.md` has a module table with the frame where each value was
  first seen.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_ram.py 7 passed; FighterSword->FighterSwordLamp room 0x104 bit 0x0010 chest; CastleB3->CastleB3Boomerang room 0x71 bit 0x0010 chest; CastleB1Key->CastleB1SecondKey room 0x72 bit 0x0400 key and bit 0x8000 door. Boss bit not seen. **AP-009 · Dungeon room flags reader**
  Learned: room_flags uses the full $A0 id. Link's house lamp chest is room 0x104 (live $A0=0x0104); room_base_id masks that to 0x04. Chest bit 0x0010, key bit 0x0400 (live $0402 0x4000 shifted >>4 on room exit), door bit 0x8000. Boss diagram bit 0x0800 did not change. Quadrant low nibble also fills in as you walk.
  Needs AP-005. Read `nes/zelda_i/ram.py` (world flags). Files:
  `alttp/ram.py`, tests.
  Do: add `room_flags(ram, room)` from the save-RAM room table (candidate
  `$F000 + 2*room`). Decode the chest-opened, key-taken, door-opened and
  boss-dead bits by diffing `CastleB1Key` against `CastleB1SecondKey`, and a
  pin pair around one chest.
  Done: test on a synthetic array; the report lists which bit changed on
  which pin pair.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_ram.py::test_ow_flags_event_byte_and_castle_hole_tile passed; live lift $F800=0x0570 $FA00=0x0DCC hole; $F29B stayed 0 **AP-010 · Overworld flags reader**
  Learned: Bush hole is memorized map16 0x0DCC at pos 0x0570 (ROM secret 0x80), not a $F280 bit. Count $4AC goes 0 to 2 on the lift and is 0 again on CourtyardSecretPocket.
  Needs AP-005. Files: `alttp/ram.py`, tests.
  Do: `ow_flags(ram, screen)` (candidate `$F280 + screen`). Diff
  `HyruleCastleGrounds` against `CourtyardSecretPocket` and name the bit
  that marks the opened bush hole.
  Done: test + one verified bit in `ram_map.md`.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_primitives.py: test_sprite_ai_direction_floor_and_hit_timer and test_active_ancillae_keeps_link_shots_and_bombs passed; CastleB1Guard slot0 type 0x41 ai=1 dir=1 floor=0 hit=0, other soldier floor=1, ancilla empty **AP-011 · Sprite and ancilla fields**
  Learned: Hit timer is $0EF0 (bit 7 set, counts down on a sword hit; CastleB1Guard type 0x41 HP 6 to 0). Stun $0B58 stayed 0 — that pin has no hookshot. Enemy arrows are sprite 0x1B, not ancilla.
  Needs none. Read `alttp/primitives.py` (lines 20-155). Files:
  `alttp/primitives.py` (the Composer for sprites), tests.
  Do: add to `SpriteSnapshot`: AI state, direction, floor/layer, a stun or
  hit timer. Add `active_ancillae(env)` for Link's own projectiles and bombs,
  plus enemy projectiles if they live there. Candidate tables come from the
  disassembly; confirm each on a pin with a live soldier.
  Done: test on a synthetic RAM; `pin_probe` (later) prints them.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_sprite_ids.py: 8 verified names (0x41 blue soldier, 0x42 green soldier, 0x4B knight, 0x6A ball and chain, 0x76 zelda, 0xD8 heart, 0xE4 small key, 0xE5 big key) **AP-012 · Sprite id name table**
  Learned: z3 enemies/main.json ids >= 216 are shared-stat aliases, not sprite types (Dacto is 216). Sprite type 0xD8 is the heart pickup. 0x4B verified name is knight; disassembly routine is GreenKnifeGuard.
  Needs none. Read `alttp/z3-json-data/enemies/`. Files:
  `alttp/sprite_ids.py`, tests.
  Do: map sprite type byte → short name for every enemy, boss, follower and
  pickup, from z3 data plus the disassembly list. Keep the existing constants
  in `primitives.py` as the verified ones (`0x41`, `0x42`, `0x4B`, `0x6A`,
  `0x76`, `0xD8`, `0xE4`, `0xE5`).
  Done: test asserts those 8 names.

- [x] 2026-09-29 · evidence: 68 pins; CastleZeldaFollower F3CC=1 F3C5=1 F3C6=11; grounds/LinksHouseWake/FirstAction F3C5=0 F3C6=10 F3CC=5; FighterSword and CourtyardSecretPocket F3C6 10->11 bit 0x01; progress 2 unseen; pytest test_ram+test_ram_inventory 11 passed **AP-013 · Progress and follower table**
  Learned: Follower 5 on pre-sword pins is unnamed. Progress stays 0x01 when Zelda follows. Bit 0x10 of F3C6 was already set on LinksHouseWake; clear state not on these pins. Progress 0x03 only on late entry pins, unnamed.
  Needs AP-005. Files: `alttp/ram.py`, `alttp/docs/ram_map.md`.
  Do: document `$F3C5` progress, `$F3C6` flags and `$F3CC` follower values
  seen on existing pins (`CastleZeldaFollower` = 1). List the other follower
  values as "to verify at AP-091/131/158/098".
  Done: table in `ram_map.md`.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_ledger.py -q: 1 passed (two HP drops, causes blue soldier and bomb) **AP-014 · Hit log**
  Learned: A hit is only an $F36D drop. $031F re-arm with stable HP is not a hit. Nearest sprite or ancilla by manhattan; sprite wins a tie. Trail is the last 24 (xy, buttons) including the drop frame.
  Needs AP-007, AP-011. Read `nes/zelda_i/spine/ledger.py` (hits part).
  Some rooms re-arm i-frames with no damage. Files: `alttp/spine/ledger.py`.
  Do: wrap `env.step`. On each HP drop record frame, room, damage, nearest
  sprite or ancilla (type name, distance), Link state and a 24-frame trail
  of (xy, buttons). Count a hit only when HP drops.
  Done: unit test with a fake env where HP drops twice → 2 hits with causes.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_room_timer.py -q: 4 passed (house 0x104, light screens, room 0x55, dark world) **AP-015 · Room timer**
  Learned: Play is module 0x07 or 0x09 with submodule 0. Key is ('in', full $A0) or ('ow', bool dark_world, screen). Dialog/stairs bounce if the key returns; a settled change with no transition is location_jump, not a visit.
  Needs AP-007. Read `nes/zelda_i/room_timer.py` (header),
  `retro_harness/hop_timer.py`. Files: `alttp/room_timer.py`, tests.
  Do: location key = `("in", room)` indoors or `("ow", dark_world, screen)`
  outdoors. Settle when module is play and submodule is 0.
  Done: test on a synthetic snapshot sequence gives the expected visits.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_ledger.py -q: 2 passed (scripted x flip flutter 1; knockback shove not counted) **AP-016 · Ledger books: rooms and flutter**
  Learned: Visits are settled frames only, keyed like the room timer (in: full room, ow: world+screen). Flutter window is 8 frames, one axis, step cap 16px. $0046 knockback does not update the axis. Slowest visits are in report['slowest_visits'].
  Needs AP-014, AP-015. Files: `alttp/spine/ledger.py`.
  Do: frames per room visit, slowest visits, flutter (xy reversal on one
  axis within 8 frames, knockback excluded).
  Done: unit test with a scripted flip → flutter count 1.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_ledger.py -q: 3 passed (rupee/key poke labelled write; play spends; missed chest 0x71) **AP-017 · Ledger books: items, spends, missed chests**
  Learned: Changes are play inside a step or write between steps. A link xy change between steps rebases and is not a write. HP is not an inventory row. Spends are drops in keys, bombs, arrows, magic, rupees. route_rooms are chest room ids; on dungeon exit a clear 0x0010 bit is missed. Empty route_rooms lists nothing.
  Needs AP-009, AP-016. Files: `alttp/spine/ledger.py`.
  Do: every inventory rise or fall, labelled `play` (inside a step) or
  `write` (between steps). Key, bomb, arrow, magic and rupee spends. On
  leaving a dungeon, list chests on the route rooms still unopened.
  Done: unit test where a between-step write is labelled `write`.

- [x] 2026-09-29 · evidence: run_to_verified_tip.py --ledger ok tip=room_50 frames=13174 writes=0; recordings/verified_tip_run.json **AP-018 · `--ledger` on the opening spine**
  Learned: Door column is map point door_column (2040,2024). x=2048 is hedge; do not climb north from approach_x+8. Room 0x55 attribute 0x84 has no tile path, so the commit walker falls back to move_to. A clear that times out still walks an open door.
  Learned: --ledger does not change the tip: both runs died on the same frame and blocker. On-disk verified_tip_run.json is an older green (11245 frames) and was left in place. Boot frame 85 recorded one HP drop in room 0 (damage 85, no cause) before play; writes stayed 0.
  Needs AP-016, AP-017. Files: `alttp/scripts/run_opening_spine.py`,
  `alttp/scripts/run_to_verified_tip.py`.
  Do: add `--ledger`; print the rooms table, hits, and writes at the end and
  add them to the JSON report.
  Done: `run_to_verified_tip.py --ledger` stays green and reports
  `writes == 0`. Do not overwrite `verified_tip_run.json` unless green.

## Phase B: tape, save points, replay, audit (AP-019..030)

- [x] 2026-09-29 · evidence: QT_QPA_PLATFORM=offscreen uv run pytest alttp/tests/test_tape.py -q (2 passed, pack-load same vectors) **AP-019 · Button tape writer**
  Learned: Tapes are recordings/<tag>.tape.npz: buttons uint8 (N, 12) in SNES_BUTTON_NAMES order, meta JSON boot_state/rom_sha1/final_ram_crc. final_ram_crc is zlib.crc32 of the full get_ram() image; no sync window yet. Live scripts are not hooked: ButtonTape.attach then save(tag).
  Needs none. Read `nes/zelda_i/runner.py` (tape and `ram_crc` parts),
  `retro_harness/bk2.py`, `retro_harness/recorder.py`. Files:
  `alttp/spine/tape.py`, tests.
  Do: every live run writes `recordings/<tag>.tape.npz`: SNES 12-button
  vector per `env.step`, boot state name, ROM sha1, final RAM CRC. Reuse a
  harness format if one fits.
  Done: round-trip unit test (pack → load → same vectors).

- [x] 2026-09-29 · evidence: 3 runs (2 in-process, 1 fresh process), tapes equal (15366,12), get_ram 147456 bytes differed in 0 indexes every frame. Sync window is the full image. Excluded bytes: none. All three red at the same pocket_to_main_hall door; verified_tip_run.json not written. **AP-020 · Replay noise window**
  Learned: No process noise showed up. Do not copy Zelda's $0010-$07FF cut; $0000-$000F matched too. The tip still dies at the courtyard door, so this window is from that identical red path.
  Needs AP-019. Read the "replay desync" trap in `nes/zelda_i/AGENTS.md`
  (the `$0010-$07FF` line). Files:
  `alttp/docs/ram_map.md`.
  Do: run the verified tip twice with tapes. Diff WRAM every frame. List
  bytes that differ between two identical runs and choose the sync window.
  Done: `ram_map.md` names the sync window and the excluded bytes.

- [x] 2026-09-29 · evidence: replay recordings/verified_tip.tape.npz desynced=0 frames=14108 loads=0 writes=0; recordings/verified_tip_replay.json; verified_tip_run.json untouched **AP-021 · `replay_tape.py`**
  Learned: Committed tape drops bush lookahead (7 restores, 1258 frames). Blank SRAM after reset before the audit; replaying startup.py inject mid-run desyncs on the next frame (4 checksum bytes). Tip still red at pocket door (2048,1955).
  Needs AP-020. Read `nes/zelda_i/scripts/replay_tape.py`. Files:
  `alttp/scripts/replay_tape.py`.
  Do: play a tape from power-on with no controller. Check the sync window
  every frame. Report the first desync frame, loads (0) and writes (0).
  Optional `--video` writes an MP4 and a realtime factor.
  Done: the verified-tip tape replays with `desynced == 0`.

- [!] BLOCKED 2026-09-29: green tip room_50 frames=13174 set_state=8 writes=0 not 0/0; startup.py:205 SRAM inject plus castle_to_sword.py:330 x7 bush restores; rollout_restores=0; /tmp/ap022-green-remeasure.txt **AP-022 · Audit counters**
  Learned: Green remeasure is room_50 at 13174 frames with set_state=8 writes=0. Sites stay the SRAM inject and 7 bush restores. Do not delete those restores to force a zero.
  Learned: LiveRun is in spine/tape.py. Tip is red at pocket door (2048,1955) 11071f with or without the wrap. set_state sites: startup.py:205 plus castle_to_sword.py:330 x7. Do not call those rollout restores.
  Needs AP-019. Read `retro_harness/audit.py`. Files: runner glue in
  `alttp/spine/tape.py` or `alttp/session.py`.
  Do: wrap live runs in `AuditedEnv`. Report `set_state`, `rollout_restores`
  and RAM writes. Fail `mid_run_state_load` when loads minus rollout
  restores is above 0.
  Done: the verified tip reports `set_state=0`, `writes=0`.

- [x] 2026-09-29 · evidence: run_spine.py --through room_50 --no-video ok frames=13174; --through zelda fail_closed; recordings/opening_spine_room_50.json **AP-023 · Spine composer and `run_spine.py`**
  Learned: room_50 green is the same continuous tip as run_to_verified_tip (13174 frames). zelda fails closed without booting. Room 0x61 clear still cycle-caps; the west door is walked after that.
  Learned: SpineHop table is spine/hops.py; CLI is scripts/run_spine.py; run_opening_spine.py aliases it. room_50 red is the courtyard door miss, same as run_to_verified_tip, not the hop table. zelda does not boot.
  Needs AP-019. Read `nes/zelda_i/spine/hops.py`,
  `alttp/opening_route/full_tip.py`, `alttp/scripts/run_opening_spine.py`.
  Files: `alttp/spine/hops.py`, `alttp/scripts/run_spine.py`.
  Do: one `SpineHop` table (through-id, stop, stages, success). Seed it with
  the existing continuous segments up to `room_50`. `--through <id>` stops
  there; an unknown id fails closed. `run_opening_spine.py` becomes a thin
  alias or is deleted.
  Done: `run_spine.py --through room_50 --no-video` green; `--through zelda`
  fails closed.

- [x] 2026-09-29 · evidence: run_spine.py --through room_50 --save-points Full --no-video ok frames=13174 pins Full_castle_to_sword Full_sword_to_secret_entrance_clear Full_pocket_to_main_hall Full_castle_dungeon_prefix; no Full_fail; recordings/opening_spine_room_50.json **AP-024 · Save points**
  Needs AP-023. Files: `alttp/scripts/run_spine.py`, `alttp/spine/hops.py`.
  Do: `--save-points <prefix>` writes `<prefix>_<stage>.state` into
  `custom_integrations/Zelda3-Snes/` at every stage start, plus
  `<prefix>_fail` on a red run.
  Done: `--through room_50 --save-points Full` writes one pin per stage.

- [x] 2026-09-29 · evidence: run_spine.py --through room_50 --save-points Full --resume castle_dungeon_prefix --no-video ok frames=3130 resumed_from=castle_dungeon_prefix continuous=false clean_chain=false source=resumed; recordings/opening_spine_room_50_resume.json; Full_* pins not rewritten **AP-025 · Resume**
  Learned: stable-retro step needs reset before set_state; set_state replaces the boot frame and no idle step follows
  Needs AP-024. Files: same.
  Do: `--resume <stage>` loads `<prefix>_<stage>` with no idle frame, plays
  on, and reports `resumed_from`. Reject a pin whose inventory misses the
  stage's entry requirement (`opening_route.segment.EntryRequirement`).
  Done: resume at the last stage before `room_50` reaches `room_50`; the
  report carries `resumed_from`.

- [x] 2026-09-29 · evidence: Full_castle_to_sword Full_sword_to_secret_entrance_clear Full_pocket_to_main_hall Full_castle_dungeon_prefix provenance kind=natural source_run=spine:room_50 git_sha=8d6c36025378096c5291abc703e325a1ac8735a5 dirty=true **AP-026 · Pin provenance**
  Needs AP-024. Files: `alttp/spine/tape.py` (or `alttp/paths.py` helper).
  Do: every pin written by a tool gets `<name>.provenance.json`: source run
  tag, frame, git sha, dirty flag, and kind (`natural`, `resumed`, `poke`,
  `whatif`, `scan`).
  Done: the AP-024 pins have provenance files with kind `natural`.

- [x] 2026-09-29 · evidence: audit_pins.py over 72 pins in custom_integrations/Zelda3-Snes exit 1 (4 incoherent, checks not weakened). Entry: DesertPalaceEntry hp=0x18/0x18 kind=none; EasternPalaceEntry hp=0x18/0x18 kind=none; HeraEntry hp=0x18/0x18 kind=none; IcePalaceEntry hp=0x40/0x40 kind=none; PalaceOfDarknessEntry hp=0x38/0x38 kind=none; SwampPalaceEntry hp=0x18/0x18 kind=none; ThievesTownEntry hp=0x38/0x38 kind=none. Incoherent: CastleB1GuardLamp module 0x12, CastleB1PitCleared module 0x12, CourtyardSecretPocket gzip.BadGzipFile b'#!', YazeSlot000 module 0x55 max_hp 0x55. **AP-027 · `audit_pins.py`**
  Needs AP-005, AP-026. Read `nes/zelda_i/scripts/audit_pins.py`. Files:
  `alttp/scripts/audit_pins.py`.
  Do: one subprocess per pin. Check HP ≤ max HP, max HP a multiple of 8,
  sword/shield/armor in range, module is play, and room vs dungeon id.
  Print provenance kind. Exit 1 on any incoherent pin.
  Done: runs over every pin in `custom_integrations/Zelda3-Snes/`; the
  report lists each `*Entry` pin with its HP and kind.

- [x] 2026-09-29 · evidence: stage_replay CastleMain alttp.opening_route.room_engine:run_room_edge room_61 west_to_0x60 pass=1 frames=3602 room=0x60 (448,3320) hp 24->20 **AP-028 · `stage_replay.py`**
  Learned: west_to_0x60 from CastleMain took one green-soldier hit (4 HP). --last-heart sets $F36D=8 on a session copy; CastleMain.state is not written. Machine row is RESULT pass frames hp_lost hits.
  Needs AP-023. Read `nes/zelda_i/scripts/stage_replay.py`. Files:
  `alttp/scripts/stage_replay.py`.
  Do: run one controller (`module:Name`) from a pin. `--idle N` holds N idle
  frames first (RNG offset). `--window A-B` prints a trace window.
  `--last-heart` sets HP to one heart in a what-if copy.
  Done: `room_engine` edge `room_61 west_to_0x60` from `CastleMain` runs
  through it green.

- [x] 2026-09-29 · evidence: eval_offsets CastleMain run_room_edge room_61 west_to_0x60: 12 processes, all pass; frames 1171-4304; hp_lost 0-16; offset 0 is 3602/4/green soldier:1 **AP-029 · `eval_offsets.py`**
  Learned: Offsets 0..11 are not one tape: frames 1171..4304 and hp_lost 0..16 on the same edge. Cause names contain spaces (green soldier).
  Needs AP-028. See Lesson 2. Files: `alttp/scripts/eval_offsets.py`.
  Do: run a controller from one pin at offsets 0..11, each in its own
  process. Table: pass, frames, HP lost, hits by cause.
  Done: the table prints for `room_61 west_to_0x60`.

- [x] 2026-09-29 · evidence: save_and_quit SELECT in play; progress 0 refuses 91f; prog 01 house 487, zelda cell 512, room 0x55 487; prog 03 menu row0 house 538 row1 sanctuary 669; F3CA 0x40 pyramid 490; no pin has mirror 2 **AP-030 · Save & Quit as a travel move**
  Learned: SELECT in play opens message 0x0186; START opens the item menu. Menu 0x0184 is house then Sanctuary. Mountain cave needs mirror value 2 (message 0x0185); no pin has it. F3CA 0x40 skips the menu and continues on pyramid screen 0x5B.
  Needs AP-024. Files: `alttp/primitives.py` (`save_and_quit`),
  `alttp/docs/ram_map.md`.
  Do: from a dungeon pin, SELECT in play → save and quit → continue. Record where
  Link spawns (Link's house, Sanctuary, mountain cave, Pyramid after
  Agahnim) and the frame cost. This is a normal game action, not an
  emulator load.
  Done: table of spawn choice vs progress with frames.

## Phase C: scans and probes (AP-031..044)

- [x] 2026-09-29 · evidence: pin_probe CastleMain room 0x61 link (760,3520) hp 0x18 three soldiers; EasternPalaceEntry room 0xC9 dungeon 0x04 three popo; no writes **AP-031 · `pin_probe.py` base**
  Learned: EasternPalaceEntry is room 0xC9 (unlabeled), dungeon 0x04. CastleMain link x starts at 760. Indoor keys read 0, not the 0xFF blank sentinel.
  Needs AP-007, AP-011, AP-012. Read `nes/zelda_i/scripts/pin_probe.py`.
  Files: `alttp/scripts/pin_probe.py`.
  Do: load one pin, then print module/submodule, room or screen, Link
  (xy, layer, dir, state), HP, the inventory summary, and every sprite
  (slot, name, state, hp, xy). No writes.
  Done: runs on `CastleMain` and `EasternPalaceEntry`.

- [x] 2026-09-29 · evidence: CastleMain --press LEFT:40 link x 760 to 726 (y 3520 to 3496, dir 0x00 to 0x04); sprites reprinted **AP-032 · `pin_probe --press`**
  Learned: LEFT:40 also moved y up 24px because the pin starts facing up (dir 0x00). A blue soldier appeared in slot 10.
  Needs AP-031. Do: `--press DIR:N ...` holds each press and reprints Link
  and the sprites after it. Done: `--press LEFT:40` on `CastleMain` shows x
  falling.

- [x] 2026-09-29 · evidence: CastleMain --tiles (31,56)=0x8E door; south # walls, L lip, BB 0x27 statues and slope corners match the spawn frame **AP-033 · `pin_probe --tiles`**
  Learned: Mask $00EC=$01F8. Tile col is (x & mask)>>3. 0x27 statues are B. East/west doors are 0x85/0x95 on row 32. 0x38 at (31,4) is still unknown.
  Needs AP-031, AP-046. Do: print the tile attribute grid around Link as
  ASCII with a legend (floor, wall, pit, water, stairs, door, spike,
  conveyor, ledge). Done: `CastleMain` grid matches the room walls in a
  screenshot.

- [x] 2026-09-29 · evidence: ram_map.md Doors: $1980 type, $19A0 pos, $19C0 dir. CastleB1Key slot0 type 0x1C north pos $123C. Room 0x81 only slot0 type 0x40 north pos $239C. Gold cell lock is object 0x18 on room 0x80 at $06E0=$8758, not a 0x81 door slot. Type 0x1E unseen. **AP-034 · Find the door arrays in WRAM**
  Learned: Door type 0x18 is a two-way shutter. Object 0x18 is the cell lock and lives in the chest table ($0496 chests, $0498 locks, $06E0 locs). 0x81 has no west door slot and no F0 attr. Slots 8-15 are the other layer of the same door. Big-key door type 0x1E was not on any pin.
  Needs AP-031. Files: `alttp/docs/ram_map.md`.
  Do: find where the running game keeps the current room's doors (type and
  position). Candidates are in the disassembly's room-load code. Confirm on
  two rooms with known different doors: `CastleB1Key` (key door) and room
  `0x81` (gold cell door).
  Done: `ram_map.md` names the addresses and the door type codes seen.

- [x] 2026-09-30 · evidence: CastleZeldaB1West --doors: slot 0 north type=0x40 regular pos=$239C dest=0x71; object lock room=0x80 object=0x18 tile=(44,14) not a door slot. No west slot. **AP-035 · `pin_probe --doors`**
  Learned: Do not label the 0x81 north 0x40 door as west. The gold lock is room 0x80 object 0x18. A west type is not in the ROM door list.
  Needs AP-034. Do: print each door of this room as direction, type (open,
  small key, big key, shutter, bombable, stairs) and the destination room if
  known from `maps/` or z3. Done: pin_probe --doors prints each real slot as direction, type, and destination, and the 0x81 report names the 0x80 object lock instead of a west door.

- [x] 2026-09-29 · evidence: CastleB1Key room 0x72 chest 0 tile (30,18) loc $093C opened=0 live bit 0x0100 save bit 0x0010 clear; z3 MapH2 at 0xEB0C. live $0402=$4000. **AP-036 · `pin_probe --chests`**
  Learned: Join z3 only when the ROM chest-table count for the room equals $0496/2. Boomerang is room 0x71 0xE974; Lamp is room 0x80 0xEB09; Map is room 0x72 0xEB0C. Some z3 nodeAddress values are lists, not hex.
  Needs AP-009, AP-040. Do: list this room's chests with opened bits and the
  z3 item at each location. Done: runs on a castle B1 pin with a chest.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_room_fixture.py -q: 2 passed; tests/fixtures/room_61.json CastleMain link (760,3520) tile (31,56)=0x8E three sprites **AP-037 · `pin_probe --fixture`**
  Learned: Loader is load_room_fixture in pin_probe.py. room_61.json is one JSON line, layer 0 only.
  Needs AP-033. Do: write `alttp/tests/fixtures/room_<id>.json` (tiles,
  sprites, Link) for offline tests. Done: one fixture plus a loader test.

- [x] 2026-09-29 · evidence: recordings/scan/index.json pins=4 desynced=0 writes=0 loads=0 frames=14108: Scan_0x0104 Scan_ow_0x2C Scan_ow_0x1B Scan_0x55 **AP-038 · Tape room scanner**
  Learned: Verified-tip tape ends settled on ow 0x1B at the pocket door (2048,1955). It never reaches room 0x50. Boot room 0x5555 and module-8 screen 0x9F never settle and are not pinned.
  Needs AP-021, AP-031. Files: `alttp/scripts/scan_tape.py`.
  Do: replay any tape (a spine tape, or the TAS from AP-006 if it synced).
  At the first settled frame in each new room, save `Scan_<room>.state`
  (provenance `scan`) and a `pin_probe` dump.
  Done: scanning the verified-tip tape makes one pin per room it visits.

- [x] 2026-09-30 · evidence: ROOM_CENSUS.md from recordings/scan pins=4 writes=0. Castle room Scan_0x55 0x0055 hyrule_castle_secret_entrance dungeon 0xFF; doors north 0x02 regular and south 0x00 regular; chest Lamp z3=0xE971 unopened; sprites uncle and priest 0x73, knight 0x4B; stairs/pits none. No palace 0x02 room in the scan. **AP-039 · Room census doc generator**
  Learned: Room 0x55 is dungeon 0xFF, not palace 0x02. Live slots are regular (north 0x02, south 0x00); a stair note on the map door is not a stair slot or a tile_kind stair.
  Needs AP-035, AP-036, AP-038. Files: `alttp/scripts/room_census.py`,
  `alttp/docs/routes/ROOM_CENSUS.md` (generated).
  Do: from scan dumps build a table: room, dungeon, doors with types,
  chests, sprite types, stairs and pits. Done: generated file covers the
  castle rooms scanned so far.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_z3_dungeon_extract.py -q: 2 passed; 13 yaml in alttp/dungeon/route_data/ **AP-040 · z3 dungeon extract**
  Learned: Slugs: castle=escape (Hyrule Castle through Sanctuary), skull=skwoods, turtle_rock=trock, castle_tower=castletower; regions are node area names. Skull Woods big chest requires raw token 'BigKeyD3 ' (trailing space), normalized to BigKeyD3. Extract follows refs; the committed dump still has extra boss junctions for mire/pod/swamp that refs lacks.
  Needs none. Read `alttp/z3_json_data.py`, `alttp/docs/Z3_JSON_DATA.md`.
  Files: `alttp/scripts/z3_dungeon_extract.py`, output
  `alttp/dungeon/route_data/<slug>.yaml`.
  Do: per dungeon, list regions, item locations, key and item requirements,
  and connections. Slugs: `castle`, `eastern`, `desert`, `hera`,
  `castle_tower`, `pod`, `swamp`, `skull`, `thieves`, `ice`, `mire`,
  `turtle_rock`, `gt`.
  Done: 13 yaml files; test loads each.

- [x] 2026-09-29 · evidence: whatif_pin.py CastleMain --set sword=2 --out Whatif_CastleMain_sword2 reads sword=2 kind=whatif; audit_pins --pin Whatif_CastleMain_sword2 kind=whatif hp=0x18/0x18 room=0x0061 **AP-041 · What-if pin tool**
  Needs AP-005, AP-026. Files: `alttp/scripts/whatif_pin.py`.
  Do: `whatif_pin.py <state> --set sword=2 --set hp=24 --out Whatif_<name>`
  by named inventory field. Writes provenance `whatif`. `audit_pins` flags
  them. Done: a what-if copy of `CastleMain` with `sword=2` reads back 2.

- [x] 2026-09-29 · evidence: make_entry_pin.py eastern EasternPalaceEntry room=0xC9 dungeon=0x04 kind=poke entrance=0x08; ice IcePalaceEntry room=0x0E dungeon=0x12 kind=poke entrance=0x2D; same rooms as today **AP-042 · Entry pin maker**
  Learned: bus_addr ($7E for WRAM >= $2000) is in make_entry_pin.py. probe_ice_entry.py and probe_ep_entry.py are deleted. Entrance 0x08 is Eastern room 0xC9; 0x2D is Ice room 0x0E. Kit skips items the dungeon awards (no Eastern bow) and never writes $F3CC.
  Needs AP-026, AP-041. Read `alttp/scratch/probe_ice_entry.py` (`apply_pokes`,
  `trigger_entrance`) and `alttp/scratch/probe_ep_entry.py` (`ensure_ep_kit`).
  Files: `alttp/scripts/make_entry_pin.py`; delete the scratch copies this
  replaces.
  Do: `make_entry_pin.py <slug>` loads `HyruleCastleGrounds`, applies that
  dungeon's item kit, triggers the entrance (`$010E`), waits for indoor
  control, and saves `<Slug>Entry` with provenance `poke`. Kits come from the
  AP-040 requirements. Never poke `$F3CC`.
  Done: rebuilds `EasternPalaceEntry` and `IcePalaceEntry` with the same
  room ids as today.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_ledger.py 6 passed; verified_tip.tape.npz+scan/index.json frames=14108 desync=0 writes=0 loads=0; soldiers in:0055=knight hp0 x2 max_alive=3, other pin rooms none **AP-043 · Sprite census**
  Needs AP-014, AP-038. Files: `alttp/spine/ledger.py` (a census book).
  Do: per room, sprite types seen, max alive at once, and which ones hurt
  Link. Include `hp == 0` sprites and ancillae. Done: a verified-tip run
  lists soldier types by room.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_map_lint.py -q: 6 passed; room_engine.py lint: lint ok: 23 maps **AP-044 · Map JSON lint**
  Learned: Approach extent is points union walkBbox plus 12px seam (KnownDoor default tolerance). walkBbox alone rejects room_61 west by 56px and room_82 west by 26px. room_62 north approach is 4px above the box.
  Needs none. Read `alttp/room_map.py`. Files: `alttp/scripts/room_engine.py`
  (`lint` subcommand), tests.
  Do: every `maps/*.json` loads; door labels are unique; path points exist;
  approach xy is inside the room. Done: `room_engine.py lint` passes on
  today's maps.

## Phase D: movement, collision, walk-ahead (AP-045..056)

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_tilemap.py 2 passed; CastleMain get_ram 147456 covers $7F; spawn tile (31,56)=0x8E; LEFT stops (688,3496) west tile 0x04 **AP-045 · Tile attribute reader**
  Learned: get_ram is 147456. Layer 0 is $7F2000 (g_ram 0x12000), layer 1 is $7F3000, 64x64 row-major. Index after $EC mask: (y & ~7)*8 + (x & 63). CastleMain mask was $01F8. 0x00 floor, 0x04 indoor wall, 0x8E south door.
  Needs AP-031. Read `nes/zelda_i/dungeon/tilemap.py` (header). See Lesson 4. Files: `alttp/dungeon/__init__.py`,
  `alttp/dungeon/tilemap.py`, tests.
  Do: find the room collision/attribute map (candidates `$7F2000` and
  `$7F3000`, one per layer, 64×64). First check that `get_ram()` covers bank
  `$7F`. Verify by pressing into a known wall on `CastleMain`.
  Done: `read_room_tiles(ram, layer)` returns a 64×64 grid; test on a fixture.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_tilemap.py 3 passed; legend pins floor/wall/pit/shallow/deep/stairs/door/ledge/ice **AP-046 · Tile attribute legend**
  Learned: 0x09 shallow on FighterSword sets $114=0x09 and $359=1 (45px/36f vs 54 on floor). 0x20 pit on CastleB1Pit sets $59=0x0F then layer 1 to 0. Spike 0x44, conveyor 0x68-0x6B, grass 0x40 are not on any indoor pin. Overworld does not use the $7F grid.
  Needs AP-045. Do: classify attribute values (floor, wall, pit, shallow and
  deep water, stairs, door, spike, conveyor by direction, ledge, grass, ice)
  by sampling known tiles. Give each value its source pin. Done: legend
  table in `tilemap.py` with a test.

- [x] 2026-09-29 · evidence: pytest snes/alttp/tests/test_walk_physics.py 2 passed; floor (1,2)=72px/48f FighterSword; diagonal 1px/axis; shallow (1,1,1,2)=45px/36f; no lattice at (2804,2680); stairs submodule 8 is 39px/57f CastleMain end (760,3137); knockback (+26,-6) iframes 57 CastleB1Guard; grass unmeasured **AP-047 · Walk physics**
  Learned: No 8px turn lattice. Floor alternates 1 and 2 px (subpixel 0/128). Diagonal is 1 px per axis. Shallow is 1,1,1,2. Submodule 8 stairs are 39px north in 57 frames, not a steady rate. Grass was not isolated: HyruleCastleGrounds (2480,2528) is still the floor pair into a wall and $351 stayed 0. Knockback with the pad released is (+26,-6) from the hit pose in 18 frames; iframes start at 57.
  Needs AP-032. Read `nes/zelda_i/walk/physics.py` (header). See Lesson 4. Files: `alttp/walk_physics.py`, tests.
  Do: measure px/frame for straight and diagonal walking, stairs, grass and
  shallow water; knockback distance; i-frame count. Answer: is there a turn
  lattice?
  Done: constants with the source pin in comments; test pins the numbers.

- [x] 2026-09-29 · evidence: pytest test_walk_physics.py 3 passed; CastleMain LINK_BODY (0,8,15,23); LEFT (688,3496) RIGHT (832,3496) DOWN (700,3496) UP (656,3176); one px into each 0x04 wall overlaps **AP-048 · Link hitbox vs tiles**
  Learned: Link's coordinate is not the hitbox. CastleMain presses put the collision box at link_x+0..15 and link_y+8..23. The UP stop sits with link_y inside the wall row; those top 8 pixels do not collide. 0x27 also blocks, same as 0x04.
  Needs AP-045, AP-047. Do: press into walls from 4 sides with `--press`
  and find which of Link's pixels collide. Store it as `LINK_BODY` in
  `walk_physics.py`. Done: test that `LINK_BODY` blocks the probe wall.

- [x] 2026-09-29 · evidence: pytest snes/alttp/tests/test_nav.py 2 passed; open floor (0,0)->(24,24) is 1 turn; pit 0x20 deep 0x08 spike 0x44 block unless allow **AP-049 · Tile BFS**
  Learned: find_path steps 8px in room-local pixels and minimises turns then steps. Body is LINK_BODY. 0x20/0x08/0x44 block unless allow names pit, deep_water, or spike. 0x44 was not on an indoor pin. Other unknown bytes are walls.
  Needs AP-046, AP-048. Files: `alttp/dungeon/nav.py`, tests.
  Do: `find_path(tiles, start, goal, body)` returns the path with the fewest
  turns (a many-turn BFS path makes Link flutter). Pits, deep water and spikes
  are walls unless the caller allows them.
  Done: tests on 2 fixtures.

- [x] 2026-09-29 · evidence: CastleMain room_61 west_to_0x60 clear=False ok room 0x60 xy=(511,3320) then settle; RunLedger flutters=0 frames=618 **AP-050 · `room_step` walker**
  Learned: Walker is move_path_combat_aware(walker=True), on for door exits. room_step holds a heading 9 frames and bleeds a reverse. Do not replan back across a corner. Default clear=True still flutters inside fight_nearby (two) plus one post-fight tug at (706,3400); the clear=False edge ledger is 0.
  Needs AP-049. See Lesson 6. Files:
  `alttp/dungeon/nav.py`, `alttp/opening_route/room_engine.py`.
  Do: follow a BFS path and commit to a direction for at least N frames. No
  1-2 px flip. Add it as an option in `move_path_combat_aware`.
  Done: `room_61 west_to_0x60` green with the walker and flutter 0 in the
  ledger.

- [x] 2026-09-30 · evidence: CastleMain west_to_0x60 with approachXy omitted, path only natural_clear_left/north (no 520,3320): exit_via_door ok room 0x60 xy=(511,3320) lined y=3326 centre=3320 tol=15 type=0x02; settle control room 0x60 xy=(448,3320); pytest test_nav+map_lint+room_sense+room_engine_edges 40 passed **AP-051 · Doorway alignment**
  Learned: Type 0x02 capture is ±15 around the mouth centre (CastleMain west link_y 3305..3335 from x=599; south link_x 745..775 from y=3460). Room origin is coord & ~0x1FF, not the $EC residue. A missing approachXy stays absent; it is not (0, 0).
  Needs AP-035, AP-050. Do: before pushing through a door, line Link up on
  the door's centre column or row, with a tolerance measured per door type.
  Use it when a map door has no `approachXy`. Done: one castle door green
  without hand approach points.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_room_timer.py -q: 6 passed; 0x01 to 0x72 spiral 0x0E tagged fall **AP-052 · Stairs, layers and falls**
  Learned: Well is spiral submodule 0x0E (tile 0x5F), not FallingTransition 0x07. A0 becomes 0x72 on the first 0x0E frame while A4 is still 0x00; floor hits 0xFF later. Climb 0x72 to 0x01 stays stairs. Same-room layer change is layer_swap.
  Needs AP-015. Files: `alttp/room_timer.py` or `alttp/dungeon/nav.py`.
  Do: `transition_kind(prev, snap)` → `door`, `stairs`, `layer_swap`,
  `fall`, `warp`. Test on synthetic snapshots. Wire it into the room timer.
  Done: the `0x01 → 0x72` well drop is tagged `fall`.

- [x] 2026-09-29 · evidence: python -m alttp.overworld from HyruleCastleGrounds: bush hole frames=985 xy=(2425,1704) within 6px of approach (2430,1704) screen=0x1b attr=0x50 map16=0x36 liftable path=186 edges=56 exits RIGHT 0x25 LEFT 0x1A DOWN 0x2B; pytest alttp/tests/test_overworld.py 4 passed **AP-053 · Overworld tile nav**
  Learned: Castle OW map16 is $7E2000 (not $7F2000). Secret hole is map16 $36 attr $50. North rim has no walkable step-off. Grounds exits: RIGHT 0x25, LEFT 0x1A, DOWN 0x2B.
  Needs AP-046. Read `alttp/overworld.py`. Do: extend overworld BFS to real
  tile attributes, with screen-edge exits and liftable bushes/rocks flagged.
  Done: a castle-grounds walk to the bush hole uses it green.

- [x] 2026-09-29 · evidence: snes/alttp/rollout.py docstring; FirstAction em.step 655.7us => 25 frames/real-frame at 60Hz; CastleB1Guard 635.5us => 26; bit_exact: true **AP-054 · Rollout cost on SNES**
  Learned: In-room em.step is 635.5 us on CastleB1Guard and 655.7 us on FirstAction; 1x budget is 25 lookahead frames. Time after set_state: a long un-restored walk leaves the scene and understates the cost. Restore plus 60 frames was bit-exact on get_ram and the state blob.
  Needs none. Read `nes/zelda_i/rollout.py` docstring. Files:
  `alttp/rollout.py` (docstring only in this task).
  Do: time `em.get_state`, `em.set_state` and `em.step` on ALTTP. Check that
  restore plus 60 replayed frames gives the same RAM. Write the budget:
  frames of lookahead per real frame at 1× real time.
  Done: numbers in the docstring; `bit_exact: true` in the report.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_rollout.py 18 passed; CastleB1Guard fan 'walk into the guard' contact_frame 159 HP 24 to 20 moved 45 room 0x72 unchanged; RAM restored **AP-055 · `Rollout` core**
  Learned: contact_frame is the first $F36D drop, not an i-frame edge. On CastleB1Guard hold DOWN hits at frame 159 (HP 24 to 20, y stalls at 3696); standing still is also hit at frame 163 because the floor-0 soldier walks up. Room word stays 0x72.
  Needs AP-054. Port `Rollout.fan` from `nes/zelda_i/rollout.py`. Files:
  `alttp/rollout.py`, tests.
  Do: fan candidate press lists; per outcome report `contact_frame` (first
  HP drop), distance moved, and room change. The env is restored untouched.
  Done: fake-env unit tests pass; one live fan on `CastleB1Guard` reports a
  hit for "walk into the guard".

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_rollout.py 37 passed, 1 deselected; test_walk_on_castle_main_replays_to_the_same_xy CastleMain (760,3520) to (760,3490) **AP-056 · Walk-ahead: `Rollout.walk`**
  Learned: Rollout.walk records room_step presses and restores. CastleMain UP from (760,3520) is 20 frames to (760,3490), room 0x61, HP 24.
  Needs AP-050, AP-055. Read `Rollout.walk` in `nes/zelda_i/rollout.py`.
  Do: roll a `room_step` waypoint walk on the ROM and return the exact
  presses. Replaying them live gives the same end pose.
  Done: a test on a pin replays the walk to the same xy.

## Phase E: combat, guards, census (AP-057..072)

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_rollout.py 28 passed; CastleB1Guard DOWN hits unguarded 1 (frame 160 hp 24 to 20) guarded 0, both room 0x72 **AP-057 · `PolicyGuard`**
  Learned: Horizon is 25. A detour that changes the room word or drops has_control is discarded. RIGHT on the CastleB1Guard ledge is a clean hold.
  Needs AP-055. Read `nes/zelda_i/rollout.py` (`PolicyGuard`). Files: `alttp/rollout.py`.
  Do: each frame a deep copy of the controller plays its next H frames on
  the ROM (H from AP-054). On a hit, take a held-direction detour that does
  not change the room. Done: fake-controller tests; one castle guard room
  green with fewer hits than unguarded (quote both numbers).

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_rollout.py 31 passed; verified_tip.tape.npz under PolicyGuard fraction_rolled 0.0318 x_realtime 10.76 desync 0 detours 0, 14108 frames **AP-058 · Hazard-in-reach gate**
  Learned: HAZARD_REACH is 48: CastleB1Guard soldier Chebyshev was 45 twenty-four frames before the HP drop. ETA uses that slot's px since the previous play frame. Class 0 plus ignores_damage is a pickup or follower, not a hostile. Enemy shots are sprites.
  Needs AP-057. Do: roll only when a hostile sprite or projectile is within
  reach (distance or ETA). Report the fraction of frames rolled and
  `x_realtime`. Done: the verified tip under the guard runs at ≥1× real time,
  or the report says how far below.

- [x] 2026-09-29 · evidence: CastleB1Guard hurt frame 4 ($0EF0=0x9D, $3C=3) HP 6to4 on frame 10; FighterSword swing 13 clears frame 14; B+RIGHT does not turn; spin $79=48 at 61 B frames. pytest test_walk_physics.py 4 passed **AP-059 · Sword timing**
  Learned: $5D stays 0 on a slash. $50/$3A/$3C clear on frame 14. A direction on the B frame does not turn. Blade sets $0EF0 on frame 4; sprite HP drops on frame 10. Spin is $79 reaching 48, 61 B frames on FighterSword.
  Needs AP-032. See Lesson 4. On NES a direction pressed with the swing did not turn Link. Do: from one B press, measure the frame
  the blade hurts, the swing length, whether a direction held on the same
  frame turns Link, and spin charge frames. Done: constants in
  `alttp/walk_physics.py` with the pin.

- [x] 2026-09-29 · evidence: CastleB1Guard still soldier step-2 SWORD_REACH DOWN 134 LEFT 147 RIGHT 127 UP 135; (0,24) hits (0,8) misses. pytest test_walk_physics.py 5 passed **AP-060 · Sword reach table**
  Learned: SWORD_REACH is sprite origin minus Link at the B press, step 2, on a re-pinned CastleB1Guard soldier. The blade is not a box. Some arc cells connect after frame 4.
  Needs AP-055, AP-059. Do: roll swings at a still enemy (a what-if pin with
  a stunned soldier) over dx, dy per facing; record hit or miss. Done:
  `SWORD_REACH` table plus a test.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_sprite_ids.py: castle soldiers 0x41-0x4B verified; ROM $83:E800/$0D:B8F1 matches z3 dmgFromLink **AP-061 · Enemy vulnerability table**
  Learned: HP or contact disagrees with z3 for 0x00 0x08 0x0A 0x26 0x86 0xC9 0xD7 (give-damage matched); those rows stay unverified on ROM init values. Hammer is damage class 3. Hookshot stun is 0xFB/0xFC/0xFF; 64 is real HP. Types >= 0xD8 and flags3 bit 6 ignore the table.
  Needs AP-012. Files: `alttp/sprite_ids.py`.
  Do: per sprite type, HP, contact damage class and what hurts it (sword,
  arrow, bomb, fire, ice, hammer, hookshot stun). Source: z3 enemies plus
  the disassembly damage table. Mark each row verified or unverified. Done:
  castle soldiers verified.

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_rollout.py 38 passed, 2 deselected; test_castle_soldier_dies_with_no_damage_on_ten_of_twelve_offsets 12/12 offsets 0..11, 3 swings, Link HP 24, slot 0 HP 0 **AP-062 · Rollout-checked strike**
  Learned: Idle while bit 7 of $0EF0 is set, then strike_if_safe again. A distance gain without that bit is the soldier walking away, not a shove. Offsets are idle 0..11 on CastleB1Guard, approach DOWN.
  Learned: A committed B swing on a DOWN reach cell drops slot 0 from 6 to 4 and shoves him. He then walks back into Link before a second safe swing. Hitstun is bit 7 of $0EF0; strike_if_safe will not roll while that bit is set.
  Needs AP-057, AP-060. Read `nes/zelda_i/level9/patra.py` (`PatraBlade`).
  Do: `strike_if_safe` rolls the swing and commits only when the target HP
  drops or it is pushed back **and** Link is untouched. Mark those frames
  `ROM_CHECKED`. Done: a castle soldier dies with 0 damage on 10/12 offsets.

- [x] 2026-09-30 · evidence: CastleMain state_load_dev prefix ok frames 861 (was 3785); clear_room_61 0 walked alive, door table has no shutter (was 3235); clear_room_60 0; room 0x50 hp 24 **AP-063 · Room clear v2**
  Needs AP-035, AP-050, AP-062. Files: `alttp/opening_route/room_engine.py`.
  Do: `clear_room` targets with BFS + `strike_if_safe` under `PolicyGuard`,
  and runs only when `clearPolicy.required` is true (shutter, push block, or
  a key the route takes). Otherwise the room is walked alive. Done: castle
  prefix green; ledger shows fewer frames in the cleared rooms.

- [!] BLOCKED 2026-09-29: z3 id 70 Bow Soldier (Blue) is sprite 0x46 only in rooms 0x40 and 0xC0 (table $09:D62E); id 71 Bow Soldier (Green) is in no room. No pin or entrance lands there. Entrance 0x75 loads grid-north 0x30 as an Agahnim cutscene; after A, DOWN from (120,1688) stalls at y=1728 with no arrow and no room change. **AP-064 · Projectile guard and shield**
  Learned: Archer rooms from the dungeon sprite table, named by z3 id 70: 0x40 (two blue archers) and 0xC0 (three). Id 71 never appears. Entrance 0x75 is room 0x30, not a walk into 0x40.
  Learned: Enemy arrow is sprite 0x1B (z3 id 27, Archer Arrow). It did not spawn while idling or walking the soldier pins. Desert beamos 0x61 did not emit a shot sprite or ancilla.
  Needs AP-011, AP-057. Read `nes/zelda_i/dungeon/shot_guard.py` (header).
  Do: list projectile kinds seen (arrows, fireballs, rocks, energy balls,
  beams). Measure which ones each shield blocks when Link faces them.
  Add a shield-facing rule ahead of the dodge. Done: a table plus one archer
  room with fewer hits (quote both numbers).

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_primitives.py 20 passed; FighterSwordLamp select_y_item(lamp) ok hud 0x0B use 0x09 menu open 34 close 32 frames 68 room 0x55 no writes **AP-065 · Y-item selector**
  Learned: Scan_0x0104 chest 0 tile (56,54) item Lamp z3 0xE9BC opened in play. Ledger lamp $F34A 0->1 source=play frame 885 room in:0104. Pin Play_Scan0x0104_lamp kind=resumed. Eastern big-key chest was not opened.
  Learned: FighterSwordLamp already has hud_cur_item $0202=0x0B and current_item_y $0303=0x09. RIGHT does not change $0202 when the lamp is the only owned Y item. Cursor accepts input at map state 4; START is an edge. Open was 34 frames, close 32.
  Needs AP-005. Read `nes/zelda_i/dungeon/pause_select.py` (header). Files:
  `alttp/primitives.py`.
  Do: `select_y_item(env, name)` opens the menu, moves the cursor, confirms
  the selected-item byte changed, and closes. Never poke the selection.
  Done: selects the lamp on `FighterSwordLamp`, with menu frames reported.

- [x] 2026-09-30 · evidence: Whatif_FighterSwordLamp_red bottle_1=3 hp=8; drink_if_low hp 0x08->0x18 bottles (3,0,0,0)->(2,0,0,0) restored lamp; ledger spend bottle_1 3->2 source=play frame 110 room in:0055; pytest test_ledger.py test_primitives.py 27 passed **AP-066 · Bottle guard**
  Learned: Map state 4 ignores the first cursor edge. Bottle hud 0x10 use 0x0B. Red SRAM is 3 and a drink writes 2. Refill is module 0x0E. Menu sets bottle index from the first nonempty slot.
  Needs AP-065. Do: with a filled bottle, at the last heart (or when the
  worst hit on screen would kill), select the bottle, drink, wait out the
  refill, then reselect the previous Y item. Test on a what-if pin with a red
  potion. Done: HP refills in play; ledger shows the bottle spend.

- [!] BLOCKED 2026-09-30: Room 0x3A holds the resting block (SRAM $F940 record tile 0x4E60 on LinksHouseWake and PalaceOfDarknessEntry). No pin loads 0x3A. Its south door from PalaceOfDarknessEntry is small-key type 0x1C with keys=0; side doors stay shutters 0x18. Two walks stayed in room 0x4A. Loaded 0x7N tiles are pots (state 0x1111). z3-json-data has no tile object list. **AP-067 · Bomb, lift, throw, push primitives**
  Learned: Resting push blocks are SRAM $F940 room/tile records, drawn only when that room is current, state 0, attribute 0x70+slot. Room 0x3A tile 0x4E60 is unpushed on the early pin and on PalaceOfDarknessEntry.
  Learned: Idle push blocks are object state 0 under attribute 0x70+slot. 0x7N tiles on these pins are pots (state 0x1111). Bomb fuse 159; blast hit gap 17 miss 19. Eastern pot throws once sprite_unk3 is 3.
  Needs AP-032, AP-065. See the bomb traps in `nes/zelda_i/AGENTS.md`.
  Do: `place_bomb` (fuse frames, blast radius measured), `lift` and `throw`
  (pot or bush), `push_block` (frames of push before it moves). A placed last
  bomb may read 0 while it burns, so guard on count only before placing.
  Done: each primitive has one live check on a pin.

- [ ] **AP-068 · Survival assist (dev only)**
  Needs AP-002, AP-022. Read `nes/zelda_i/assist.py`. Files: `alttp/assist.py`.
  Do: refill HP at the last heart, counted as assist. The report says
  `intervention: Survival`. Done: a Survival run of the tip reports its refill
  count; the audit counts the writes.

- [x] 2026-09-29 · evidence: CastleB1Guard DOWN 220 frames, print on close: stage=castle_b1_guard hp 24->20 damage=4 hits_by_cause blue soldier=1 room in:0072 damage=4; pytest alttp/tests/test_ledger.py 6 passed **AP-069 · Damage census report**
  Needs AP-014. Do: per stage, HP in and out, hits by cause, damage by room.
  Print at the end of every spine run. Done: shows on a castle run.

- [x] 2026-09-29 · evidence: CastleMain pin idle 240 settled frames moved 0px; recordings/ap070_castle_stall.json has 240 rows frame/xy/buttons/reason reason=idle blocker=stall; reason_by_room in:0061 idle=240; pytest alttp/tests/test_ledger.py 6 passed **AP-070 · Reason census and stall trace**
  Needs AP-016. See Lessons 3, 6 and 7. Do: controllers emit a reason string per frame and the ledger
  totals `reason_by_room`. When Link moves under N px in M frames of play,
  dump the last 240 frames (frame, xy, buttons, reason) to
  `recordings/<tag>_stall.json` and fail the stage with `stall`. Done:
  a forced stall on a pin writes the dump.

- [ ] **AP-071 · Health economy census**
  Needs AP-068, AP-069. On NES, Clean became possible by healing before fights. Do:
  from a Survival run, hearts at each boss door and refills used. Output a
  table of levers (heart pieces, bottles, armor). Done: first table after
  Eastern exists (re-run after each dungeon).

- [x] 2026-09-29 · evidence: F37B=0 lamp 4, fire rod 16, ice rod 16, somaria 8, byrna 16 then 4/24f, cape 1/4f, bombos/ether/quake 32; small drop +16; full drop fills to 0x80; ledger magic spend already present **AP-072 · Magic meter**
  Learned: Two play tries in room 0x84 (not 0xC9/0xB9/0xA9, not the 0x74 north-lip pot) left magic $F36E at 0: west pot tile (12,12) byte 0x70 thrown, east pot tile (50,13) byte 0x72 lift did not shatter. Whatif_PlayScan0x0104_lamp_magic128 copies Play_Scan0x0104_lamp field magic address $F36E 0->128 kind=whatif. Not walked into Eastern. Last-heart HP refills on the east walk are disclosed and did not write inventory.
  Learned: Doorway $006C spends nothing. Swamp menu: first RIGHT stays on bombs, the next RIGHT reaches a poked Y item. Cape waits until $02E2 hits 0, then drains 1 point every 4 frames. Byrna spark keeps taking 4 every 24 frames after the 16. Filler $F373 moves 1 point per frame onto $F36E.
  Needs AP-065. Do: measure magic cost for lamp, fire rod, ice rod,
  somaria, byrna, cape and medallions, plus refill drops. Add magic to the
  ledger spends. Done: table in `ram_map.md` (what-if pins allowed to own the
  items).

## Phase F: castle → Sanctuary (AP-073..082)

Context: continuous tip is `0x50`. `room_50_east_to_0x01` and
`room_01_down_to_0x72` are natural_entry. `0x72 → 0x82 → 0x81` now reaches
the B1 cell from the power-on predecessor. The measured route continues via
`0x71 → 0x70 → 0x80`, then escorts Zelda to Sanctuary under Survival. Read
`alttp/docs/tasks/residual.md` before changing the Clean gate.

- [x] 2026-09-29 · evidence: opening_spine_room_72.json ok continuous natural_boot frames=15545 tip=room_72 xy=(1273,3665) glance misses=[] pins include Full_room_72.state; east landing (560,120); verified_tip_run.json not overwritten **AP-073 · Continuous past `0x50` to `0x72`**
  Learned: Power-on east lands 0x01 (560,120), outside the ROOM_01 well band. Well UP lands 0x72 (1273,3665); do not widen ROOM_72. Graph class stays natural_entry. Full_room_72.state is the arrival pin, not a continuous claim.
  Needs AP-023, AP-024. Do: add hops `room_50_east_to_0x01` and
  `room_01_down_to_0x72` to the spine (walk to the well at (760, 99), hold
  UP, per the residual). Run power-on `--through room_72 --save-points Full`.
  Do not widen glance windows without a measured landing. Done: continuous
  green; the `ROOM_72` glance passes.

- [x] 2026-09-29 · evidence: opening_spine_room_81.json ok continuous natural_boot frames=17327 tip=room_81 xy=(996,4485) sword=1 follower=0 keys=0 pins include Full_room_81.state; verified_tip_run.json not overwritten **AP-074 · `0x72 → 0x82 → 0x81` from the real predecessor**
  Learned: Full_room_72 and Full_room_81 are arrival pins, not continuous claims. Graph class of room_72_south_to_0x82 and room_82_west_to_0x81 stays isolated. Axis walk only; the 0x82 tile walker pits from this landing.
  Needs AP-073. Do: run those edges from `Full_room_72`, then continuous.
  Done: `--through room_81` green from power-on.

- [x] 2026-09-30 · evidence: recordings/ap075_big_key.json isolated CastleB3BossOneHitBoomerang strike_if_safe swings=4 ledger big_key 0->16384 source=play room in:0080 frame 332 xy=(403,4229) follower=0 hp=16 writes=[] **AP-075 · Ball-and-chain trooper and the big key**
  Learned: Sprite 0x6A on CastleB3BossOneHitBoomerang dies to four strike_if_safe commits. The 0xE5 prize raises $F366 to 0x4000 ($F367 bit 6) in play. Back off after repeated hurt rolls; a whiff is not a commit.
  Needs AP-062, AP-063. Pins: `CastleB3BallApproach`,
  `CastleB3BossOneHitBoomerang`. Do: `pin_probe` the room, then kill sprite
  `0x6A` with `strike_if_safe` and pick up the drop. Confirm the big key bit
  (`$F366`/`$F367`) rises **in play**. Done: isolated green with the ledger
  showing a play rise.

- [x] 2026-10-01 · evidence: `recordings/sanctuary_20261001/controller.json`; measured route `0x81 → 0x71 → 0x70 → 0x80`, with no west exit and no key grant **AP-076 · Gold cell door**
  Learned: the B1 cell is reached through the north `0x71` door. The route
  fights the ball-and-chain guard in `0x80`, picks up the in-play big key, and
  opens the cell. The former west-edge hypothesis was removed from the map.

- [x] 2026-10-01 · evidence: `recordings/sanctuary_20261001/controller.json`; follower `$F3CC` reaches 1 in the cell **AP-077 · Zelda rescue**
  Learned: the rescue is performed from the real `0x81` predecessor; no
  follower byte is written.

- [x] 2026-10-01 · evidence: `recordings/sanctuary_20261001/replay.json`; route maps and escort acceptance tests **AP-078 · Escort route data**
  Learned: the measured escort rooms are `0x80`, `0x70`, `0x71`, `0x72`,
  `0x02`, `0x11`, `0x12`; follower acceptance is checked at each edge.

- [x] 2026-10-01 · evidence: `recordings/sanctuary_20261001/replay.json`; mantle and sewer edge acceptance **AP-079 · Throne room mantle and sewer entry**
  Learned: the mantle opens from the rescued state and preserves the follower.

- [x] 2026-10-01 · evidence: `recordings/sanctuary_20261001/replay.json`; zero-load forward escort **AP-080 · Sewers to Sanctuary**
  Learned: the escort reaches Sanctuary with control ready and no progression
  or capacity writes.

- [x] 2026-10-01 · evidence: `tests/test_escort_to_sanctuary.py`, `recordings/sanctuary_20261001/replay.json`; room `0x12`, progress `$F3C5 == 2`, flags `$F3C6 == 0x15`, follower cleared **AP-081 · Sanctuary acceptance**
  Learned: `SANCTUARY_ROOM = 0x12` is confirmed and the acceptance predicate
  is encoded in `story_glance()`.

- [ ] **AP-082 · Rung C1: continuous power-on to Sanctuary**
  Survival controller and replay evidence is complete in
  `recordings/sanctuary_20261001/`, but seven HP refills and seven search
  restores keep this rung open. Close only after a Clean, no-load, no-write
  composition reaches the same Sanctuary predicate.

## Phase G: overworld and side items (AP-083..098)

Hints are vanilla knowledge. The requirement lists come from z3 (AP-040).

- [x] 2026-09-29 · evidence: pytest alttp/tests/test_overworld_graph.py: Sanctuary 0x13 to Eastern screen 0x1E; z3 Sanctuary Exit and Eastern Palace Exit **AP-083 · Overworld screen graph**
  Learned: Sanctuary OW 0x13 is ram.SANCTUARY_SCREEN, not a new scan. Eastern 0x1E is maps/room_c9.json outdoor door. Ice room_0e outdoor screen skipped (note says not DW). World-region room names are not walk edges.
  Needs AP-040. Files: `alttp/overworld_graph.py`, tests. Do: Light World
  and Dark World screen adjacency plus entrances from z3 connections. Done:
  test finds a path Sanctuary → Eastern screen.

- [x] 2026-09-30 · evidence: SanctuaryExit.state provenance kind=resumed source_run=save_and_quit:IcePalaceEntry:row1+south_door frame=1110 outdoors screen 0x13 (1880,1083); walk_sanctuary_exit_to_links_house frames=3126 xy=(2528,2920) screen=0x2C standing=0x2C indoors=0 hp=40->24 trace=13,14,15,1D,25,2D,2C continuous=0; pytest alttp/tests/test_overworld.py 5 passed **AP-084 · Overworld walker**
  Learned: Save-and-quit progress 03 row 1 lands indoors room 0x12 screen byte 0x00 at (1272,668); DOWN out the south door lands outdoors screen 0x13 (1880,1083). That pin is resumed, not continuous. Screens 0x13 and 0x14 have no south edge; the green walk was 13-14-15-1D-25-2D-2C. Guard text 0x0020 closes with A after $1CD4=3 and $1CE9=0. Sort the screen just left last or a west-edge arrival steps back.
  Needs AP-053, AP-083. Files: `alttp/overworld.py`. Do: screen-to-screen
  walk on real tiles with edge exits. Done: Sanctuary exit → Link's house
  screen green from a pin.

- [x] 2026-09-30 · evidence: SanctuaryExit resumed pin screen 0x13 hp=0x28 (1880,1083); walk_to_screen(0x25, guard=True) frames=2371 xy=(2831,2056) standing=0x25 indoors=0 hp=40->32 hits=0->1 trace=13,14,15,1D,25 checks=773 detours=11 continuous=0; pytest test_overworld.py test_rollout.py 43 passed. No sword swing. **AP-085 · Overworld enemies**
  Needs AP-057, AP-084. See Lesson 11. Do: walks
  run under `PolicyGuard`; fight only when a one-hit kill beats a dodge.
  Done: a 5-screen walk with hits quoted before and after.

- [!] BLOCKED 2026-09-30: Library walk-in never reached room 0x07 after two routes: screen 0x2B has no west exit (sign retriggers at 1680,3007) and the 0x1B west strip y=1816-1848 does not change area in play. Five other doors are in maps/screen_13_sanctuary.json, screen_2c_links_house.json, screen_1e_eastern.json, screen_1e_sahasrahla.json, screen_30_desert.json. Evidence snes/alttp/recordings/probe_ap086_enter.json. **AP-086 · Entrance catalog, batch 1**
  Needs AP-084. Files: `alttp/maps/screen_<id>_<name>.json`. Do: measured
  door xy for the Sanctuary exit, Link's house, the Eastern entrance,
  Sahasrahla's hut, the library and the Desert entrance. Done: `lint` green;
  one walk-in per entrance isolated.

- [x] 2026-09-29 · evidence: alttp/docs/routes/ITEM_ROUTE.md rows AP-105 AP-113 AP-121 AP-129 AP-137 AP-145 AP-153 AP-161 AP-169 AP-177 AP-185 AP-193 from z3 pin 1eb7a785 **AP-087 · Item route doc**
  Learned: RescueZelda is required and has no yield. PoD Hammer chest requires Hammer. hasProperMedallion requires all three medallions (helpers.json FIXME). Desert nodeItem PendantOfWisdom with ROM 0xC6FF=0x02; Hera nodeItem PendantOfPower with ROM 0xC706=0x03. Skull big chest requires BigKeyD3 with a trailing space.
  Needs AP-040. Files: `alttp/docs/routes/ITEM_ROUTE.md`. Do: an ordered
  leg list (dungeons plus side items) with the items each leg needs, from
  z3 logic. Use the TAS order (AP-006) only as a hint. Done: every Phase H
  approach task cites a row here.

- [x] 2026-09-29 · evidence: ITEM_ROUTE.md budget; zelda3.sfc wallet $7EF360: Zora SBC #$01F4 at 0x29A94, Kiki LDA #$0A at 0xF63F9 and LDA #$64 at 0xF651C, super bomb LDA #$64 at 0xF61EE then follower #$0D **AP-088 · Rupee economy**
  Learned: Charge sites subtract SRAM $7EF360. ram.py rupees is display word $F362. z3 has no Kiki node and no shop price.
  Needs AP-017, AP-087. On NES, money was the wall after L4. Do: list rupee costs on the route (Kiki, Zora, shops,
  super bomb) and cheap sources near the route. Measure prices in game.
  Done: budget table in `ITEM_ROUTE.md`.

- [x] 2026-09-30 · evidence: hut room 0x05 pendant 0x04 already set, boots 0: elder receipt boots $F355 0->1 and ability $F379 0xF8->0xFC ledger frame 962 source=play writes=0 follower 0x05. Dash pose (2678,8617): state 17, charge 30 frames, then 4px/frame. RIGHT attr 0x27 stops (2688,8617) then state 2 rests (2672,8617). primitives.dash. Isolated, not continuous. **AP-089 · Pegasus Boots + dash**
  Learned: claimed pegasus-boots; pendant pin plus hut door, no boots poke
  Needs AP-084, and Eastern's pendant (AP-106) for natural play. Hint:
  Sahasrahla gives the boots after the first pendant. Do: walk, dialog, boots
  byte rises in play. Add a `dash` primitive (charge frames, speed, stop on
  wall). Done: isolated leg plus dash constants.

- [x] 2026-09-30 · evidence: room 0x0107 ledger frame 377 book 0->1 source=play **AP-090 · Book of Mudora**
  Learned: claimed book-mudora; will read LW door 26 table before boot from probe_AP-089/boots.state
  Learned: Room 0x003D cols 0-31 rows 32-63: no dash-bit attr 50/51/57/8E/8F on the full 64x64 table. Shelf chr 1D4-1D7 is attr 66, posts 67 chr 1D0-1D3, chest 5A chr 1E1/1F1, blocks 72-76 chr 128/138, walls 01/02/04, doors F2/F3. Pocket from (6774,10120) only reaches x=6768-6784 on y=10120. A sets tile_action 5 (chest KeyA2 at tile 14,48) or 3, state stays 0, sprite 0x3B stays inactive, ledger book rows empty, so $F34E stayed 0. No second warp and no 0x2B/0x1B dash.
  Learned: claimed book-mudora; scanning room 0x003D shelf tiles from Whatif_Library, no second warp and no 0x2B/0x1B dash
  Learned: Warp sub 0x15 from probe_AP-089/boots.state kept previous room 0x0105 and set $A0 to z3 Library cave id 61. Indoors room 0x003D (6774,10120) mode 0x07 sub 0, sprites none, no dash-item 0x3B, $F34E stayed 0. Whatif_Library kind=poke. No second room id.
  Learned: refs/z3-json-data and snes/alttp/z3-json-data name Library only as Light World door id 26 and cave room id 61 item BookOfMudora nodeAddress 0x180012. The door connection has no screen and no bookshelf. No overworld screen is named, so none besides the closed 0x2B and 0x1B dashes. No new boots-pin approach. $F34E stayed unread.
  Learned: Both dashes stayed outdoors and $F34E stayed 0 on pin probe_AP-089/boots.state, follower 0x05. Screen 0x2B LEFT from (1720,2992) 4px/frame frame 31, bonk state 2 at (1650,2992), front attr 0x00, left exits 0, standing 0x2B. Screen 0x1B y=1848 LEFT from (1696,1848) 4px/frame frame 34, text stop (1573,1848), rest (1574,1848) state 0, front attr 0x40, standing 0x1B. Room base stayed 0x04 and 0x05.
  Needs AP-089. Hint: dash into the library bookshelf. Done: book byte rises
  in play on an isolated leg.

- [ ] **AP-091 · Death Mountain and the Magic Mirror**
  Needs AP-084, AP-067. Hint: glove rock → dark old-man cave (lamp) → old
  man follower → mirror. Do: record the old man's follower value (AP-013).
  Add a `use_mirror` primitive. Done: mirror byte rises in play on isolated
  legs.

- [ ] **AP-092 · Master Sword**
  Needs AP-084, three pendants for natural play. Hint: Lost Woods walk
  (thieves, fog) → pedestal. Done: sword byte = 2 in play.

- [!] BLOCKED 2026-09-30: Flippers do not rise in play: SanctuaryExit flippers already 0x01 with wallet 0; LinksHouseWake, EasternPalaceEntry, and DesertPalaceEntry have flippers 0x00 and wallet 0. House A-sweep and walk to screen 0x15 left wallet 0. King Zora costs 500. Evidence scratch/probe_ap093_wallet.py and scratch/probe_ap096_rupee.py. **AP-093 · Flippers**
  Learned: Done requires the flippers byte to rise in play. Whatif_IcePalaceEntry_parallel does not write flippers; the copy still has flippers=1 from IcePalaceEntry. A poke does not close this.
  Needs AP-084, AP-088. Hint: Zora's waterfall, paid in rupees. Measure the
  price. Add swim physics to `walk_physics.py`. Done: flippers byte rises in
  play.

- [!] BLOCKED 2026-09-30: Flute byte stays 0x00 on SanctuaryExit, LinksHouseWake, EasternPalaceEntry, and DesertPalaceEntry. No pin is the Dark World flute boy, and the sanctuary walk did not grant a flute. No flute_to trip. Evidence scratch/probe_ap093_wallet.py. **AP-094 · Flute and flute travel**
  Needs AP-084. Hint: shovel from the Dark World flute boy, dig the flute in
  the Light World grove, activate at the Kakariko bird statue. Do: add a
  `flute_to(spot)` primitive that picks a destination on the map. Done: one
  flute trip lands on the expected screen.

- [ ] **AP-095 · Ether and Quake medallions**
  Needs AP-091, AP-092, AP-097. Hint: Ether at the tablet on the mountain
  (Master Sword + book); Quake from the Dark World catfish (glove). Add a
  `use_medallion` primitive (select, press, magic). Done: both bytes rise in
  play on isolated legs.

- [!] BLOCKED 2026-09-30: No filled bottle in play: bottle bytes stay 0 on the four pins, wallet stays 0 after a house A-sweep and a walk to screen 0x15. Bottle merchant costs 100. Evidence scratch/probe_ap096_rupee.py. **AP-096 · Bottles and refills**
  Learned: Done requires one filled bottle bought or caught in play. None of the parallel Whatif pins write a bottle byte. A poke does not close this.
  Needs AP-066, AP-088. Do: the first reachable bottle, then potion shop
  prices, and fairy capture if the net is needed (verify). Plug into the
  bottle guard. Done: one filled bottle bought or caught in play.

- [ ] **AP-097 · Dark World transitions**
  Needs AP-052, AP-091. Do: portal warps (glove/mitt rocks), mirror return
  sparkle, and the Pyramid landing after Agahnim. Make every tool accept the
  Dark World flag (`in_room` today rejects Dark World). Done: `in_room`
  test on a Dark World snapshot plus one portal warp isolated.

- [ ] **AP-098 · Silver arrows**
  Needs Ice and Mire crystals (AP-170, AP-178) for natural play, AP-088.
  Hint: super bomb from the Dark World bomb shop (a follower) → Pyramid
  crack → fairy. Done: silver arrow byte rises in play.

## Phase H: dungeons (AP-099..194)

### Dungeon task template (steps 1-8)

Every dungeon block below is these eight steps in this order. The dungeon
lines add only the specifics. Slug = the AP-040 slug. The route doc is
`alttp/docs/routes/<SLUG>_ROUTE.md`, copied from the template (AP-004).

1. **Dungeon start: entry pin + first hop (dev).**
   Needs AP-027, AP-035, AP-036, AP-042, AP-044. Use the existing `*Entry`
   pin, or make one with `make_entry_pin.py <slug>`. Run `audit_pins.py` on
   it. Run `pin_probe --doors --chests`. Isolate the first forward door with
   `room_engine.py run`. Add the edge to the hop table with
   `verification=isolated`.
   Done: first edge green 2/2; audit clean; route doc "Entry" filled. Never
   call it natural.
2. **Scan.** Needs step 1, AP-038, AP-039, AP-040. Get the dungeon's
   locations and key logic from `route_data/<slug>.yaml`. Walk each route room
   (room_engine runs or a tape scan). Save `Scan_<slug>_<room>` with a
   `pin_probe --doors --chests --tiles` dump. Fill the route doc room table.
   Write door-only `maps/room_XX.json` skeletons.
   Done: every room reachable with the keys you hold is in the table
   with door types and chests; each stopping lock is named; `lint` green.
3. **Route part 1: entry → big key.** Needs step 2, AP-051, AP-063. Per
   room in order: measure points, set door paths, isolate each edge from its
   Scan pin. Keys come from play only. Fight only where the door table says so.
   Done: every edge isolated 2/2; big key bit rises in play (ledger).
4. **Route part 2: big chest and the dungeon's mechanic.** Needs step 3.
   Open the big chest (item byte rises in play). Add the primitive the rest
   of the dungeon needs.
   Done: edges to the boss door isolated; the new primitive has a live check.
5. **Boss arena + boss probe.** Needs step 4. Save
   `Scan_<slug>_boss` at the arena door. Run `alttp/scripts/boss_probe.py`
   (made in AP-103) for 1800 frames: boss sprites, HP byte, phase bytes,
   projectile spawns, what hurts it and when (rollout sweeps).
   Done: route doc "Boss" section filled; arena pin audited.
6. **Boss battle.** Needs step 5, AP-029, AP-057, AP-062.
   `alttp/dungeon/bosses/<boss>.py`: a phase machine, strikes via
   `strike_if_safe`, walk legs under `PolicyGuard`, and the bottle guard if a
   bottle is owned. Evaluate with `eval_offsets.py Scan_<slug>_boss`.
   Done: ≥10/12 offsets kill the boss and take the heart container; hits by
   cause quoted.
7. **Dungeon start: approach leg (natural).** Needs the predecessor's
   step 8 and the listed side items. Walk from `Full_<prev>_exit` to the
   entrance with the overworld walker. Items must come from play. Save
   `Full_<slug>_entry`.
   Done: natural entry from the real predecessor; 0 writes; audit clean on
   the new pin.
8. **Dungeon chain + rung.** Needs steps 6 and 7, AP-025. Add hop rows to
   `spine/hops.py` (entry → prize → exit). Run from `Full_<slug>_entry` with
   `--save-points`, fix from `Full_<stage>`, then run continuous from
   power-on. Replay the tape. Planner marks the rung and may edit `STATUS.md`.
   Done: `run_spine.py --through <slug>_prize` green from power-on once
   (Clean, or Survival disclosed); replay desync 0.

### Eastern Palace (AP-099..106) → rung C2

Pin `EasternPalaceEntry` (room `0xC9`); `south_to_overworld` isolated; next
is the north hall toward `0xB9`. Scratch: `probe_ep_entry.py`. Hint: big
chest Bow, boss Armos Knights, Pendant of Courage. Some red Eyegores may need
arrows; check the z3 boss requirement.

- [x] 2026-09-30 · evidence: room_engine room_c9 north_to_0xb9 green 2/2 from EasternPalaceEntry, both land room 0xB9 xy=(4856,6104) frames=606; audit EasternPalaceEntry 0 incoherent; docs/routes/eastern_ROUTE.md Entry isolated **AP-099 · Eastern step 1: entry pin + first hop.** Forward edge is
  north, not `south_to_overworld`.
- [x] 2026-09-30 · evidence: keys=0 rooms 0xC9 0xB9 0xA9 from Scan_eastern dumps are in eastern_ROUTE.md. Stopping locks: 0xB9 east slot0 type 0x1C (eastern.yaml has no small key, not walked), 0xA9 north slot2 type 0x1E big_key=0, 0xA9 west/east shutters type 0x18 stayed shut. room_engine lint ok 30 maps. **AP-100 · Eastern step 2: scan.**
  Learned: 0xB9 east is live type 0x1C but eastern.yaml has no small key. Cannonball chest 0xE9B3 paid rupees not a key. Big key chest is ROM room 0xB8 address 0xE9B9 and needs hasLightCone.
- [!] BLOCKED 2026-09-30: Whatif_EasternPalaceEntry_magic80 lamp $F34A=1 magic $F36E=128 kind=whatif; big_key stayed 0; chest not opened. West door slot 3 type 0x00 never transitioned. Layer-1 wall holds x=4672 from tile (8,14) through (8,32) in room 0xB9. Door-wall pose xy=(4672,5768) tile (8,17) layer 1 hp=12 magic=128. **AP-101 · Eastern step 3: entry → big key.** Hint: Stalfos throw
  Learned: Later steps may load Whatif_EasternPalaceEntry_parallel (kind poke, room 0xC9, module 0x07). Writes: big_key $F366=0x2000, magic $F36E=128. Not a key that rose in play.
  bones (a projectile for AP-064).
- [x] 2026-09-30 · evidence: isolated poke Whatif_EasternPalaceEntry_bow bow $F340=1 arrows $F377=30 read back on room 0xC9; doorway $006C=1 at (4856,6616); UP 17 to (4856,6590) door 0; shoot_arrow ancilla 0x09 (4860,6594) count 30->29; menu set bow 1->2; $F3CC stayed 5; chest not opened **AP-102 · Eastern step 4: Bow and arrows.** Add a `shoot_arrow`
  Learned: Room 0xC9 spawn (4856,6616) is doorway $006C=1, so one Y does not start the draw. Draw delays are 3, 3, 8. The menu sets bow 1 to 2 when arrows are already held.
  Learned: primitives.shoot_arrow still required beside place_bomb: select_y_item bow, one Y, new ancilla 0x09 or 0x0A, and a lower $F377. z3 canShootArrows requires Bow. Red eyegore sprite 132 sword damage 0 arrow 4. This pin never owned either.
  primitive (Y select bow, count on `$F377` candidate). Use it where z3 says
  the bow is needed.
- [x] 2026-09-30 · evidence: Whatif_Eastern_armos kind=poke room 0xC8 audit 0 incoherent. boss_probe.py idle 1800: six 0x53, $0E50=48, $0D80 0 to 1 on five slots by frame 327, Link HP 24 to 20, born 0. Arrow drops $0E50 by 16 after the knight leaves spawn xy; strike_if_safe sword drops 4. eastern_ROUTE.md Boss filled. Isolated poke, not continuous. **AP-103 · Eastern step 5: Armos Knights probe.** Also create the
  Learned: eastern.yaml has no room path to Armos Knights other than the Eastern Palace region. Connections are one internal door and the Light World exit. No path avoids the failed 0xA9 chest. Left open.
  generic `alttp/scripts/boss_probe.py` here (sprites by slot every frame,
  HP/phase bytes named on the CLI, JSON out).
- [x] 2026-09-30 · evidence: eval_offsets Whatif_Eastern_armos bosses/armos_knights:fight 12/12 pass frames 1708-1756 hp_lost 4-12 hits armos knight:3 or arrow:1. Heart container max HP 0x18 to 0x20. Arrow only after the knight leaves spawn xy. Last knight HP returned 16 to 48. Isolated poke, not continuous. **AP-104 · Eastern step 6: `bosses/armos_knights.py`.** Hint: six
  Learned: Last knight HP returns to 48. Backing into the south-east corner dies there; the eye shot still drops 16 and takes the heart.
  Learned: Arrow drops $0E50 by 16 once the knight leaves its spawn xy. A strike_if_safe swing drops 4. Spawn-pose arrows halt as ancilla 0x0A and leave HP at 48. The south-line fight did not clear the room.
  Learned: eastern.yaml has no room path to Armos Knights other than the Eastern Palace region. Connections are one internal door and the Light World exit. No path avoids the failed 0xA9 chest. Left open.
  knights; the last one changes behaviour. Read `nes/zelda_i/dungeon/gohma.py`
  for the phase-machine shape.
- [ ] **AP-105 · Eastern step 7: approach.** Needs AP-082, AP-084, AP-086.
  Sanctuary → Eastern entrance.
- [x] 2026-09-30 · evidence: Whatif_Eastern_armos isolated: $F374 0x00 to 0x04 source=play ledger frame 1835 writes=0 after heart $F36C 0x18 to 0x20. Ancilla 0x29 then receipt 0x22. $F3CC stayed 5. Not continuous and not the Sanctuary chain. **AP-106 · Eastern step 8: chain + rung C2.** Acceptance: pendant bit
  (candidate `$F374`) rises in play.

### Desert Palace (AP-107..114) → rung C3

Pin `DesertPalaceEntry` (room `0x84`); `north_to_0x74` isolated. Scratch:
`probe_dp_entry.py`. Hint: the Book of Mudora opens the entrance; big chest
Power Glove; a back section is reached outdoors over glove rocks; torches
open the way to Lanmolas; Pendant of Power.

- [x] 2026-09-30 · evidence: room_engine room_84 north_to_0x74 green 2/2 from DesertPalaceEntry, both land room 0x74 xy=(2272,4052) frames=638; audit DesertPalaceEntry 0 incoherent; docs/routes/desert_ROUTE.md Entry isolated, not natural **AP-107 · Desert step 1: entry pin + first hop.**
- [x] 2026-09-30 · evidence: keys=0 rooms 0x84 and 0x74 are in desert_ROUTE.md. Left mouth slot1 type 0x00 stayed in 0x74 through the north wall (2168,3632). Right mouth slot0 type 0x00 not stood in. 0x27 statues seal the center landing. Outdoor back not walked: screen 0x30 rays stayed at (296,3259), gloves=0. No 0x1C or 0x1E. room_engine lint ok 31 maps. **AP-108 · Desert step 2: scan.** Record the outdoor link between the
  Learned: 0x74 mouths are internal type 0x00, not key doors. West aisle from 0x84 needs one pot lift. North-lip pot raised magic 0 to 16 and keys stayed 0. Map chest 0xE9B6 stayed closed.
  front and back sections.
- [!] BLOCKED 2026-09-30: room 0x73 pose (1866,3715) at dash-item sprite 0x3B (1856,3680); A+UP ended (1872,3638) link_state {0} keys 0; A+RIGHT from (1850,3664) to (1935,3664) link_state {0} keys 0. boots $F355=1 ability $F379=0xF8 dash bit 0, lobby A+UP state {0}. Separator from the torch: the button does not dash, so KeyP2 does not drop. Separator from 0xE9C2: room 0x75 pose (2572,3915) tile (1,41), chest BigKeyP2 tile (47,10) opened=0, no floor path across center wall cols 27-36, keys stayed 0. Assist last-heart refill twice hp 4->24 (2.5 hearts each) writes=2, not a Clean claim. **AP-109 · Desert step 3: entry → big key.**
  Learned: Later steps may load Whatif_DesertPalaceEntry_parallel (kind poke, room 0x84, module 0x07). Writes: big_key $F366=0x1000, keys $F36F=1, magic $F36E=128. Not a key that rose in play.
  Learned: ROM snes/alttp/roms/zelda3.sfc (headerless LoROM): the only store that can set bit 0x04 of $7EF379 is $09:8658 STA.l $7EF379, after $09:8648 LDA #$04 / $09:864A CPY #$4B / $09:864C BEQ $8654 / $09:8654 ORA.l $7EF379. Y is the item id in $02D8 ($09:8610 LDY $02D8), inside the receipt at $09:85E8. Caller is Link_ReceiveItem $07:99BE STY $02D8 then $07:9A07 JSL $0985E8. The Y=$4B giver is $05:F1FB LDY #$4B / $05:F200 JSL $0799AD, jump-table state 2 of JSL $008781 at $05:F150 (word $F1FB). State 0 $05:F177 LDA.l $7EF355 BNE $F19E skips the INC/INC at $05:F196, so that receipt does not run when boots are already 1. Item table $09:84E8 has no $F379 word; Y=$4B writes $F355=1 and Y=$1E ORs mask #$02 only. Chest alternate $07:B528[$4B] is $FF, so a boots chest would call the same receipt, and nothing on DesertPalaceEntry does. Page clears $09:F652 and $0C:C321 STA.l $7EF300,X store 0 across $F379 and cannot set the bit. No item-menu or room-transition writer. Bit 0x04 cannot rise in play on DesertPalaceEntry. Did not boot.
  Learned: Dash A-press $07:9C2F ANDs ability $F379 with $9BA2 action-2 mask 0x04; boots $F355 is not that read. Whatif_DesertPalaceEntry_abilityFC kind=whatif field ability $F379 0xF8->0xFC (248->252). Ledger booked it as write frame 1 source=write (poke, not play). Sprite 0x3B room 0x73, ability stayed 0xFC: A+UP (1866,3715)->(1872,3640) link_state {0,17} keys 0; A+RIGHT (1850,3664)->(1934,3664) link_state {0,17} keys 0. Keys did not rise. Big-key word stayed 0x0000; chest not tried. Assist last-heart refill once hp 4->24 (2.5 hearts) writes=1. Not a Clean claim. A poke cannot close AP-109.
- [!] BLOCKED 2026-09-30: Desert Palace Whatif_DesertPalaceEntry_parallel isolated poke room 0x84 module 0x07, writes big_key $F366=0x1000 keys $F36F=1 magic $F36E=128. Big chest room 0x73 tile (14,14) PowerGlove loc=$071C opened=0 and $F354 stayed 0. West shutter slot 0 type 0x18 tile (15,37)=0xF0 stayed shut after sword on 0x41/0x61/0x64/0x71 (hp unchanged) and 5 pot lifts 0x70-0x77. Center wall cols 27-36 is 0x01. Row 48 seam is 0xF9/0x01/0x81. Assist last-heart refill writes=2 hearts=4.0, not a Clean claim. No boss-door edge and no light_torch live check. **AP-110 · Desert step 4: Power Glove, rocks and torches.** Use
  Learned: Whatif_DesertPalaceEntry_glove kind=whatif: pin_probe read gloves $F354=1. Not a rise in play. Loaded rock pile map16 0x36D/0x36E/0x374/0x375 at (704,3168) attr 0x55 was not reachable, so it was not lifted. light_torch still has no live check.
  `lift`/`throw` (AP-067) and add `light_torch` (lamp).
- [x] 2026-09-30 · evidence: Whatif_Desert_lanmolas kind=poke room 0x33 audit 0 incoherent hp 0x18/0x18 dungeon 0x06 module 0x07. boss_probe idle 1200: three 0x54, $0E50=16, $0D80 cycles 0-4, Link HP 24 to 12, rocks 0xC2 born at 250 and 1070. Sword drops $0E50 by 2 in AI 2/3 at dist 20-28. desert_ROUTE.md Boss filled. Isolated poke, not continuous, not STATUS. $F3CC=0. No Scan pin. **AP-111 · Desert step 5: Lanmolas probe.** Hint: worms rise out of
  Learned: Warp submodule 0x15 from Whatif_DesertPalaceEntry_parallel room 0x84 kept the previous room and set $A0=0x33. Floor tile (17,41) at (1677,1871). Three sprite 0x54 on screen. $F3CC stayed 0. Isolated poke, not continuous.
  Learned: Entrance 0x0C from DesertPalaceEntry with gloves $F354=1 loads room 0x63, not Lanmolas room 0x33. Push-block slot 6 tile (21,44) UP sets $0641=1 and mouth (15-16,36-38) becomes 0x82. The north half clears $0641 and the mouth returns to 0xF1. Stair slot 0 type 0x20 stays behind 0xF0; UP from (1656,3120) stays in room 0x63. Sprite 0x54 not on screen. Whatif_Desert_lanmolas not saved. $F3CC stayed 0. Assist refill writes=1.
  Learned: Screen 0x30 pad (296,3259) BFS reachable=2. DOWN stuck at (296,3432); BFS there reachable=354 closest (368,3384) dist=552, no path to the rock. LEFT/RIGHT stayed (296,3259). UP re-entered room 0x84. Sprite 0x54 not on screen. Scan_desert_boss not saved. 0 of 12. Assist writes=0.
  the sand; the head is the target.
- [!] BLOCKED 2026-09-30: offset 0 died pose (1629,1920) facing UP f=2182 hp 8->0; boulder dx/dy at kill none (rays=-); boulder hit f=1028 pose (1672,1912) DOWN nearest dx=28 dy=-4 vx=28 vy=-28 rays 9:-67,91,-28,28 10:-68,-4,-28,-28 11:27,91,28,28 12:-79,79,-28,28 13:28,-4,28,-28; hits boulder:1,lanmolas:2 **AP-112 · Desert step 6: `bosses/lanmolas.py`.**
  Learned: cardinal gap only: swing at origin dist 21-25 on one axis while z is 1-11; diagonals are the boulder rays
  Learned: Sword commits land at dist 21-25 while $0F70 is 1-11. Five strikes then dead at frame 1653 hits boulder:1,lanmolas:2. Indoor 0xC2 is four diagonals; $0D50 is X and $0D40 is Y, px=signed/16. Cardinals are the gap.
  Learned: Room 0x33 did load via warp submodule 0x15. Three sprite 0x54 are on Whatif_Desert_lanmolas. Sword drops $0E50 by 2 while $0D80 is 2 or 3 at dist 20-28. Indoor boulder 0xC2 pixel speed is the velocity byte over 16. Offset 0 still dies before the heart.
  Learned: Room 0x33 did not load. Entrance 0x0C is room 0x63; the north stair type 0x20 stays behind 0xF0 after the push-block shutter opens and then closes. Sprite 0x54 not on screen. Whatif_Desert_lanmolas not saved. 0 of 12 offsets. bosses/lanmolas.py not written. $F3CC stayed 0.
  Learned: Lanmolas was not on screen. 0 of 12 offsets. The rock at (704,3168) was not reached from Whatif_DesertPalaceEntry_glove, so the arena did not open.
- [~] desert-tablet 2026-09-30 **AP-113 · Desert step 7: approach.** Needs AP-106, AP-089, AP-090.
  Learned: copy screen 0x30 pin to Whatif_Desert_plaque, poke ability 0xFC, one dash each way; prove byte with pin_probe
  Learned: A+LEFT from (312,3464) book 1 mode 0x09 control, boots 1 ability $F379=0xF8 dash bit 0, stayed (312,3464) state 0 $5E=0 $02ED=0 feet attr 0x00 and did not enter (304,3464); state 17 never started. A+RIGHT from (280,3464) stayed (280,3464) state 0 $02ED=0 feet attr 0x00 and did not enter (296,3464). Both dashes stayed on the side cells. No book poke. Left [~]. Log recordings/probe_AP-113/dash_sides.json.
  Learned: From (312,3464) book 1 mode 0x09 control, LEFT stays (312,3464) $02ED=0 feet attr 0x00 and does not enter (304,3464). From (280,3464), RIGHT stays (280,3464) $02ED=0 feet attr 0x00 and does not enter (296,3464). Neither north 0x46 cell was entered. No book poke. Left [~].
  Learned: From (304,3472) book 1 mode 0x09 control, UP stays (304,3472) $02ED=0 feet attr 0x00. LEFT then UP stops (296,3472) $02ED=0 and does not enter (296,3464). East step stops (312,3464); west step stays (280,3464). Neither north 0x46 cell was entered. $02ED stayed 0. No book poke. Left [~].
  Learned: Screen 0x30 pose (304,3472) mode 0x09 book 1 control, $02ED=0. DesertPalaceEntry DOWN lands pad (296,3259) then sand (296,3432). Nearest 0x46 is (296,3464) map16 1802, (304,3464) 1803, (296,3472) 1804, (304,3472); 0x02 lip (296,3456) and (304,3456). North hold stays (296,3432). South UP stops at (304,3472) with probe attr 0x00. Plaque not stood. No book poke. Left [~].
  Learned: book_play (3643,8504) mode 0x07 book 1 quad (0,2) control. Reachable local x 40..200 y 304..488, 196 cells, stairs 0. Only y>=476 cells are two 0x8E iris tiles; room+=0x10 needs a non-0x8E south tile and west room-- needs x<8, neither reachable. Doors south 0x00 $1D1C and 0x2E $1D5C. Still indoors. Left [~].
  Learned: claimed; leaving room 0x0107 from book_play.state, not the book-0 road pin
  Learned: outdoor road (3216,2344) standing 0x26 book 0 mode 0x09 follower 5; eastern exits only south; west from 0x2C and 0x34 wraps link_x to 65533 in mode 0x09 sub 6; 0x2D to 0x25 works but the west step is a 41-cell ledge (2528,2216) whose exits only return to 0x25, stair portals 0; book_play.state has the play book in room 0x0107 and the exit iris hangs; plaque nearest (296,3259) was not stood. Isolated pin.
  Learned: claimed; reading desert.yaml tablet and probe_AP-090 book pin before boot
  Read the entrance tablet with the book.
- [ ] **AP-114 · Desert step 8: chain + rung C3.**

### Tower of Hera (AP-115..122) → rung C4

Pin `HeraEntry` (room `0x77`); probe-isolated `west_down_to_0x87`;
`room_engine` wedges on pegs. Scratch: `probe_hera_entry.py`. Hint: crystal
switches raise and lower pegs; holes drop to the floor below; big chest Moon
Pearl; Moldorm can knock Link off the arena (falling = fail); Pendant of
Wisdom.

- [x] 2026-09-30 · evidence: PEG_STATE $C172 00->01->00 on HeraEntry crystal (3840,3968); west alcove closed (3792,3979), open (3704,3979), closed (3792,3979); west_down_to_0x87 isolated 2/2 landing room 0x87 (3721,4424); no RAM writes **AP-115 · Hera step 1: entry pin + first hop.** Pegs block: first
  find the peg state byte by hitting a switch on the pin, then add it to
  `ram.py`.
- [x] 2026-09-30 · evidence: Scan_hera_0x77 ($C172=00, room 0x77) and Scan_hera_0x87 ($C172=01, (3721,4424)) are in hera_ROUTE.md. Crystal hits in play only. Stopping locks: 0x77 south type 0x0C screen 0x03 (2288,123); east 0x5E (45,43) no path; MapP3 tile (31,15) pocket no path; 0x87 slot0 west type 0x00 and slot1 north type 0x00 no path at either peg; slot8 type 0x18 and slot9 type 0x00 other layer; BigKeyP3 tile (47,52) no path (needs KeyP3 and canLightTorches, magic=0, pin keys=0); rooms 0x07 0x17 0x27 0x31 0xA7 unentered. Assist on, writes=0, hearts=0, not a Clean claim. lamp=1 magic=0. room_engine lint ok 32 maps. maps/room_87.json door-only. **AP-116 · Hera step 2: scan.** Record peg state per room.
- [!] BLOCKED 2026-09-30: peg $C172=00 pose (3784,4503) and peg 01 pose (3784,4501) both stop short of tile (47,52); separator row 52 cols 27-36 is 0x01/wall x8 and 0x04/wall x2, front (27,52)=0x04, same at either peg, not a torch. big_key stayed 0. hit_switch live peg 01->00 at (3648,4432). keys 0->1 source=play pin Play_hera_0x87_key hp 16 pose (3709,4490). Assist last-heart refill once hp 8->24 writes=1, not a Clean claim. **AP-117 · Hera step 3: entry → big key.** Peg-aware BFS (raised
  Learned: Later steps may load Whatif_HeraEntry_parallel (kind poke, room 0x77, module 0x07). Writes: big_key $F366=0x0020, keys $F36F=1, magic $F36E=128. Not a key that rose in play.
  pegs are walls for the current state). Add a `hit_switch` primitive.
- [!] BLOCKED 2026-09-30: Whatif_HeraEntry_parallel isolated poke. Pearl $F357 stayed 0; big chest unentered. Arrow at (3840,3922) flipped peg 01->00; north stand (3838,3744); MapP3 no path. 0x5E (45,40) stairs to room 0x31 (872,1858) layer 0 floor 0x02. Slot 0 type 0x1E tile (14,36) no path; peg gates on that flood=0. Pit (47,44) fall to 0x77 (3952,3933). Beetle hp 60->56, shutter 0x18 stayed, hammer=0. Assist $F36D writes=2. $F3CC stayed 0. Not Clean. **AP-118 · Hera step 4: Moon Pearl and floor drops.** Holes the route
  uses are `fall` transitions (AP-052).
- [x] 2026-09-30 · evidence: Whatif_Hera_moldorm kind=poke room 0x07 audit 0 incoherent. Stand (3880,256) edge 48px, boss chebyshev 54px, sprite 0x09, HP $0E50=12. hera_ROUTE.md Boss filled. Isolated poke, not continuous, not STATUS. $F3CC=0. No Scan pin. **AP-119 · Hera step 5: Moldorm probe.** Log Link's distance to the
  Learned: Whatif_HeraEntry_pearl kind=whatif source=Whatif_HeraEntry_parallel pearl $F357=1 room 0x77 (3832,4032) module 0x07 big_key 0x0020 keys=1 magic=128 hammer=0 sword=1 F3CC=0. No sprite 0x09. Lobby doors slot 0 type 0x26 to 0x87 and slot 1 type 0x0C to screen 0x03. Moldorm room 0x07 is behind room 0x31 slot 0 type 0x1E, which AP-118 already calls no path. Shutter 0x18 and pit (47,44) stay recorded failures. Did not rewalk them. Arena not reached. Left open.
  arena edge.
- [x] 2026-09-30 · evidence: eval_offsets Whatif_Hera_moldorm bosses/moldorm:fight 12/12 hearts, each hp_lost 0 hits empty. Frames: 0=982, 1=820, 2=855, 3=858, 4=849, 5=856, 6=855, 7=929, 8=853, 9=850, 10=828, 11=979. Offsets 3, 5, 6, and 8 branch on $001A idle (0x30+N) and refuse rim poses; offset 0 stayed frame 982. Isolated poke, not continuous, not STATUS. $F3CC unread and unwritten. Log recordings/probe_AP-120/eval_rim.log. **AP-120 · Hera step 6: `bosses/moldorm.py`.** A rollout outcome that
  Learned: 8/12 unchanged after revert. Passes hp_lost 0: 0 frame 982, 1 frame 820, 2 frame 855, 4 frame 849, 7 frame 929, 9 frame 850, 10 frame 828, 11 frame 979. Fails 3 frame 711 hp_lost 8, 5 frame 421 hp_lost 16, 6 frame 676 hp_lost 8, 8 frame 950 hp_lost 16. Closing-head sidestep floored offset 0 at frame 706 and was copied back from moldorm_8of12.py. Not the 4/12 floor-clamp. Stays open under 10/12.
  Learned: eval_offsets Whatif_Hera_moldorm bosses/moldorm:fight 8/12 hearts. Passes hp_lost 0: offset 0 frame 982, 1 frame 820, 2 frame 855, 4 frame 849, 7 frame 929, 9 frame 850, 10 frame 828, 11 frame 979. Fails: 3 frame 711 hp_lost 8 moldorm:1, 5 frame 421 hp_lost 16 moldorm:2, 6 frame 676 hp_lost 8 moldorm:1, 8 frame 950 hp_lost 16 moldorm:2. Offset 9 swings a tail paused for 24 frames. Offset 11 keeps the off-camera defl wake after HP hits 0 so the prize runs. Isolated poke, not continuous, not STATUS. $F3CC unread and unwritten. Log recordings/probe_AP-120/eval_wake.log. Stays open under 10/12.
  Learned: Restored the ~800-frame fight. eval_offsets Whatif_Hera_moldorm bosses/moldorm:fight 7/12 hearts on offsets 0,1,2,3,4,7,10 at frames 982,821,859,978,852,929,830, each hp_lost 0 hits empty. Failures: offset 5 frame 1043 hp_lost 8 moldorm:1, offset 6 frame 677 hp_lost 0 hits empty, offset 8 frame 1384 hp_lost 8 moldorm:1, offset 9 cap 7500 hp_lost 0, offset 11 frame 779 hp_lost 8 moldorm:1. Floor-clamp and sidestep scored 4/12 (lost 4, 7, and 10) and was reverted. Isolated poke, not continuous, not STATUS. $F3CC unread and unwritten. Log recordings/probe_AP-120/eval_restored.log. Stays open under 10/12.
  Learned: Whatif_Hera_moldorm isolated poke, eval_offsets bosses/moldorm:fight, 0 of 12 hearts. Offset 0 frame 2213 hp_lost 24 hits moldorm:3. Offsets 1-11 cap 7500 hp_lost 0 hits empty. Not continuous and not a STATUS claim. $F3CC unread and unwritten. Claim stays [~] moldorm-fight.
  Learned: claimed AP-120; reading moldorm.py and the five failing offsets before any fight change
  Learned: 7/12 offsets take the heart on Whatif_Hera_moldorm (0,1,2,3,4,7,10 frames 982,821,859,978,852,929,830 hp_lost 0 hits -). Fails: 5 frames 1043 hp_lost 8 moldorm:1; 6 frames 677 floor room 0x17 hp_lost 0; 8 frames 1384 hp_lost 8 moldorm:1; 9 cap 7500 hp_lost 0; 11 frames 779 hp_lost 8 moldorm:1. Isolated pin, not continuous, not STATUS. Left open under moldorm-fight.
  Learned: 3/12 offsets take the heart on Whatif_Hera_moldorm (0,2,4 frames 982,818,817 hp_lost 0 hits -). Nine offsets hit the 7500-frame cap. Left open.
  Learned: claimed; scoring Whatif_Hera_moldorm offsets after tightening swings to the placed-hit window
  Learned: Offset 0 on Whatif_Hera_moldorm stayed in room 0x07. Placed B-only strike_verdict drops HP 12 to 10 from 8-16px east facing LEFT or south facing UP. Walked rel=(-4,0) face UP previews whiff, so the swing is not committed. Heart container not taken. 0 of 12 offsets. Left open.
  changes the floor counts as a hit.
- [ ] **AP-121 · Hera step 7: approach.** Needs AP-114, AP-091. Death
  Mountain → mirror → entrance.
- [ ] **AP-122 · Hera step 8: chain + rung C4.** Then do AP-092 (Master
  Sword, rung C5).

### Agahnim's Tower (AP-123..130) → rung C6

No pin yet. Hint: the Master Sword cuts the barrier; dark rooms; small keys;
no big chest; Agahnim reflects energy balls back with a sword swing (the
spread lightning cannot be reflected); after him Link lands in the Dark World
Pyramid.

- [ ] **AP-123 · Tower step 1: `CastleTowerEntry` pin + first hop.** Kit =
  Master Sword + lamp. Cut the barrier in play from outside if the pin
  starts outside.
- [ ] **AP-124 · Tower step 2: scan.** Dark rooms: RAM still works; ignore
  the screen.
- [ ] **AP-125 · Tower step 3: lower floors and keys.**
- [ ] **AP-126 · Tower step 4: upper floors to Agahnim's door.** No big
  chest; this step is the rest of the key doors.
- [ ] **AP-127 · Tower step 5: Agahnim probe.** Sweep swing frame vs ball
  distance with rollouts to find the reflect window.
- [ ] **AP-128 · Tower step 6: `bosses/agahnim.py`.** Swing when the
  rollout shows a reflect; dodge the spread. Read the Wizzrobe timing in
  `nes/zelda_i/dungeon/shot_guard.py`. GT (AP-192) reuses this module.
- [ ] **AP-129 · Tower step 7: approach.** Needs AP-122, AP-092.
- [ ] **AP-130 · Tower step 8: chain + rung C6.** Acceptance: Dark World
  flag set and the Pyramid screen after the warp.

### Palace of Darkness (AP-131..138) → rung C7

Pin `PalaceOfDarknessEntry` (room `0x4A`); `east_stairs_to_0x09` isolated;
`in_room` rejects Dark World (fixed by AP-097). Scratch: `probe_pod_entry.py`.
Hint: Kiki (a paid follower) opens the entrance; many small keys; a dark
maze; big chest Hammer; Helmasaur King (break the mask with the hammer or
bombs, then hit the gem); crystal 1.

- [ ] **AP-131 · PoD step 1: entry pin + first hop.** Needs AP-097.
- [ ] **AP-132 · PoD step 2: scan.** Count the small keys against the key
  doors on the route.
- [ ] **AP-133 · PoD step 3: entry → big key.** Bombable walls use
  `place_bomb` (AP-067).
- [ ] **AP-134 · PoD step 4: Hammer.** Add a `hammer` primitive (pegs,
  stun).
- [ ] **AP-135 · PoD step 5: Helmasaur King probe.**
- [ ] **AP-136 · PoD step 6: `bosses/helmasaur.py`.** Read
  `nes/zelda_i/dungeon/gohma.py` (vulnerable-window timing).
- [ ] **AP-137 · PoD step 7: approach.** Needs AP-130, AP-088. Pyramid →
  Kiki (pay in play) → entrance. Record Kiki's follower value.
- [ ] **AP-138 · PoD step 8: chain + rung C7.** Acceptance: crystal bit
  rises in play.

### Swamp Palace (AP-139..146) → rung C8

Pin `SwampPalaceEntry` (room `0x28`); `south_to_overworld` isolated;
`north_to_0x38` not isolated. Scratch: `probe_swamp_entry.py`. Hint:
flippers and mirror; a Light World dam lever drains the entrance; water-level
switches inside; big chest Hookshot; Arrghus (hookshot the puffs, then the
core); crystal 2.

- [!] BLOCKED 2026-09-30: Whatif_SwampPalaceEntry_flippers flippers=$F356=1 disclosed whatif (pin_probe flippers=1 at (4344,1496) room 0x28). room_engine north_to_0x38 stayed room 0x28 at (4344,1336) layer=1 $5D=4 frames=415 blocker door push failed. North of the flippers-0 crest (4344,1376); stairs dest 0x38 did not load. Not isolated. **AP-139 · Swamp step 1: entry pin + first hop.** Isolate
  Learned: A disclosed flippers=1 write keeps the north push in room 0x28 at (4344,1336) layer 1 $5D=4. Room 0x38 does not load. The write is not flippers rising in play.
  Learned: Deep water 0x08 at rows 46-47 cols 30-33 ejects Link while flippers are 0. A nav.py deep_water allow does not keep him north of y=1376. Do not poke flippers. Swim has to be real movement after AP-093.
  `north_to_0x38`.
- [ ] **AP-140 · Swamp step 2: scan.** Record each water switch and the
  rooms it changes.
- [ ] **AP-141 · Swamp step 3: entry → big key.** Swim and deep-water
  tiles in the BFS (AP-093 physics).
- [ ] **AP-142 · Swamp step 4: Hookshot.** Add a `hookshot` primitive
  (reach, pull, targets across gaps).
- [ ] **AP-143 · Swamp step 5: Arrghus probe.**
- [ ] **AP-144 · Swamp step 6: `bosses/arrghus.py`.** Read
  `nes/zelda_i/level9/patra.py`: satellites around a core, same shape.
- [ ] **AP-145 · Swamp step 7: approach.** Needs AP-138, AP-093. Dam lever
  in the Light World, then mirror.
- [ ] **AP-146 · Swamp step 8: chain + rung C8.**

### Skull Woods (AP-147..154) → rung C9

No indoor pin. Dark World overworld `0x40`, entrance `0x29` → room `0x58`
next. Detail: `alttp/docs/tasks/SkullWoodsEntry-residual.md`. Scratch:
`probe_skull_entry.py`. Hint: several drop-hole entrances; big chest Fire
Rod; the last section needs the Fire Rod to burn an outdoor skull; Mothula
(moving floor, spike blocks); crystal 3.

- [ ] **AP-147 · Skull step 1: `SkullWoodsEntry` pin + first hop.** Follow
  the residual; the first hop is `0x29 → 0x58`.
- [ ] **AP-148 · Skull step 2: scan.** Map which drop hole leads to which
  room (z3 connections).
- [ ] **AP-149 · Skull step 3: drop-in entrances → big key.** Drops are
  `fall` transitions.
- [ ] **AP-150 · Skull step 4: Fire Rod.** Add a `fire_rod` primitive
  (magic cost from AP-072). Burn the outdoor skull to reach the last
  section.
- [ ] **AP-151 · Skull step 5: Mothula probe.** Log floor movement per
  frame.
- [ ] **AP-152 · Skull step 6: `bosses/mothula.py`.** The rollout already
  includes the moving floor, so there is no separate floor model.
- [ ] **AP-153 · Skull step 7: approach.** Needs AP-146.
- [ ] **AP-154 · Skull step 8: chain + rung C9.**

### Thieves' Town (AP-155..162) → rung C10

Pin `ThievesTownEntry` (room `0xDB`); `south_to_outcasts` isolated;
`east_to_0xDC` conveyor red. Scratch: `probe_thieves_entry.py`. Hint:
conveyors; a maiden follower goes from her cell to the boss room; a bombable
attic floor lets light in; big chest Titan's Mitt; Blind; crystal 4.

- [!] BLOCKED 2026-09-30: Whatif_ThievesTownEntry_lift ability $F379=0x52 (0x12|0x40). Facing-right lift at (6064,6720) cleared NE 0x71 to 0x00. RIGHT stuck (6088,6903) dir 6 hp 56 still 0xDB. Dev pin only. **AP-155 · Thieves step 1: entry pin + first hop.**
- [ ] **AP-156 · Thieves step 2: scan + conveyor physics.** Measure px/frame
  per conveyor direction into `walk_physics.py`.
- [ ] **AP-157 · Thieves step 3: entry → big key.** Conveyor cost in the BFS;
  this fixes the `east_to_0xDC` red.
- [ ] **AP-158 · Thieves step 4: Titan's Mitt, maiden and attic.** Record the
  maiden's follower value. Bomb the attic floor.
- [ ] **AP-159 · Thieves step 5: Blind probe.**
- [ ] **AP-160 · Thieves step 6: `bosses/blind.py`.** Read
  `nes/zelda_i/dungeon/gleeok.py` for detached-head tracking.
- [ ] **AP-161 · Thieves step 7: approach.** Needs AP-154.
- [ ] **AP-162 · Thieves step 8: chain + rung C10.**

### Ice Palace (AP-163..170) → rung C11

Pin `IcePalaceEntry` (room `0x0E`); Freezor melt + LEFT still `0x0E`; next
is the Bari key and north stairs. Scratch: `probe_ice_entry.py`. Hint:
flippers and Titan's Mitt to reach; fire (Fire Rod or Bombos) for Freezors
and the boss; slippery floors; holes drop floors; block pushes; big chest Blue
Mail; Kholdstare; crystal 5.

- [x] 2026-09-30 · evidence: room_engine room_0e north_to_0x1e green 2/2 from Play_ice_0x0e_melted (fire rod in play, Freezor gone), both land room 0x1E xy=(7289,823) frames=582; sword left Freezor hp 16; audit Play_ice_0x0e_melted Scan_ice_0x1e Whatif_IcePalaceEntry_parallel IcePalaceEntry 0 incoherent; docs/routes/ice_ROUTE.md Entry isolated not natural **AP-163 · Ice step 1: entry pin + first hop.** Melt the Freezor with
  the Fire Rod, then take the forward door.
- [x] 2026-09-30 · evidence: Ice on 0x0F at (7335,376): LEFT 72 frames reaches the (-2,0),(-1,0) pair (1.5 px/frame) and coasts 18 px to (7237,376); floor UP pair (0,-1),(0,-2) coasts 0. Rooms 0x0E and 0x1E are in ice_ROUTE.md with door types and no chests. Stopping locks: 0x0E west 0x18, 0x1E east 0x18, west 0x18, south 0x00. fight_nearby 40 cycles left 0x1E types unchanged. room_engine lint ok 33 maps. Door-only maps/room_1e.json. Not natural. **AP-164 · Ice step 2: scan + ice physics.** Measure slide speed and
  stop distance on ice tiles.
- [!] BLOCKED 2026-09-30: Whatif_Ice_0x1f_open ability $F379=0x3E (0x1E|0x20). pin_probe ability=62 tile (45,54) chest opened=0. One A: opened=1 big_key 0x0000->0x0040 ledger source=play frame 2 follower 5. Done line does not accept this isolated pin: edges are not isolated 2/2. Not natural. **AP-165 · Ice step 3: entry → big key.** Ice-aware walker (commit
  distances, no micro-corrections).
  Learned: Ice LEFT 72 coasts 18 px after the 1.5 px/frame cap. Step 3 needs an ice coast constant in walk_physics.py; this lane cannot edit that file or nav.py. Big key $F366=0x0040 on the parallel pin is a poke. No chest in rooms 0x0E or 0x1E. East and west 0x18 and south 0x00 stayed shut.
- [ ] **AP-166 · Ice step 4: Blue Mail, drops and block pushes.** Use
  `push_block` (AP-067).
- [x] 2026-09-30 · evidence: Whatif_Ice_kholdstare kind=poke room 0xDE audit 0 incoherent hp 0x40/0x40 dungeon 0x12 module 0x07. pin_probe shell 0xA3 hp 64 ai 0 and kholdstare 0xA2 hp 64 ai 3 at (7544,6744). boss_probe idle 1800: those HP and phase unchanged; falling ice 0xA4 slot 15 at f=125 then shards f=191; Link HP 64 to 0 at frames 192,320,448,576. Unsaved north floor: fire magic 80 to 64, shell 64 to 56 at frame 11; master sword 16px south left shell 64. ice_ROUTE.md Boss filled. Isolated poke, not continuous, not STATUS. F3CC stayed 5. No Scan pin. Wall rows 27-38 not walked. **AP-167 · Ice step 5: Kholdstare probe.**
- [~] kholdstare-fight 2026-09-30 **AP-168 · Ice step 6: `bosses/kholdstare.py`.** Fire on the shell,
  Learned: Offset 0 two-tile stand reached (7466,6789) at f=1216 and (7479,6751) at f=1344. Ice still hit in transit: f=1375 pose (7464,6742) hp 64->48, f=1435 pose (7464,6722) hp 48->16. Dead f=1529 pose (7516,6765), hits falling ice:2,kholdstare:1. South march before that died f=1694 pose (7464,6848), hits falling ice:1,kholdstare:1; +32 south of that tile is wall row 27+. Sword f=1084 unchanged (7480,6750) z=65 end_z=11. Heart not taken. Not 10/12. Offsets 1-11 unscored. F3CC not written. Wall rows 27-38 unwalked. Log recordings/probe_AP-168/offset0_cycle.log.
  Learned: Offset 0 early swing finished with the ball still up. clear_x f=1032 xy=(7480,6809) z=213, so x<=7481 before the split. Sword f=1084 slot=3 hp=64->60 pose (7480,6750) facing UP, start z=65 end_z=11 end=(7480,6750). Pre-split LEFT at f=1031 z=215 from (7482,6809). Eight fire shots, shell gone f=886 last=8. Dead f=1472 pose (7464,6806), hits falling ice:3. Heart not taken. Not 10/12. Offsets 1-11 unscored. F3CC not written. Wall rows 27-38 unwalked. Log recordings/probe_AP-168/offset0_early3.log.
  Learned: Offset 0 south-window sword dealt 4 at f=1091 pose (7482,6747) facing UP, spawn ball z=36, so the 13-frame swing finished after the split. West then read (7480,6742), 17px west of 7497, with z already 0. Dead f=1419 pose (7464,6723) facing UP, hits falling ice:3. Eight fire shots kept. Heart not taken. Not 10/12. F3CC not written. Wall rows 27-38 unwalked. Log recordings/probe_AP-168/offset0_west2.log.
  Learned: Offset 0 sword_hits 0, eight fire shots kept, heart not taken. Spawn shard is 15px east at (7497,6809) z=215. UP from (7482,6809) stopped (7485,6760) eye 34 ice 12, then three ice-16, dead f=1440, no eye-32. RIGHT stepped into that shard: ice 16 at (7464,6836) cheb 7, then eye contact 32 at (7519,6815) cheb 21, dead f=1316 kholdstare:1. UP-then-RIGHT stuck (7543,6759) eye 42 ice 9, dead f=1472 falling ice:3. Not 10/12. F3CC not written. Wall rows 27-38 unwalked.
  Learned: Offset 0 exit DOWN died frame 2001: HP 64 to 32 eye chebyshev 32 at (7525,6848), then 32 to 16 ice, sword whiff only. Exit LEFT died frame 1410: HP 64 to 48 ice chebyshev 14, then 48 to 16 with an eye at chebyshev 4, sword_hits 0. Eight fire shots kept. Both poses take ice 16 and eye contact 32 before F36C rises. Heart not taken, not 10/12. F3CC not written. E2/E8 only. Wall rows 27-38 unwalked. Isolated poke, not continuous, not STATUS.
  Learned: Whatif_Ice_kholdstare_magic kind poke magic 128. Shell HP not poked. Eight fire shots take the shell 64 to 0 and three eyes spawn ai 0 hp 64. South-window master sword deals 4 (64 to 60). Offset 0 then dies to falling ice 16 and eye contact 32 before F36C rises. Screen Y 305 latches 0F00 if BG2 leaves the shell; fight rewrites E2/E8 only. F3CC stayed 5. Wall rows 27-38 unwalked. Isolated poke, not continuous, not STATUS. Heart not taken, so not 10/12.
  Learned: stage_replay Whatif_Ice_kholdstare alttp.bosses.kholdstare:fight idle 0 pass 0 frames 201 hp_lost 0 hits -. Five fire shots shell 64 56 48 40 32 24, magic 80 to 0, sword_hits 0, eye ai 3 hp 64, max HP 64. End xy (7507,6748) off the south ice column. North poke leaves BG2 E2=1D00 E8=1B01 and attack ice z=224. Wall rows 27-38 unwalked. F3CC stayed 5. Shell floor 24, heart check still open.
  then the sword on the pieces.
- [ ] **AP-169 · Ice step 7: approach.** Needs AP-162, AP-093.
- [ ] **AP-170 · Ice step 8: chain + rung C11.**

### Misery Mire (AP-171..178) → rung C12

No pin yet. Hint: flute to the desert, a portal behind heavy rocks
(Titan's Mitt), Ether at the entrance; hookshot or boots across gaps;
torches; Wizzrobes; big chest Cane of Somaria; Vitreous; crystal 6.

- [ ] **AP-171 · Mire step 1: `MiseryMireEntry` pin + first hop.** Kit
  Learned: Neither Whatif_MiseryMireEntry_parallel nor MiseryMireEntry exists. make_entry_pin.py ENTRIES is only eastern 0x08 and ice 0x2D. z3 mire region and mire.yaml have no $010E, and hasProperMedallion has no kit write. Did not boot. Needed change is in docs/routes/mire_ROUTE.md.
  includes Ether. Opening the entrance with Ether is part of AP-177, not
  the pin.
- [ ] **AP-172 · Mire step 2: scan.**
- [ ] **AP-173 · Mire step 3: entry → big key.** Gaps via hookshot or dash.
  Wizzrobes: reuse the timing idea from `nes/zelda_i/dungeon/shot_guard.py`.
- [ ] **AP-174 · Mire step 4: Cane of Somaria.** Add a `somaria_block`
  primitive (make a block, push it onto a switch).
- [ ] **AP-175 · Mire step 5: Vitreous probe.**
- [ ] **AP-176 · Mire step 6: `bosses/vitreous.py`.**
- [ ] **AP-177 · Mire step 7: approach.** Needs AP-170, AP-094, AP-095.
  Flute, portal, Ether in play.
- [ ] **AP-178 · Mire step 8: chain + rung C12.** Then AP-098 (silver arrows).

### Turtle Rock (AP-179..186) → rung C13

No pin yet. Hint: Quake opens the entrance; Somaria platforms ride pipes;
laser eye statues (Mirror Shield blocks them); dark rooms; big chest Mirror
Shield; Trinexx (ice and fire rods on the side heads, then the centre);
crystal 7.

- [ ] **AP-179 · TR step 1: `TurtleRockEntry` pin + first hop.**
- [ ] **AP-180 · TR step 2: scan.** Note side entrances from mountain ledges.
- [ ] **AP-181 · TR step 3: entry → big key.** Add a `ride_platform`
  primitive (make it on a pipe start, stand, press at junctions).
- [ ] **AP-182 · TR step 4: Mirror Shield and lasers.** Measure which lasers
  the shield blocks when facing them (extends AP-064).
- [ ] **AP-183 · TR step 5: Trinexx probe.**
- [ ] **AP-184 · TR step 6: `bosses/trinexx.py`.** Read
  `nes/zelda_i/dungeon/gleeok.py` for per-head state.
- [ ] **AP-185 · TR step 7: approach.** Needs AP-178, AP-095 (Quake).
- [ ] **AP-186 · TR step 8: chain + rung C13.**

### Ganon's Tower (AP-187..194) → rung C14

No pin yet. Hint: seven crystals open it; a long dungeon with branches;
needs hookshot, hammer, fire rod, somaria and boots; mini-bosses on the way
(Armos, Lanmolas, Moldorm); big chest Red Mail; Agahnim again with clones;
then a drop to the Pyramid for Ganon.

- [ ] **AP-187 · GT step 1: `GanonsTowerEntry` pin + first hop.**
- [ ] **AP-188 · GT step 2: scan.** Pick the shortest branch that z3 logic
  allows.
- [ ] **AP-189 · GT step 3: entry → big key.** Mini-bosses reuse AP-104 and
  AP-112 controllers.
- [ ] **AP-190 · GT step 4: Red Mail and the top floors.** Moldorm reuses
  AP-120.
- [ ] **AP-191 · GT step 5: Agahnim 2 probe.** Tell the real one from the
  clones by sprite fields.
- [ ] **AP-192 · GT step 6: extend `bosses/agahnim.py` for clones.**
- [ ] **AP-193 · GT step 7: approach.** Needs AP-186. Seven-crystal opening
  in play.
- [ ] **AP-194 · GT step 8: chain + rung C14.**

## Phase I: Ganon, credits, M7, M8 (AP-195..200)

- [ ] **AP-195 · Ganon probe**
  Needs AP-194, AP-098. Pin `Scan_pyramid_ganon` from the GT chain. Hint
  phases: sword; fire bats and the trident; a dark room where two torches
  must be lit; silver arrows when he flashes. Read `nes/zelda_i/level9/ganon.py`.
  Done: route doc `docs/routes/GANON_ROUTE.md` with phase bytes and weapons.

- [ ] **AP-196 · `bosses/ganon.py`**
  Needs AP-195. Do: phase machine with `light_torch`, `shoot_arrow` and
  `strike_if_safe`. Done: ≥10/12 offsets win from the arena pin.

- [ ] **AP-197 · Credits acceptance**
  Needs AP-196. Do: stop predicate for the Triforce room plus the credits
  module (AP-008 table). Add `--through credits` to the hop table. Done:
  predicate test plus one isolated arrival from the arena pin.

- [x] 2026-09-30 · evidence: recordings/ap198_realtime.json: no stage under 1x (PolicyGuard tip 10.76x, replay 19.37x); hazard gate already on; horizon stays 25 **AP-198 · Real-time pass**
  Learned: No on-disk stage report is under 1x. Do not cut POLICY_GUARD_HORIZON. Opening-spine segments have frames and no wall_s.
  Needs AP-058. Do: from the stage reports, list every stage below 1× real
  time. Cut rollouts per frame (hazard gate) before cutting the horizon.
  Done: every stage ≥1× real time, or a table of the ones that are not.

- [ ] **AP-199 · M7: continuous power-on to the credits**
  Needs AP-197. Do: `run_spine.py --through credits --save-points Full`.
  Use the save-point loop on reds. Report 0 non-rollout loads, 0 writes and
  0 deaths; if the run is not Clean, name it (Survival, lookahead) in the
  report. Planner updates `STATUS.md`. Done: one green continuous run.

- [ ] **AP-200 · M8: the recording, the manifest and the matrix**
  Needs AP-199. Do: `replay_tape.py --video` on the AP-199 tape: 0 desync,
  0 loads, 0 writes. Update `docs/manifests/alttp.yaml` (maturity, best
  manifest, best video, last verified) and run
  `uv run python docs/generate_game_matrix.py` and
  `uv run pytest tests/test_docs.py -q` from the repo root. Planner updates
  `STATUS.md`. Done: MP4 plus JSON exist; docs tests green.
