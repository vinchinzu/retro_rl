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

## Onward toward Red Candle — resume point

The BFS `preferred_path(GORIYA_HINT → RED_CANDLE_CELLAR)` mainline does
**not** go through `0x6C`; it goes `0x6B` **LEFT → `0x6A`**, then a **BOMB
gate UP** out of `0x6A` toward the compass/stalfos cluster, then
`…→ GORIYA_PRE_HUNGRY → HUNGRY_GORIYA (KEY, needs Food) → MAP →
HIDDEN_RUPEES (BOMB) → GORIYA_POST_RUPEE → WEST_LOCK_SKIP → CANDLE_PUSH
(BOMB) → RED_CANDLE_CELLAR`. All rooms past `0x6C`/`0x6A` are still
**hypothesis**.

**Next sitting starts here:** from `Level7InteriorReconFixture`, walk
`0x6B` LEFT → `0x6A`, then recon the `0x6A` **north bomb-wall** (bomb stand
x, target y, `opens_to` room id). `dungeon/bomb_wall.py`
`BombWallController` is the reusable driver — it needs a `BombWallLike`
geometry object once the north wall is measured. The fixture already
carries 8 bombs + 4 keys + Food 1 for the whole downstream chain. Do
**not** poke `ADDR_CANDLE`.

Alternative unverified: `0x6C` (`DIGDOGGER_1`) RIGHT → `STALFOS_KEY`
(source), and `0x6B` UP retried off-centre / after a true KILL_CLEAR
sentinel for `OLD_MAN_NOSE`.

## Dead beliefs (this sitting)

1. Dead: `0x6B` UP at centre-x on the `y=93` band reaches `OLD_MAN_NOSE`.
   Blocked 2/2 at `(128,93)` even after the goriya clear.
2. Dead (carried): `0x6B` RIGHT → `DIGDOGGER_1` is the Red-Candle mainline.
   It is a skippable whistle-split spur; the mainline is LEFT + bomb-UP
   through `0x6A`.
3. The `sweep_l7_room6b.py` "all four doors OPEN" read: RIGHT and LEFT
   confirmed OPEN 2/2; UP is not walkable from centre; DOWN (toward entry
   `0x6A`/`0x79` side) not separately re-probed this sitting.

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
- **evidence label:** **fixture-live entry pin + `0x79→0x69→0x6A→0x6B→0x6C`
  prefix** (plus `0x6B` LEFT↔`0x6A` backtrack); rooms past `0x6C`/`0x6A`-north
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
     `make_room6b_east_controller()` **(new, 2026-09-04, 2/2)** walks `0x6B`
     west mouth → OPEN east doorway to live `$EB=0x6C` (`DIGDOGGER_1`,
     `(16,141)`, 283f). Waypoint: ride `y=109` band east past the central X,
     drop `x=200`→`y=141`, push `x=224`. **Assumes `0x6B` goriya cleared
     upstream** (fixture or a not-yet-built kill-clear stage). Not a spine
     stage; `0x6C` is the skippable whistle-split spur, not the Candle
     mainline.
  2. `level7_entry_to_hungry_goriya` — `make_entry_to_goriya_controller()`
     (fails `hungry_goriya_requires_food` if Food=0; else room unobserved)
  3. `level7_tip_of_nose_stairs` — `make_tip_stairs_controller()` (blocker + ledger notes)
  4. `level7_red_candle_pickup` — `make_red_candle_controller()`
     (`ADDR_CANDLE` 1→2 natural; room unobserved)
  Executable chapter chain (`level7_red_candle_chapter_stages`) is unchanged:
  `entry_first_door → entry_to_hungry_goriya (fail-closed) → tip_stairs →
  red_candle_pickup`. Stages 1a/1b/1c are recon-wired only.
- **endpoint:** `level7_red_candle_stop` — Candle==2, TF `0x3F`, Whistle
  retained, Food==0, exact live room. Room id `None` → fail closed.
- **resume point:** from `Level7InteriorReconFixture`, `0x6B` LEFT → `0x6A`
  (2/2), then recon the `0x6A` **north bomb-wall** geometry and drive it
  with `dungeon/bomb_wall.py` `BombWallController`. Mainline is LEFT+bomb-UP
  through `0x6A`, NOT `0x6C`.
- **expected deltas:** Food 1→0 at Hungry Goriya; Candle 1→2 at cellar.
  Ledger hyp net (dungeon): keys +1+1−1−1+1−1, bombs several −1 wall skips.
  Prefer bomb north of Map over locked east (fifth lock).
- **dead beliefs:** fifth lock required; source RAM room ids as stop specs;
  N path is Moldorms (live dest `0x69` is goriya `0x05`); `0x69`/`0x6A`/`0x6B`
  east are key/kill-clear doors (all OPEN — `cur_opened_doors` stays 0 on
  every live L7 doorway); per-pixel occupancy is safe for a short in-room
  traverse (four graded misses on one cell box Link in — use waypoints);
  `0x6B` UP at centre-x reaches `OLD_MAN_NOSE` (blocked 2/2 at `(128,93)`);
  `0x6B` RIGHT → `DIGDOGGER_1` is the Candle mainline (it is a skippable
  spur).
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
