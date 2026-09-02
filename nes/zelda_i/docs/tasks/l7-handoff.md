# L7 Wave A handoff (copy for integrator)

Did not STATUS-promote. Did not edit `STATUS.md` / `.beads`. Did not write
shared spine. `route_eligible=false` on every fixture.

Public `--through` targets unchanged: `level7-entry`, `level7-red-candle`,
`level7`.

---

## L7-A — topology + entry preparation

- **chapter id:** `rr-8t4.1` / `level7-entry`
- **evidence label:** mixed. Offline graph + Bait/post-L6 interface =
  **hypothesis**. Start-based `0x53→0x52` inland-left micro =
  **fixture-live**. Post-L6 bait prefix `0x22→0x32→0x33→0x23→0x24→0x25` =
  **fixture-live**. Pond `0x42`, drain, and entry room still **hypothesis**.
  Stop at fixture-live; integrator owns natural-segment / spine-green.

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
     (still unverified; drain needs naturally selected Whistle)

  Recon-only (not a spine stage): `OverworldToBaitShopController` from
  `Level6ExitOverworld`; `OverworldToLevel7PondController` from
  `PostSwordStart` (no Whistle).

- **exact endpoint predicate:** `level7_entry_stop` — live L7 play in the
  observed entry room, TF `0x3F`, Whistle and Food owned, `route_eligible`
  and `evidence` in `{natural-segment, spine-green}`. Room id is still
  `None` so the predicate fails closed.

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

- **L7 start pin — BLOCKED.** A `Level7Entrance` save-state cannot be built
  yet: the `Level6ExitOverworld` precedent poked inventory onto a *live* OW
  leftover, but **no L7 room has been observed live** (`level7/dungeon.py`
  room ids all `None`), so there is no state to anchor. The pin folds into
  `rr-8t4.2` (L7-B) first-room recon — its first observed room *is* the L7
  start. Closest current checkpoint: the continuous-tape leftover at
  `level7_pond_drain_entry` (post-Bait OW `0x25`, Food 1).

- **fixture provenance:** Phase 1 is a **continuous power-on** tape, not a
  save-state fixture. `--through level7-entry` (`recordings/l7p1_foodfix.json`)
  drives power-on → measured L6 fanfare exit → `level7_post_l6_overworld`
  green `0x22→0x25` → `level7_bait_purchase` green (Survival Food fixture) →
  fail-closed at `level7_pond_drain_entry`. `set_state=0`, `deaths=0`. Pond
  recon remains `recordings/l7_dnp_pond_53.json` leftover play `0x52` `(112,181)`.

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
- **evidence label:** **hypothesis** (fail-closed factories; graph only)
- **predecessor:** L7-A endpoint (unobserved entry room, TF `0x3F`, Food owned)
- **required:** Food ≥1 (Hungry Goriya), Whistle, bombs for wall skips, keys
  for three locks (skip fifth via bombs)
- **stages / factories:**
  1. `level7_entry_to_hungry_goriya` — `make_entry_to_goriya_controller()`
     (fails `hungry_goriya_requires_food` if Food=0; else room unobserved)
  2. `level7_tip_of_nose_stairs` — `make_tip_stairs_controller()` (blocker + ledger notes)
  3. `level7_red_candle_pickup` — `make_red_candle_controller()`
     (`ADDR_CANDLE` 1→2 natural; room unobserved)
- **endpoint:** `level7_red_candle_stop` — Candle==2, TF `0x3F`, Whistle
  retained, Food==0, exact live room. Room id `None` → fail closed.
- **expected deltas:** Food 1→0 at Hungry Goriya; Candle 1→2 at cellar.
  Ledger hyp net (dungeon): keys +1+1−1−1+1−1, bombs several −1 wall skips.
  Prefer bomb north of Map over locked east (fifth lock).
- **dead beliefs:** fifth lock required; source RAM room ids as stop specs.
- **fixture:** none. `route_eligible=false`.
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
