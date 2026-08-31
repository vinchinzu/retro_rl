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
  **fixture-live**. Pond `0x42`, drain, and entry room still **hypothesis**.
  Stop at fixture-live; integrator owns natural-segment / spine-green.

- **exact predecessor:** not L7-ready. Current Survival tip is L6 residual
  play `0x09` `(56,109)` mode 5, keys=3, bombs=8, Bow=1, Rod=0, TF=`0x1F`.
  Do not treat this as an L7 start. Expected L6 leave (unmeasured): overworld
  play, TF `0x3F`, Whistle, Rod, Bow; Food likely 0; Candle 1 (blue). Do not
  hardcode OW `0x22`.

- **required inventory/capabilities:** TF `0x3F`, Whistle ≥1, Rod ≥1, Bow ≥1,
  sword. Food 0 until natural 60R Bait at shop hyp `0x34`. Candle remains 1
  until Red Candle. Geometry-only pond mapping may run without Whistle.

- **ordered internal stage names and controller factories:**
  1. `level7_post_l6_overworld` — `make_post_l6_overworld_controller(handoff, hops)`
     (`PostLevel6Handoff`; default `UNMEASURED_POST_L6_HANDOFF` refuses to move)
  2. `level7_bait_purchase` — `make_bait_purchase_controller(plan)`
     (60R, shop `0x34`, no Food/rupee write; fails `bait_need_60_rupees` /
     `bait_shop_geometry_unobserved`)
  3. `level7_pond_drain_entry` — `make_pond_entry_controller()`
     (still unverified; drain needs naturally selected Whistle)

  Recon-only (not a spine stage): `OverworldToLevel7PondController` from
  `PostSwordStart` (no Whistle).

- **exact endpoint predicate:** `level7_entry_stop` — live L7 play in the
  observed entry room, TF `0x3F`, Whistle and Food owned, `route_eligible`
  and `evidence` in `{natural-segment, spine-green}`. Room id is still
  `None` so the predicate fails closed.

- **expected inventory deltas:** Food 0→1 (natural 60R buy). No TF change.
  Keys/bombs unchanged on OW. Candle stays 1. No `ADDR_FOOD` / rupee /
  Whistle / door / TF writes.

- **known dead beliefs / first missed RAM claim:**
  - Dead: `hop10_ay` DOWN from `0x53` `(224,173)` (v9). Inland-left first.
  - Dead: start-`0x77` pond walk as the spine post-L6 path.
  - Dead: OW `0x22` as proven L6 leave; live L6 prefix `0x09` as L7 start.
  - First missed RAM this sitting after `0x53→0x52`: play `0x52` `(112,181)`
    hop `0x42` UP (`hop11_ax` then `unstick_wait`). North gap is not x=112.

- **fixture provenance:** `PostSwordStart` Survival `--no-video`.
  `recordings/l7_dnp_pond_53.json` leftover play `0x52` `(112,181)`,
  `success=false`, `route_eligible=false`. Prior miss v9 `0x53` `(224,173)`.

- **files changed / public target:** `nes/zelda_i/level7/{dungeon,graph,entry,overworld,path,hops}.py`,
  `nes/zelda_i/docs/LEVEL7_ROUTE.md`, `nes/zelda_i/tests/test_level7_{overworld,dungeon,hops}.py`,
  `nes/zelda_i/docs/tasks/l7-handoff.md`, gitignored `recordings/l7_dnp_pond_53.json`.
  Public target: **`level7-entry`**.

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
