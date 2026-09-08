# Zelda I — post-alpha cleanup & polish campaign

Alpha is done: power-on → credits is green (`recordings/level9_credits.json`,
354,346f, 198 stages, `set_state_count=0`, deaths 0). This plan turns that
tape into a codebase worth keeping.

Goals, in the owner's words: *beautiful codebase · finish faster · clean
step-by-step finish · better overworld combat · 100% items · more fun to
watch · no assists · modular and reusable for Zelda 3 / SMZ3 later.*

**Randomizer integration is explicitly out of scope for this campaign.** Phase 2
preserves and widens the seam (`retro_harness.adventure` already holds
`ProgressionState` / `ItemCheck` / `SeedPlacement`); nothing is built on it now.

## Measured baseline (the regression oracle)

| Metric | Value | Source |
|--------|-------|--------|
| Power-on → credits | 354,346 f ≈ **98.3 min** | `recordings/level9_credits.json` |
| Boot → first playable | 199 f (3.3 s) | same — **already fast, do not touch** |
| Stages | 198; top-10 = 37.8%, top-20 = 55.6% | same |
| Combat stages | 53 stages, 144,100 f (41%) | same |
| Backtrack stages | 26,569 f (7.5%) | same |
| `attack_cooldown` idle | 4,997 f (83 s), 69% of Ganon | same |
| `occupancy_misses` | 3,183 | same |
| Assist damage restored | 1,117 units over 85 rooms | same |
| Unit tests | 891 pass in **1.25 s** | `pytest nes/zelda_i/tests` |
| Production LOC (`level*/`) | 57,241 | — |
| `scratch/` | 48,083 LOC, 205 tracked files | — |

Every phase below re-runs `pytest nes/zelda_i/tests -q` (1.25 s) and, at phase
close, one power-on `--through level9-credits` compared against this row.

---

# Phase 0 — Safety net

Before any deletion or refactor.

- [x] **0.1** Freeze the baseline. Copy `recordings/level9_credits.json` to
      `recordings/BASELINE_20260907.json` and reference it here. It is the
      only full green tape; every later run diffs against it.
- [x] **0.2** `nes/zelda_i/scripts/compare_run.py` reads two spine JSON reports
      and prints end_frame delta (B-A), ok/failed_stage, a per-stage
      frames/success table aligned by name, and only-in-A/B stages.
- [x] **0.3** Note the tripwires so no phase trips them blind:
      `tests/test_hygiene_architecture.py:242 test_root_is_thin` asserts exact
      set equality against `_ALLOWED_ROOT_PY` (`:29`); `test_no_leftover_flat_modules`
      does `find_spec` over a gone-modules list. Structural changes must update
      both in the same commit.

---

# Phase 1 — Cruft ✅ **DONE 2026-09-07** (53,964 LOC out)

`scratch/` has **zero import coupling** — there is no `scratch/__init__.py`, it
is not an importable package, and repo-wide `from zelda_i.scratch` returns 0
hits. All ~95 cross-references are provenance comments.

Order matters: **move before delete**, so a tracked `.provenance.json` is never
left without its regenerator.

- [x] **1.1** Move 12 fixture/pin builders `scratch/` → `scripts/fixtures/`.
      These produce gitignored `.state` files whose `.provenance.json` sidecars
      *are* tracked in `custom_integrations/` — they are the only way to
      regenerate fixtures on a fresh checkout.
      `build_level8_bush_candle_fixture.py`, `capture_level8_entrance_fixture.py`,
      `pond/capture_pond_natural_pin.py`, `build_level8_interior_recon_fixture.py`,
      `build_level7_interior_recon_fixture.py`,
      `build_level7_interior49_ladder_fixture.py`, and the six `pin_*.py`
      (1,563 LOC). *Check first:* if `--through level9` runs power-on without
      loading pins, the six `pin_l9_*`/`pin_ow_row0` are DELETE instead.
- [x] **1.2** `git rm -r nes/zelda_i/scratch/` — 193 files, **~48,400 LOC**.
      All for closed beads (`rr-8t4.1/.2/.3`, `rr-6o7.*`, `rr-sz8.1–.7`,
      `rr-tne2`). Includes the `experiment_l9_05_push_v{3,4,5}.py` stacked
      residual banned outright by CODING_STANDARDS.
- [x] **1.3** Delete 17 stale `docs/tasks/` files (**~5,090 lines**), all for
      beads `bd` reports closed. Largest: `l7-handoff.md` (1,556),
      `rr-n91a-residual.md` (583), `overnight-spine.md` (415, self-labelled
      "historical BLOCKED log"). **Keep** `rr-ps7-l7-l9-parallel-plan.md`
      (bead open). **Merge, don't delete** `rr-5eb2-gleeok-model.md` (391) —
      its live Gleeok body-type/HP model goes into `docs/LEVEL8_ROUTE.md`.
      **Unsure:** `ow-handoff.md` (165) — check against open `rr-8t4.4/.5`.
- [x] **1.4** Rewrite dangling scratch citations in `docs/LEVEL{7,8,9}_ROUTE.md`
      (~42 references) as measured-constant statements. Do this in the same
      commit as 1.2 so no doc points at a deleted path.
- [x] **1.5** Delete dead production code (**~4,500 LOC**):
      - `route/health_cost.py` (229) — zero importers.
      - `route/item_gate_routes.py` + `route/item_gate_hops.py` (1,127) — a
        closed 2-module island; delete both or neither.
      - `route/heatmap.py` + `scripts/rank_damage_heatmap.py` (257) — cited in
        zero docs. **Note:** re-derivable, and Phase 6 wants damage ranking —
        prefer *keeping* `heatmap.py` if Phase 6 is near-term.
      - `level8/bush.py` (44) — self-labelled "Backward-compatibility re-export shim".
      - `level6/east3a.py` (171) — docstring says "Historical … diagnostic";
        its conclusion is "RIGHT cannot transition". Unwire from
        `level6/spine_suffix.py:29`.
      - `level8/recon.py` (293) — dev evidence, not runtime.
      - `level9/stair_session.py` (720) + `level9/stair_suffix.py` (560) +
        `level9/stair_run.py:586-638` (six clone `run_roomNN_*_to_credits`
        runners) — probe CLIs carrying the `FULL_LOADOUT` god-mode fixture
        (25 addresses incl. `ADDR_TRIFORCE=255`). Not on the spine.
      - `level9/room51.py` (593) — an env-loop probe living in production.
      - `dungeon/ops.py poke_link_position` + `level8/overworld.py:843
        poke_candle_for_recon` — zero call sites, retired by contract, and
        `poke_candle_for_recon`'s own docstring says "Not allowed under
        ASSIST_CONTRACT". Drop the `assist.py:23,280` re-exports too.
- [x] **1.6** Trim the dead re-export bodies in `door_graph/__init__.py` (162)
      and `level8/__init__.py` (67) — nobody imports through these facades.
      Keep the files (package markers).
- [x] **1.7** Reclaim 2.8 GB of stale agent worktrees.
      `agent-ac8ee322d70115515` is fully merged → remove.
      `agent-afbe9499e5cc76c3b` holds **unmerged** `bd9ac79f` (L7-A bait-route
      recon) — **owner decision required** before removal; that recon may feed
      open bead `rr-8t4.4`.
- [x] **1.8** `docs/STATUS.md` — **not edited.** `AGENTS.md` is explicit that
      the planner owns STATUS and that assisted greens are not Clean STATUS, so
      the M5 Clean gate ("Level 1 complete") stays as written. Two factual
      staleness flags for the planner: "Last verification 2026-07-28", and
      "Ready frame (probe) ~567" — the measured boot is now **199 f**.

## Phase 1 result (measured)

| | Before | After | Δ |
|---|---|---|---|
| Tracked files in `nes/zelda_i` | 828 | **592** | −236 |
| Python LOC | 147,157 | **93,193** | **−53,964 (−36.7%)** |
| `route/` | 4,964 | 1,400 | −3,564 |
| `level9/` | 8,520 | 6,038 | −2,482 |
| Unit tests | 891 pass / 1.25 s | **886 pass / 1.19 s** | −5 (deleted with `east3a`) |
| Stale worktrees | 2.8 GB | 0 | −2.8 GB |

`compileall` clean across every package; `zelda_i.spine.survival` imports with
all 93 through-stops incl. `level9-credits`; zero dangling references to any
deleted module.

**Deviations from the plan, and why:**
- **`level8/recon.py` kept** — the audit called it dead, but `level8/dungeon.py:80`
  imports it for live `LEVEL8_HYPOTHESIS_ROOMS` room data.
- **`level9/room51.py` kept, reduced 593 → 110 LOC.** `level9/natural_path.py:66`
  needs exactly one export (`room51_to_41_step`). Extracting the live half is
  what freed the whole `stair_run`/`stair_session`/`stair_suffix` island.
- **All 12 pin/fixture builders moved, none deleted.** CODING_STANDARDS: *"If the
  A/B loop would lose load-pin, play, or compare, fold; do not delete."*
  Pin builders **are** load-pin.
- **`bd9ac79f` tagged, not cherry-picked.** The commit adds four `scratch/`
  probes; merging it would resurrect the tree just deleted. It is now permanently
  reachable at tag **`recon/rr-8t4.5-bait-shop`**, and its actual findings (the
  `0x34` shop geometry the open bait bead needs) are written into
  `docs/OVERWORLD_DOORS.md`.
- **`rr-5eb2-gleeok-model.md` and `ow-handoff.md` merged, not deleted** — into
  `docs/LEVEL8_ROUTE.md` (the three-Gleeok body-type/HP/stand table that Phase 2.6
  needs) and `docs/OVERWORLD_DOORS.md` (leave→mouth stitch + the open bait blocker).

**Owner decisions gating 1.5 — resolved 2026-09-07: all three deleted.**
- `route/composer.py` + `catalog.py` + `catalog_later.py` + `legs.py` +
  `legs_later.py` + `scripts/compose_named_route.py` (**2,072 LOC**) — a closed
  dependency island, referenced in no doc, but it *is* the "named route catalog"
  concept and open epic `rr-ps7` may want it. Also a second dispatcher parallel
  to the Composer (see 2.7).
- `tas/` (733 LOC) — self-contained; `trace_route.py:1` says "planning tool, not
  product evidence". All L1–L9 routes are built, so it has served its purpose.

---

# Phase 2 — Consolidation (~17,000 LOC out of `level*/`)

The house rule (CODING_STANDARDS § Composer) is one dispatcher, behavior added
as **rows**. The row pattern already exists and is used by exactly one level:
`level6/door_hop.py:65 DoorHopSpec` — a 30-field frozen dataclass driving 12
door hops through one controller. Phase 2 is "promote that, everywhere".

Ordered by effort/risk ratio, easiest proof first.

- [x] **2.1** *(the proof)* One `TriforceSettleSpec` row table replacing five
      verbatim clone controllers — `overworld/settle.py` (`TRIFORCE_SETTLES`).
      Deltas are `require_screen`, the TF bit, and a `raft` item predicate.
      Historical no-arg names (`PostL4TriforceSettleController`, …) are bound
      subclasses. **Done 2026-09-07.**
- [x] **2.2** Delete the self-labelled shims: `level1/path.py` `_CLEAR_EXPORTS`
      `__getattr__`, `level3/dungeon.py` path/raft/geometry `__getattr__`.
      Callers import `level1.clear`, `level3.path` / `raft_path` / `geometry`,
      and `anchors.TF_BIT_L3`. **Done 2026-09-07.**
- [~] **2.3** Promote `DoorHopSpec` → `dungeon/door_hop.py`; convert the ~33
      one-room modules to rows (**~7,500 LOC**). **Engine promoted and L8 done
      2026-09-08** — `dungeon/door_hop.py` holds `DoorHopSpec`/`DoorHopController`
      (L6-isms injected: `level`, `success_fn`, `record_fn`, band ys) plus
      `RoomHopSpec`/`RoomHopController` for the one-frame cardinal step-hop
      family. `level6/door_hop.py` 509 → 175; L8 `path` 741 → 481, `stairs`
      173 → 102, `triforce` 387 → 336, `cellar` 168 → 127, `passage` 200 → 139.
      Proven by a differential harness against the pre-change modules over
      ~40k synthetic frames per controller (actions, reasons, notes, reports):
      0 mismatches. **Left as novel:** `gleeok_entry.py` (drives a
      `BombWallController` sub-machine), `north_column.py` (multi-room),
      `triforce.py`'s two latch controllers, the fail-closed stubs.
      **Remaining: L7, L4, L6, L9, L1–L3 — all now pure subtraction.** Row conversion preserves the
      hard-won geometry verbatim — it is mechanical, not a rewrite. Level order
      by safety:
      1. **L8** — `path.py:175/264/353/436` are **85–89% line-identical**;
         four controllers differing only in `(origin, dest, direction, step_fn,
         spec_id, done_reason)`. Plus `cellar.py`, `passage.py`, `stairs.py`,
         `triforce.py`, `gleeok_entry.py`, `north_column.py`.
      2. **L7** — `west.py:167 _WestHop` base already exists with 8 leaves
         (~612 LOC → 8 rows). Plus `cellar.py`, `shard.py`, `pre_boss.py` (46
         LOC of pure config).
      3. **L4** — largest, no base yet, zero `HopController` use: `north30.py`
         (110 LOC to say "align x≈120, hold UP, stop at 0x30"), `west31.py`,
         `exit60.py`, `map21.py`, `keyup20.py`, `bomb11.py`, `key01.py`,
         `clear12.py`, `mappick.py`.
      4. **L6** — `cellar08.py`, `inland29.py`, `north39.py`, `stairs18.py`,
         `exit75.py`, `rod.py`; `room19.py:362-394` already has *five factories
         over one controller* — the pattern found and not generalized.
      5. **L9** — the 12 near-empty `prefix.py` leaves (`:342,:358,:367,:696,
         :709,:722` are 9–15 LOC each). Keep the novel stairs/east/arrows kernels.
      6. **L1/L2/L3/L5** — `bow.py`, `bow_rejoin.py`, `bow_pickup.py` (use the
         existing `dungeon/hop_controller.py:66 CellarCross`), `clear.py`,
         `enter_1e.py`, `clear5b.py`, `west_path.py` (`tf_path.py` deleted in 2.4).
      **Keep as novel:** `level6/stairs3a_warp.py`, `level6/stairs09.py`,
      `level7/hungry.py`, `level7/pond.py`, `level7/warp.py`,
      `level7/digdogger.py` (whistle-shrink), `level1/bow_cellar.py` push seq,
      `level5/cellar_path.py` block-stairs.
      **Trap:** `level4/gleeok13.py` is *not* a Gleeok fight — it is the
      0x12→0x13 entry hop. Do not merge by filename.
- [x] **2.4** Fold the `level5/path.py` facade. **DONE 2026-09-08** — the six
      facade modules 3,044 → 2,319 LOC (−725); −765 counting `level5/spine.py`,
      `level5/dungeon.py` and `door_graph/level5_exits.py`.
      `_LAZY_EXPORTS` / `__getattr__` gone; callers import the owning module. Three clone families became rows:
      `Level5NavSpec` (0x66 return / 0x77 east key), `WestLeaveSpec`
      (0x27/0x26/0x25 west leaves), `BombWallSpec` (bomb west-66 / west-65 /
      east-65). One `walk_axis(stall_limit=, done=)` replaced the three
      copy-pasted axis walkers. Deleted facade-only dead code: `tf_path.py`
      (whole file), the 0x27/0x56 nav controller clones, `should_force_keys_zero`,
      `walk_east_from_65`, `cellar_07_to_64`, `take_center_stairs_06`, the two
      `exit_whistle_04` aliases. Shared L5 room ids + bomb stands moved to
      `level5/dungeon.py`, which also breaks the latent
      `whistle_path → level3.dungeon → door_graph → level5_exits → whistle_path`
      import cycle.
- [ ] **2.5** Migrate the **53 hand-rolled phase machines** (38 files) onto
      `dungeon/hop_controller.py:86 HopController`. Two generations of the same
      skeleton coexist; the migration stopped at the L1–L5 boundary. **~2,000 LOC.**
- [ ] **2.6** One `BossSpec` table over the `dungeon/gleeok.py` primitives.
      There are **three** Gleeok implementations: `level6/gleeok18.py:65` and
      `level8/gleeok.py:70` (structurally identical, deltas are body type
      `0x44`/`0x45`, clip Y, dodge order, a heart tail) and
      `level4/boss_combat.py:148`, which doesn't import `dungeon/gleeok.py` at
      all and is a 437-LOC env-stepping loop. Also fold the env-loop boss
      modules `level2/boss_combat.py`, `level3/boss_combat.py` and the
      `Level3BossCombatMixin:160` mixin cluster. Model to copy:
      `level7/aquamentus.py:1-3`, which reuses `level1.finish` and adds 137 LOC.
      **~1,800 LOC out, but HIGH risk** — fights are timing-fragile, ROM eval
      per boss. Do this last.
- [x] **2.7** One `SPINE_LEVELS` row table. **Done 2026-09-08:** nine
      `SpineLevel` rows and one loop; `_through_for_predecessor` is now
      `SpineLevel.target()` driven by the `handoff` column and applied
      uniformly (it used to fire only for L6/L7/L8). `SPINE_THROUGH` and
      `SPINE_STOPS` derive from the rows. L1/L2/L3 became rows with zero
      level-module edits. Gate: the 93-id through catalog and each resolved
      stop name are byte-identical before/after and now pinned by a test.
      **`route/chain.py` is kept** — the second dispatcher 2.7 named was
      `route/composer.py`, deleted in 1.5; what remains is the Composer's
      engine layer (`run_controller_stage`, `ControllerStageResult`,
      `boot_to_ready`, `run_natural_to_milestone`) with 11 live importers
      including the protected A/B loop. Its `run_natural_to_milestone` +
      `_MILESTONE_ORDER` 5-entry ladder is a real residual: fold into L1
      `SpineHop` rows when L1 is touched, and send the rest to
      `retro_harness.spine` with 2.9.
      *(original audit)* `spine/survival.py:463` is an
      imperative ladder: L1/L2 inline, then seven near-identical
      `continue_levelN_spine(...)` calls at `:620,631,643,655,667,681`.
      `_through_for_predecessor:441` exists only to paper over per-level
      dispatch. Meanwhile `route/chain.py` + `route/composer.py:19` are a
      **parallel** dispatcher — CODING_STANDARDS: "A sitting that needs a second
      dispatcher has not finished." Only 56 `SpineHop` rows exist game-wide;
      L1 and L2 have **zero**. **~900 LOC.**
- [ ] **2.8** Add `ReadySpec`/`EntrySpec` fields to `overworld/path.py:50
      OverworldPathController`, removing the `_at_stop` / `_after_hops` /
      `_before_play` / `report` overrides present in **all 10** subclasses.
      Migrate `overworld/nav.py:82` (L1, the last hand-rolled predecessor,
      288 LOC) onto the engine. Adopt `overworld/stitch.py` beyond its current
      2-of-9 levels — or delete it as a third generation started before the
      second finished. **~800 LOC.**
- [ ] **2.9** *(reuse seam — enables Zelda 3 / SMZ3 later, builds nothing now)*
      Promote to `retro_harness`, following the `room_timer.py` precedent
      (generic engine in harness, game constants injected):
      - `walk/physics.py` (183) → `retro_harness.occupancy`. Only Zelda-isms are
        `DEFAULT_BOUNDS:34` and `WALK_SPEED=1`, both already constructor args.
      - `door_graph/core.py` (370) → `retro_harness.adventure.door_graph`.
        Fully abstract already.
      - `dungeon/hop_controller.py` — **the highest-value unblock**. Its only
        coupling is `:13 from zelda_i.ram import PLAY_MODE, ZeldaSnapshot`;
        make it `Generic[SnapT]` + a `GameModes` dataclass. The Zelda-1 facts
        are only `DEATH_MODE=17`, `CELLAR_MODE=9`, `WAIT_SCROLL` at `:20-24`.
      - Split `spine/hops.py`: `SpineHop` + `attach_hops` (~60 LOC, entirely
        generic) → `retro_harness.spine`; `play_ready` (Zelda-1 inventory) stays
        as `zelda_i/spine/ready.py`.
      Do 2.9 **before** 2.5 so the phase-machine migration lands on the generic base.

---

# Phase 3 — Speed & watchability (target 98 min → ~72 min)

Ordered by frames saved. Combined realistic target **70,000–95,000 f (20–26 min)**.

- [x] **3.1** **Kill the UP mash.** `_patrol` at a vertex idles instead of
      walking into the north wall. Open-floor `engage_distance` still gates
      chase vs patrol. Occupancy mazes follow a BFS path from any distance;
      no-path + far patrols (do not greedy through water). No-path + inside
      the room cap still closes — occupancy may have miss-blocked the
      enemy pixel. Parked Wallmasters are not chase targets. Open-floor Gels
      (0x42/0x43) raise the room cap to 160. Unconditional greedy chase is
      not this sitting.
- [ ] **3.2** **Overworld align shuffle** — **9,000–13,000 f**.
      `overworld/common.py:151-220 align_and_push` re-aligns x, then y, then
      pushes after *every* scroll. `enter_level3` = 10,112 f for 17 hops =
      595 f/screen against a ~300 f floor. Same for `enter_level2/5/6`,
      `level8_post_l7_to_bush`, `level9_post_l8_overworld` (~44k f total).
- [ ] **3.3** **L5 imperative → polled controller** — **8,000–12,000 f**.
      `level5/whistle_path.py` (809 LOC) is 32 blind `idle(env, assist, n)`
      calls plus `walk_axis(..., max_f=400)` chains. L5 is 42,949 f across 5
      stages — the worst frames-per-stage in the run. Pairs naturally with 2.4.
- [ ] **3.4** **Hold travel direction through scrolls** — **5,000–10,000 f**.
      `dungeon/hop_controller.py:114-115` defaults `scroll_action` to
      `nes_idle_action()` during modes 2/3/4/6/7/10/16; `*scroll*` reasons total
      5,104 f. Holding the direction lands Link deeper into the next room,
      removing the re-walk from the door plane. Several controllers already
      override this (`level7/hungry.py:344`, `level6/exit75.py`) — make it the default.
- [ ] **3.5** **Kill the blind sleeps** — **7,000–8,500 f**, and it is *all*
      visible dead air (Link at a dead stop):
      `level6/path.py:561` SETTLE_18=512 · `level6/room19.py:221` 160×5=800 ·
      `level8/suffix.py:111` CELLAR_2F=600 (a *floor*, `arrived()` cannot
      return early) · `overworld/white_sword.py:61` DIALOG=300 blind (vs the
      *polled* `overworld/sword_cave.py:38-39` which used 35 — copy that
      pattern) · `dungeon/bomb_wall.py:24` BLOW=100 × ~10 walls ·
      `level7/pond.py:104` + `level7/digdogger.py:82` BLOW_WAIT=240 each ·
      `level1/finish.py:141-148` WAIT_HINT=180 · `level9/overworld.py:481,:1002`
      blast=180 · `level4/exit60.py:185-188` 150 ·
      `dungeon/pause_select.py:38 OPEN_SETTLE=20` ×14 sites ·
      L5/L3 `idle(env,…)` glue ~1,765 f.
      Also delete `level9/overworld.py:328-331 spectacle_rock_screenshot_hold`
      — 30 f of a recon artifact left in the shipping path.
- [ ] **3.6** **Boss cooldowns: move instead of freezing** — **4,000–7,000 f**,
      and it fixes the *worst-looking moment in the entire run*.
      `level9/ganon.py:161` and `level9/patra.py:104` return `nes_idle_action()`
      during attack cooldown: **Ganon is motionless for 2,901 of 4,200 f (69%)**,
      final Patra 1,067 of 2,189 (49%). Replace with a dodge/approach step.
      Also tighten `level9/patra.py:96-102 align_south_*` (±4 px tolerance
      causing 4,916 f of visible jitter).
- [ ] **3.7** **Cut route backtracks** — **4,000–8,000 f** of 26,569 f (7.5%)
      where the viewer watches the same corridors two and three times.
      Worst: `level8_post_l7_to_bush` 6,080, `level9_post_l8_overworld` 4,184,
      `level7_room4a_return` 2,524, `level7_recorder_warp` 1,899.
      L1 `exit42` no longer walks the old-man hint (skipped 2026-09-08).
      Remaining backtracks are L7–L9.
- [ ] **3.8** **Never stand still forever.** `overworld/common.py:131-149
      unstick_wiggle` does 16 frames of wiggle then `nes_idle_action()`
      *indefinitely* (`:146`, `reset_after` deliberately ignored), triggered at
      `stuck > 50` across 10 sites. `walk/physics.py:134` has the same terminal
      behaviour ("No path → stand (no hunt)") and logged 3,183 misses this run.
      Give both a real escape ladder. Also `level7/hungry.py:281-282 spawn_wait`
      freezes Link until an enemy deigns to spawn — `level7_room38_up` takes
      164 s to move one room north.
- [ ] **3.9** Tighten `max_frames`. **Zero saving on a green run** (15%
      utilization; the loop breaks on done at `route/chain.py:324-336`) — this
      is purely failure-detection latency. `level3_boss_tf` budgets 120,000 f
      and uses 8%; a stall there costs 30+ min before the run gives up.

---

# Phase 4 — Overworld combat helpers

Today the entire overworld policy is `overworld/common.py:47-72`: swing if
something is in the sword box **along the travel direction**, else walk. No
retreat, no dodge, no reposition, no health check, no B-item. The only failure
branch is death (`overworld/path.py:401-402`).

- [x] **4.1** **Wire in `dungeon/behaviors.py`.** `walk_or_swing` builds
      `engagement_hint` for the nearest threat and uses `hint.face` when that
      facing has a hitbox/contact. Dungeon engine still does not import hints
      (cycle: behaviors → engine `AliveRule`).
- [x] **4.2** **Catalog the overworld enemies.** Jump table
      (`aldonunez/zelda1-disassembly` `UpdateObject_JumpTable`): Lynel 0x01/02,
      Moblin 0x03/04, Octorok 0x07–0x0A, Tektite 0x0D/0E, Leever 0x0F/10,
      Zora 0x11, Peahat 0x1A, Armos 0x1E, Ghini 0x21/22, rock 0x53. The
      `level8/overworld.py` "Octoroks (type 0x03)" comment is Blue Moblin.
      Peahat flying / Armos statue: notes only (RAM not on `ZeldaObject`).
- [x] **4.3** **Face contact-range off-axis threats.** `walk_or_swing` turns
      only when a body is in the contact guard (else hops never left spawn).
      Far side hitboxes keep the travel direction. `THREAT_RADIUS` remains
      approach-only.
- [x] **4.4** **Dodge projectiles.** `overworld/common.py` now calls
      `projectile_threats` through `overworld_projectiles` +
      `answer_projectile`, sidestepping perpendicular to travel before the
      terminal walk in `walk_or_swing` (so every `_swing` / `align_and_push`
      call site inherits it). Shots carry `hp=0`, so
      `overworld_threat_objects` could never see them — they need their own
      collector. The sidestep flips side at a screen edge.
- [x] **4.5** **Detect knockback.** `common.py track_knockback` charges the
      stuck counter `KNOCKBACK_STUCK_PENALTY=20` per health-byte drop, so three
      hits inside one hop reach the 50-frame unstick bar that zero-movement
      tracking never saw. `OverworldPathController` counts `hits_taken` (reset
      on hop advance, reported for diagnostics).
- [x] **4.6** **Clean the threat set.** `overworld_threat_objects` drops
      type 0x60 rupee drops and `hp<=0`. Contact guard no longer slashes
      pickups/corpses.
- [ ] **4.7** **Use B-items on the overworld.** Bow, Magical Rod (owned from
      L6) and Magical Boomerang (owned from L2, stuns nearly every OW enemy)
      are never selected outside dungeons. All machinery exists
      (`dungeon/pause_select.py`, `level9/hops.py:149`).
- [x] **4.8** **Shield awareness.** `magic_shield` is on `ZeldaSnapshot` now,
      and `behaviors.shield_blocks` splits the shots the small shield eats
      (rock / Moblin arrow / Lynel sword shot) from the ones needing the
      Magical Shield (fireball, Manhandla residual); a Goriya boomerang is
      blockable by neither. `answer_projectile` (folded together with 4.4)
      keeps walking a blockable lane with the A pulse suppressed — walking
      *is* facing — and only sidesteps what the shield cannot eat.
- [x] **4.9** **Low-heart behavior.** `OverworldPathController._farm_action`
      diverts into `HeartFarmController` on the current screen when
      `filled_hearts < farm_below_hearts`, then hands the hop back
      (`BAND_SWEEP_WAYPOINTS` is the screen-agnostic patrol; every spine screen
      has the y≈141 corridor the hop crossed). Fail-soft both ways:
      the farm quits on its own timeout or on leaving the screen, and
      `max_farm_attempts` stops a farm/starve loop. `farm_below_hearts`
      defaults to 0, so the hook is inert until the health assist comes off.

---

# Phase 5 — 100% items

The run currently reaches the endgame with **10 of 16 heart containers**
(`level9/hops.py:99-102`) — which is why the White Sword detour has to be
spliced into L9 entry and why the Magical Sword (12 HC) has never been attempted.

- [ ] **5.1** Fix the stale rows in `route/treasures.py` — Red Candle
      (`:171-182`), Magical Key (`:207-218`) and Silver Arrows (`:231-242`) are
      marked unrouted but are live on the spine. The table is the 100% ledger;
      it must be true before it can drive anything.
- [ ] **5.2** Cheap wins first:
      - **Power Bracelet** (Armos, screen `0x24`) — **the spine already walks
        `0x24`** (`level6/overworld.py:135`, `level7/overworld.py:76`).
      - **Wooden Boomerang** (L1 `0x44`) — `treasures.py:88-98` `LIVE_SKIPPED`,
        the only "just add the room" item on the list.
      - **Book of Magic** (L8 staircase) — `L8_THROUGH` has no book stop.
- [ ] **5.3** **Generalize overworld secrets.** Today there are three bespoke
      one-offs: bomb only on `0x05` (`level9/overworld.py:476,:996`), candle
      burn only on `0x6D` (`level8/overworld.py:676`), push-rock nowhere.
      `dungeon/bomb_wall.py` is dungeon-only by construction (takes a
      `DungeonRoomSpec`, `level: int = 2`). Everything needed for a general
      burn sweep already exists — `ADDR_CANDLE_USED = 0x0513`
      (`level8/overworld.py:188`) and `B_ITEM_CANDLE = 0x04` (`:189`); only the
      target table is missing.
- [ ] **5.4** **Heart containers.** No general HC pickup step exists — only L2
      has an explicit one (`level2/boss_tf.py:221-231`). Add the raft heart
      (`0x3F` → island `0x2F`) and the ladder heart (coast `0x5F`) — the
      Stepladder is owned from L4 and has **never been used on the overworld**.
      Then the bomb/burn secret-cave HCs, which are entirely absent from the
      codebase (no screen ids, no notes).
- [ ] **5.5** Long tail, mostly needing rupees and the Phase 6 farm: Letter →
      Potion, Blue Ring (250R), Magical Shield, both bomb upgrades (L5/L7 old
      men, 100R each), Magical Sword (grave `0x21`, needs 12 HC and a
      gravestone-push controller that does not exist; `0x21` is currently
      unreachable — `level7/overworld.py:91-95`).
- [ ] **5.6** **Extend the overworld graph.** Only **63 of 128 screens (49%)**
      are walked. Worse, `overworld/graph.py:258-302
      build_overworld_grid_graph` emits all 128 nodes with pure grid adjacency
      and **models no walls**, so any planner over it produces impossible
      routes. Fix that before adding screens. Known-sealed and already
      live-falsified: `0x67`, `0x79`, `0x0B` west, Lost Hills `0x1B` (wraps to
      itself all four directions), `0x09`, `0x22`, `0x41` east, `0x5E`
      east/south, `0x4B` north trap.

---

# Phase 6 — Strip the assists → Clean credits (`rr-ps7` → `rr-npv`)

**Structural blocker first:** `spine/survival.py:482-483` *raises* when `assist
is None`, and `scripts/run_survival_spine.py:54` hardcodes
`UnlimitedHealthAssist(enabled=True)`. A Clean spine run is currently
impossible without a code change. Note also `runner.py:46-52` makes
`--infinite-life` **default True** for every script using `add_common_args`.

Strip order is dependency-driven, cheapest first:

- [ ] **6.1** Dead pokes — `poke_link_position`, `poke_candle_for_recon`. Zero
      call sites. *(Already covered in 1.5.)*
- [ ] **6.2** **L1 key top-up** (1 write). `ROOM_72_SPEC` + west-door hops exist.
      Splice onto the default tape red: `to_entrance` from the clear53 leftover
      died in 0x63 (diamond). Poke at `backtrack44` stays. Independent of
      everything else.
- [ ] **6.3** **Rupee farm** — do this *before* its consumers; it is the shared
      unlock. `overworld/rupee_farm.py:73 RupeeFarmController` is finished,
      write-free, fail-closed, and wired nowhere on the spine. It is the sole
      prerequisite for both the 80R arrow buy and the 60R Bait. Wiring costs one
      `before=` hook per shop stage. Then delete `SPINE_L7_RUPEE_RETOPUP`.
- [ ] **6.4** **Wooden arrows** (`rr-wabn`). Depends on 6.3. The buy controller
      already works (`level1/arrow_shop.py`); the only red leg on `--through
      level1-arrows` is the farm (`docs/plan.md:250`, rupees 9→10), so 6.3
      likely greens it outright. Removes the `ADDR_ARROWS`+`$0656` pair at
      `level6/gohma.py:228`.
- [ ] **6.5** **Bait / Food** (`rr-8t4.4`). Needs 6.3's rupees but is *not*
      unlocked by it — the blocker is unmapped geometry (the post-L6 pocket has
      no southward outlet to shop `0x34`). A route-mapping campaign. Once
      mapped the flip is cheap: `survival=False` at `level7/spine.py:56`, plus
      `shop_geometry_verified=True` on the plan so
      `NaturalBaitPurchaseController` (`level7/entry.py:107`) stops failing closed.
- [ ] **6.6** **Bomb + key counts** (`rr-doua`). Highest volume — 10 bomb, 2 key,
      11 `selected_item` writes across five wiring points — and the **only item
      with no natural replacement anywhere in the tree**: there is no bomb-farm
      module. Needs a new farm plus a per-gate budget proof.
- [ ] **6.7** **`UnlimitedHealthAssist`** (`rr-npv`). Last by a wide margin.
      Requires (a) removing the hard dependency and adding a real flag,
      (b) reverting every `survival=True` combat concession — `tank_hits`
      (`level1/finish.py:761`, `level7/aquamentus.py:93`),
      `ROOM_44_SURVIVAL_SPEC` (`level1/east_dungeon.py:79-91`), widened
      `engage_distance` (`level1/finish.py:735-751`) — and (c) surviving
      **1,117 damage units across 85 rooms** on 10 containers. Phases 3.1 and 4
      are what make this reachable; Phase 5's heart containers are what make it
      survivable.

---

## Sequencing

```
Phase 0  ──▶ Phase 1 (cruft)  ──▶ Phase 2 (consolidate)  ─┬─▶ Phase 6 (Clean)
                                                          │
             Phase 3 (speed)  ────────────────────────────┤
             Phase 4 (OW combat) ──▶ Phase 5 (100%) ──────┘
```

Phase 1 is safe and unblocks everything by shrinking the search space.
Phase 3 and Phase 4 are independent of Phase 2 and can run in parallel lanes —
but per `zelda-parallel-lane-coordination`, lane ownership must be declared and
never `git add -A`.
Phase 6 depends on 3.1 (aggressive combat), 4 (survivable overworld) and 5.4
(heart containers); attempting it earlier measures against a moving target.
