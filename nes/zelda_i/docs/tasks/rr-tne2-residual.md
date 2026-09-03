# Residual — rr-tne2 L6 power-on recompose to TF 0x20 (closed, historical)

**`rr-tne2` is closed.** This file is historical — kept for evidence, not a
live target. Current living residual: `docs/tasks/l7-handoff.md` (`rr-8t4.2`,
dest `0x6B` GORIYA_HINT).

**Status:** Gohma reactive kill **GREEN** 2026-09-02.
- `--through level6-gohma` **1/1** (play `0x1C`, body gone, TF `0x1F`,
  keys 2, rupees 43→42 = one arrow, `set_state=0`).
- `--through level6` **1/1** acceptance: mode 18, room `0x0C`, TF `0x1F→0x3F`
  (shard `0x20` claimed), keys 2, bombs 8, health `0x77`, `set_state=0`,
  `deaths=0`. Gohma stage: 1 pulse, connect f117.
- Lab (`gohma_lab.py`): kills f114, 1 pulse, from `GohmaEntryLive` pin.

## L6 exit → overworld — GREEN 2026-09-02 (`--through level6-exit`)

`--through level6-exit` **1/1** (`l6_exit_ow.json`, 213,420f): after the
shard fanfare the engine auto-returns Link to **OW `0x22` `(112,125)`**
(the Dragon mouth tile, standing on it in play mode 5, not transitioning).
Final: TF `0x3F`, keys 2, bombs 8, rupees 42, Rod 1, Bow 1, arrows 1,
health `0x77` (8 containers), `set_state=0`, `deaths=0`. Exit stage is a
598f idle through fanfare (mode 18) → warp; no walking, no RAM write.
`Level6ExitController` in `level6/finish.py`.

**Return position `(112,125)` matches L1/L2/L3** in `ow-handoff.md` (L1
leave `0x37` ~(112,125), L2 `0x3C` ~(112,125)) — the post-Triforce engine
return is always the dungeon-entrance tile. The Level 7 route's
`HYPOTHESIZED_POST_L6_EXIT` `(120,221)` came from the dev-poke `L6Probe_22`
recon fixture, **not** a real fanfare; and its "`(112,125)` is a dead
belief / mode 16 → dungeon" note is wrong for the *return* (that trap only
fires when walking UP into the mouth, not when emerging onto it).

`test_level6_gohma.py` 10/10. Bead `rr-tne2` + `rr-17co` stay
`in_progress` — close is the planner's call (needs the Phase-4 AuditedEnv
sign-off; the spine already wraps `AuditedEnv`, `set_state=0`). No STATUS.
No push.

## L7 seam wired into the main spine — 2026-09-02

`zelda_i.spine.survival` now imports `continue_level7_spine`; `L7_THROUGH`
(`level7-entry`, `level7-red-candle`, `level7`) is appended to
`SPINE_THROUGH`. For an L7 target the L6 suffix is driven to `level6-exit`
(measured OW `0x22` `(112,125)` TF `0x3F`) and L7 continues from there with
`MEASURED_POST_L6_EXIT` as the handoff. Chapter controllers stay fail-closed:
`--through level7-entry` runs the full power-on tape through the L6 fanfare
exit, then stops at `level7_post_l6_overworld`.

### L7 refactor — 2026-09-02 (Phase 0)

`level7/` cleaned to the seam contract (1844 → 1668 LOC):

- `PostLevel6Handoff` (L7-local dup) **deleted**; `MEASURED_POST_L6_EXIT` is
  now the shared `zelda_i.overworld.stitch.OverworldHandoff` packet
  (`verified=False`). Reason string `post_l6_handoff_unmeasured` →
  `handoff_unmeasured`.
- Fabricated `POST_L6_EXIT_LOADOUT` + `apply_post_l6_exit_pokes` +
  `HYPOTHESIZED_POST_L6_EXIT` (White Sword / 80R / keys 3 poke) **deleted** —
  contradicted the measured leave and violated the no-poke rule.
- `PostLevel6OverworldController` now carries the fixture-live bait prefix
  `POST_L6_TO_BAIT_HOPS` as default hops (ready the moment the handoff
  verifies) + the L6-cave-mouth re-entry refusals (absorbed from
  `OverworldToBaitShopController`, which stays as ungated recon).
- Dead source tables `LEVEL7_BAIT_SHOP_HOPS` / `LEVEL7_POND_FROM_SHOP_HOPS`
  **deleted**. `hops.py` stage builders de-duped.
- `test_level7_hops.py` rewritten for `OverworldHandoff`; `test_level7_*`,
  `test_survival_spine`, `test_spine_catalog`, `test_overworld_stitch` green
  (411 pass; the 2 `test_level2_spine` fails + 2 collection errors in
  `test_level3_spine` / `test_level6_overworld` are pre-existing, unrelated).

Deferred to Phase 1: consume `stitch.inland_then_descend` in the pond `0x53`
micro; documented Survival rupee top-up for the Bait buy; move the
`0x77`-start pond recon out of `overworld.py`.

Verified 1/1 power-on: `--through level7-entry --no-video --trials 1`
(`l7_refactor_verify.json`) — `ok=False failed=level7_post_l6_overworld`,
`set_state=0`, room `0x22` `(112,125)` TF `0x3F` keys 2 bombs 8 rupees 42.
Green stages through `level6_exit_ow` (598f) → `level7_post_l6_overworld`
fails closed in 1f (`handoff_unmeasured`, `evidence=measured-level6-exit-1of1`,
`writes=0`). Identical outcome to the pre-refactor baseline.

### L7 Phase 1 — handoff verified + bait prefix green — 2026-09-02

**Item 1 (keystone).** Re-measured `--through level6-exit`
(`l7p1_l6exit.json`, 2/2 with `l6_exit_ow.json`, byte-identical). `spine_final_fields`
now takes `ram` and captures the B-slot / Whistle / Food / Candle bytes:
measured `selected_item=2` (arrows, from the Gohma kill), `whistle=1`,
`food=0`, `candle=0`, 8 HC full. `MEASURED_POST_L6_EXIT.verified=True`,
`evidence="measured-level6-exit-2of2"`, `route_eligible` still `False`.

The measured leave stands **on** the `0x22` mouth tile `(112,125)`, so
`PostLevel6OverworldController._reentry_refusal` was refusing on frame 1
(`l6_cave_mouth`). Fixed with a `_left_mouth` latch: the position refusal
(now `l6_cave_mouth_reentry`) only arms after Link has stepped off the
mouth once. `mode==16` cave-enter and `level==6` refusals are unchanged.

**Item 2.** `SPINE_L7_RUPEE_RETOPUP` = `{"level7_bait_purchase"}`. New
`poke_rupees` / `apply_owned_inventory(rupees=)` / `topup_owned_rupees`;
`rupees` added to `OWNED_INVENTORY_FIELDS` and threaded as `rupee_retopup`
through `_run_stages`. Tops the owned count 42→60 before the Bait stage.
Logged in `ASSIST_CONTRACT.md`. `progression_writes=capacity_writes=0`.

Power-on `--through level7-entry` (`l7p1_entry_v2.json`, `set_state=0`,
`deaths=0`):

- `level6_exit_ow` green (598f) → `level7_post_l6_overworld` **green**
  (1577f): walks `hop_0_32 → hop_1_33 → hop_2_23 → hop_3_24 → hop_4_25 →
  path_complete`, phase DONE, `stuck=0`, `writes=0`. First continuous
  power-on walk of the bait prefix from the *measured* leave.
- rupee top-up fires (42→60, one `rupees` write).
- fails closed at `level7_bait_purchase` → **`bait_shop_geometry_unobserved`**
  (was `bait_need_60_rupees`). Final: room `0x25` `(0,141)` TF `0x3F`
  keys 2 bombs 8 rupees 60.

Tests: 413 pass (`tests/`, minus the 2 pre-existing `test_level2_spine`
fails + 2 pre-existing collection errors in `test_level3_spine` /
`test_level6_overworld`). `test_level7_hops` rewritten for the verified
handoff + mouth latch; `test_dungeon_ops` gains rupee-topup coverage.

**Next (Phase 1 item 5, segment 1):** live recon `0x25 (0,141)` → bait
shop `0x34`, observe the Armos-staircase shop geometry, fill
`BaitPurchasePlan.shop_cave_xy` / `shop_geometry_verified`. Then pond
`0x42` drain + entry room. Still deferred: `stitch.inland_then_descend`
in the `0x53` pond micro (item 3); move the `0x77`-start pond recon
controller out of `overworld.py` (item 4).

## The Gohma kill — SOLVED 2026-09-02

**Root cause of every red (v1–v4 + 3 reactive passes): Link fired the arrow
sideways.** `nes_action("UP", "B")` in a single frame straight off a
sideways strafe does not flip Link's facing in time, so the arrow leaves
EAST/WEST along `y≈165` and never reaches Gohma. `ghp` never moved because
no arrow ever arrived. Fix: emit a bare `UP` (`face_up`) frame to set
facing `0x08`, then `UP+B` the next frame.

Secondary factors, all now handled:
- **Eye blink.** RAM `0x03C7` = `0xC0` for a ~17f closed blink, lower
  (`0x70`/`0x60`/`0x58`…) while open, ~65f cycle. An arrow only damages
  Gohma while the eye is open. Any fixed firing cadence lands near ~64f
  (arrow flight + bounce-fall) which aliases onto the blink → 100% misses.
  Policy fires only on the **rising edge** (`0x03C7` just left `0xC0`,
  within `EYE_EDGE_WINDOW`), so the arrow lands mid-open.
- **Doorway.** Enters `(120,205)` wedged in the tile-118 notch — LEFT/RIGHT
  do nothing. Climb straight UP to `STAND_Y=162` first.
- **Gohma's own projectile** is type `0x56`, present most frames — never
  used as an "arrow on screen" gate.
- Movement: track `body.x` with a light velocity lead, `FIRE_TOL=8`
  commit window (recompose hit at `dx=8`), knockback recovery by re-climb.

Live: kills in ~114f with **one** arrow, from the power-on walked warp.
`nes/zelda_i/level6/gohma.py` rewritten; `test_level6_gohma.py` 10/10.

### Fast iteration (per user, 2026-09-02)

`nes/zelda_i/scripts/gohma_lab.py --pin` builds `GohmaEntryLive.state` from
a real power-on `--through level6-north2c`; `gohma_lab.py --tag X` runs only
the fight (~15s). NOT an isolated BFS — real walked-warp RNG phase. Re-pin
after upstream changes; re-validate full power-on `--through level6-gohma` +
`--through level6` every ~10 iterations.

## History — the Gohma blocker as first diagnosed 2026-09-02

**Not a code regression with a revert.** The `rr-17co` warp rewrite
(position-poke → CheckWarp south-band walk-on, which is the wanted change)
moves Gohma-room entry **+223 global frames** later:

| hop | poked warp (`l6_gohma_recompose`) | walked warp (`l6_gohma_column_shot`) | Δ |
|-----|-----------------------------------|--------------------------------------|---|
| stairs3a-warp | base 210711, 115f | base 210711, 290f | +175 |
| cellar08 | 479f | 465f | −14 |
| south1d | 258f | 258f | 0 |
| west2d | 335f | 381f | +46 |
| north2c | 308f | 324f | +16 |
| **gohma entry (frame_base)** | **212206** | **212429** | **+223** |

**Verified:**
- Gohma **spawn pose is byte-identical** on both entries: f1 pre-hatch
  type `0x34` gx=80 gy=93 ghp=0 gst=0; f2 spawn gx=128 gy=112 ghp=32 gst=0.
- On the walked warp Gohma **strafes right immediately**: gx 128→153 over
  the first 54f, monotone, ~0.5 px/f (every tape: column_shot, spawn_shot,
  stand165, after_walk). `gy` stays 112, `ghp` stays 32 forever.
- `l6_gohma_recompose` (poked warp, **1/1 in 54f**) fired ONE arrow up the
  x=120 column ~f28 during Link's walk-up and it connected in the
  spawn-open eye window. That kill needed Gohma parked near gx=128; the
  +223-frame RNG phase on the walked warp no longer parks it there.
- Wooden arrow travel up the column is slow (~2.2 px/f: recompose shot ~f28
  from y≈170, kill f54). By the time a column arrow reaches gy=112 Gohma
  has drifted ~20–25 px right of x=120 → every column/mouth shot misses.
- v4 "mouth shot" is additionally broken: firing UP+B from y=205 stands on
  door **tile 118**, which eats the arrow — `rupees` does not even
  decrement (43→43). Need to be ≥ ~4 px inland (y ≤ ~201) for B to fire.
- `ZeldaObject.state` (RAM `0x00AC+slot`) reads **0** for Gohma at all
  times — it is NOT the eye-open field. Eye-open address is unknown; needs
  a live probe. Bow `0x065A`, arrows `0x0659`, rupees are ammo.

**Dead beliefs (superseded — the real cause was sideways arrows, above):**
- Any fixed-column / mouth shot (v1–v4): aim was never the problem; the
  arrow left sideways.
- Any fixed firing cadence (`stand165` spray, `SHOT_PERIOD` detune):
  aliases onto the 65f eye blink — every arrow arrives shut.
- `ZeldaObject.state` (`0x00AC+slot`) is not the eye field (reads 0). The
  eye is `0x03C7`.
- Isolated BFS states (`BFS_1C/1D/2C/2D`), poked `0x3A` teleport tape pins:
  still banned. `GohmaEntryLive.state` is fine (real walked-warp phase).

## Reset pins (walked-warp leftovers on `l6_gohma_column_shot.json`)

| hop | leave |
|-----|--------|
| stairs3a-warp | mode 9 cellar `0x08` `(208,93)` `position_writes=0` |
| cellar08 | play `0x1D` `(96,157)` keys=3 rupees=43 |
| north2c | play `0x1C` `(120,205)` keys=2 rupees=43 |

`STAIRS3A_DEST` keys spec is stale (wants 4, live 3); xy/mode/room are the
walk. Glance with `zelda_i.screen_glance` before the next Gohma edit.

## L4 Gleeok continuous TF-exit (green 1/1, prior sitting)

`--through level4 --tag l4_gleeok_continuous`: ok, 110926f,
`set_state_count=0`, TF `0x07→0x0F`, fanfare `0x03` `(120,149)`. Spine
Gleeok is `continuous_mode=True`.

## Arrow splice (plan, `rr-wabn`)

`ADDR_ARROWS` is item type; rupees are ammo after the 80R buy. Live 80R
merchant is OW `0x4A`. `--through level1-arrows` red (`l1_arrows` farm
`(63,173)` 9→10R). `poke_wooden_arrows` stays on Gohma. Do not splice into
default L6. Do not close `rr-wabn`.

## Commands

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  nes/zelda_i/tests/test_level6_gohma.py \
  nes/zelda_i/tests/test_level6_stairs3a_warp.py -q

QT_QPA_PLATFORM=offscreen uv run python \
  nes/zelda_i/scripts/run_survival_spine.py \
  --through level6-gohma --no-video --trials 1 --tag <tag>
```

One `--through` per policy change. Stop at the first red. Glance leftover
PNG + JSON samples before the next edit.
