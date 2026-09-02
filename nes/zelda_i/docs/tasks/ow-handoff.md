# Overworld stitch handoff (L1 leave → L9 enter)

**Lane:** overworld glue. Isolation worktree. Did not STATUS-promote.
Did not commit. Did not edit `STATUS.md`, `.beads`, `level6/**`,
`level7/**`, `level8/**`, `level9/**`, or `spine/survival.py`.

Packet: `zelda_i.overworld.stitch.OverworldHandoff`. Same field names as
L8 `PostLevel7Handoff` plus `mode` / `triforce` / `magic_key`. Defaults
`verified=False`, `route_eligible=False`. `complete()` is false until
`verified=True` **and** every measured inventory field is filled. L8
keeps its local dataclass; L7/L9 should consume this shared packet.

---

## Stitch table

| From | Leave (screen, x/y, mode, TF, items) | To mouth | Enter items | Status |
|------|--------------------------------------|----------|-------------|--------|
| L1 | **`0x37`** ~(112,125) mode 5, TF `0x01` | L2 **`0x3C`** | wooden sword; TF1 | **verified** leave + mouth |
| L2 | **`0x3C`** ~(112,125) mode 5, TF `0x03` | L3 **`0x74`** | wooden sword | **verified** leave + mouth |
| L3 | **`0x74`** ~(128,125) mode 5, TF `0x07`, raft=1 | L4 **`0x45`** via dock `0x55` | Raft | **verified** leave + mouth |
| L4 | **`0x45`** island settle 284f; x/y **not packed**, TF `0x0F` | L5 **`0x0B`** | none | **live** leave; mouth verified |
| L5 | **`0x0B`** settle 510f; x/y **not packed**, TF `0x1F`, Whistle earned | L6 **`0x22`** | none | **live** leave; mouth **verified** (`0x22`) |
| L6 | **`0x22` `(112,125)`** mode 5, TF `0x3F`, keys 2 bombs 8 rupees 42 Rod 1 Bow 1 arrows 1, 8 HC full — **measured** post-fanfare engine return (`--through level6-exit` 1/1, `l6_exit_ow.json`). `(112,125)` = the Dragon mouth tile; matches L1/L2/L3 pattern. | L7 pond source **`0x42`**; bait shop source **`0x34`**; live approach through **`0x53` (224,173)** LEFT-inland-before-DOWN | **Whistle** to drain; **Bait** inside (Hungry Goriya) | leave **measured**; mouth **hypothesis**; 0x53 leftover **live fail** |
| L7 | **UNMEASURED** (expect TF `0x7F`, Candle 2, Whistle retained) | L8 bush **`0x6D`** from **`0x5D` south x≈48** | **Candle 2** (from L7). Blue Candle shop `0x5E` is fallback-only | leave **UNMEASURED**; bush screen **verified**; burn **unsolved** |
| L8 | **UNMEASURED** (expect TF `0xFF`, Magic Key, bombs) | L9 Spectacle Rock **`0x05`** bomb left rock | bombs; TF `0xFF`; Magic Key | leave **UNMEASURED**; mouth **source / fixture-live** (entry room `0x76`); not natural post-L8 |

Cumulative TF after clear: L1 `0x01` … L6 `0x3F` … L7 `0x7F` … L8 `0xFF`.
L9 Old Man wants `0xFF`.

Later OW shortcuts (needed, **do not grant**):

| Capability | Screen | Requires | Status |
|------------|--------|----------|--------|
| White sword cave | `0x0A` hyp (region `0x0B` live) | 5 heart containers | source / region live |
| Magical sword grave | `0x21` | 12 HC; push 3rd-from-left middle gravestone | source |
| Bracelet Armos | `0x24` | none; top-right of 10 | source |

---

## Live vs source

**Live / verified (assisted OK, not Clean):**

- L1–L3 post-fanfare OW poses and mouths (`0x37`, `0x3C`, `0x74`).
- L4 island mouth `0x45` / dock `0x55`; L4 settle onto `0x45`.
- L5 mouth `0x0B` / Lost Hills `0x1B`; L5 settle onto `0x0B`.
- L6 entrance **`0x22`** verified (`l6_entry_continuous_v2`); L6 **exit**
  `0x22` `(112,125)` measured (`--through level6-exit`, `l6_exit_ow.json`).
- L7 pond **approach leftover** `0x53` (224,173) — live fail, not pond.
- L8 bush pocket **`0x6D`** (enter only from `0x5D` south @ x≈48). Blue Candle shop `0x5E` live; not on the Red-Candle mainline.
- L9 Spectacle Rock **`0x05`** reached via authentic OW scroll + bomb in fixture recon; entry room `0x76`. Fixture inventory. `route_eligible=false`.

**Source / hypothesis / UNMEASURED:**

- Post-L7 / post-L8 fanfare leftovers. (Post-L6 **measured** — see table.)
- L7 pond `0x42` drain geometry and entry room.
- L7 bait shop `0x34` Armos (top-middle staircase) — **TBD live**.
- L8 bush burn tile / facing / `ADDR_CANDLE_USED` → mode-16 mouth.
- L9 natural walk from the real L8 leftover to the bomb rock.
- White sword cave tile, mag-sword grave push, Bracelet Armos.

**This sitting did not boot a ROM.** `roms/` is absent. No fixture-live bait-shop or Spectacle Rock recon ran. Halt-3-reds unused.

---

## Helpers added

`nes/zelda_i/overworld/stitch.py` (new, ~410 lines):

- `OverworldHandoff` — leftover packet. `complete()` / `mismatch()` /
  `handoff_from_ram()`.
- `MOUTH_STITCHES` — one row per L1 leave → L9 enter.
- `TF_BIT_BY_LEVEL` `0x01`…`0x80`, `CUMULATIVE_TF` through `0xFF`.
- `enter_gate_ok(level, handoff)` — L7 Whistle, L8 Candle 2, L9 TF `0xFF`.
- `inland_then_descend` + `InlandDescendSpec` + `HYP_SCREEN_53_INLAND_DESCEND`
  — y-band LEFT inland, then descend, then travel. Pattern from 0x53
  leftover (224,173). **Not** an L7 pond controller.
- `y_band_travel_hop` — `ScreenHop` with `y_band`, **not** `align_y`
  (align_y would DOWN first and re-hit the 0x53 east-edge trap).

Expected consumer: L7 `_extra_hop_action` on `0x53` (L7 owns
`level7/overworld.py`). Call `inland_then_descend(snap, spec)` then
`_swing(direction, …)`. Do not encode pond hops here.

Did not add a knob to `graph.py` (already 530). Did not grow `path.py`.

---

## Dead beliefs

- ~~Post-L6 leftover is OW `0x22` but UNMEASURED~~ — **measured 2026-09-02**:
  `--through level6-exit` 1/1 returns Link to **`0x22` `(112,125)`** mode 5
  TF `0x3F`. The screen `0x22` assumption was right; the position is the
  Dragon mouth tile, **not** the `(120,221)` south edge the L7 fixture
  guessed. "First L7 hop from `0x22`" is now unblocked.
- **`(112,125)` on `0x22` is NOT a dead spot.** It is the real fanfare
  return (same as L1 `0x37` ~(112,125)). The "mode 16 → dungeon" trap only
  fires on a fresh UP into the mouth, not on emerging onto it; the bait
  prefix's first move (DOWN → `0x32`) walks away from it.
- Current L6 play **`0x09` (56,109)** is the post-L6 leave. It is the
  interior Survival tip (`rr-tne2`). Not a fanfare leftover.
- Direct **DOWN** from `0x53` (224,173) toward y≈189 reaches `0x52`. Live
  fail: east-edge DOWN is blocked. LEFT inland first.
- Start-`0x77` pond hops / L8 bush hops are the post-dungeon approach.
  They are recon-only; replace from the measured leftover.
- Source bait path through **`0x67`**. Live 0x67 is a sealed tree pocket
  (L3).
- Exhausting the L8 burn budget on `0x6D` is entry success. Fail-closed:
  still on 0x6D is failure.
- L8 needs the 60R Blue Candle shop on the mainline. Canonical Candle is
  Red Candle 2 from L7.
- Pond / bush / Spectacle Rock are spine-green. They are not.
- White sword / mag sword / Bracelet may be poked to unlock later OW
  shortcuts. **Do not grant.**
- Do not poke Whistle / Candle / TF for recon claims.

---

## Files changed

| Path | Change |
|------|--------|
| `nes/zelda_i/overworld/stitch.py` | new leftover packet, mouth table, inland-then-descend helper |
| `nes/zelda_i/tests/test_overworld_stitch.py` | complete() / TF bits / L7–L9 gates / 0x53 LEFT-before-DOWN |
| `nes/zelda_i/docs/OVERWORLD_DOORS.md` | stitch section + pointer here |
| `nes/zelda_i/docs/tasks/ow-handoff.md` | this file |

Not changed: `graph.py`, `path.py`, dungeon `overworld.py`, `STATUS.md`,
`.beads`, `spine/survival.py`.

---

## What owners must fill when they measure leave

Call `handoff_from_ram(ram, evidence="live", verified=True)` on the
**settled post-fanfare overworld frame** (mode 5, level 0, not
transitioning). Keep `route_eligible=False`. Integrator promotes.

Every leave packet needs:

| Field | Notes |
|-------|-------|
| `screen`, `mode`, `link_x`, `link_y` | settled OW pose; xy tolerance 4 |
| `triforce` | exact cumulative: L6 leave `0x3F`, L7 leave `0x7F`, L8 leave `0xFF` |
| `keys`, `bombs`, `rupees`, `heart_containers`, `selected_item` | counts, not grants |
| `whistle`, `food`, `rod`, `bow`, `arrows`, `candle` | L7 enter needs whistle; L8 enter needs candle **2** |
| `magic_key` | L9 enter; record even if 0 |
| deaths, assist writes, post-reset state-load count | chapter handoff extras (not on the dataclass) |

Then replace the start-based hop table with hops **from that leftover**
to the next mouth. Natural-entry: segment is not route-ready until it
clears from the real predecessor.

| Owner | Measure | Then |
|-------|---------|------|
| **L6** | Post-TF-`0x20` fanfare OW leftover **measured**: `0x22` `(112,125)` mode 5 TF `0x3F` keys 2 bombs 8 rupees 42 Rod 1 Bow 1 arrows 1, 8 HC full (`--through level6-exit`, `l6_exit_ow.json`). Packet = `MEASURED_POST_L6_EXIT` in `level7/entry.py` (`verified=False` until L7 owner attaches). | Hand the packet to L7. |
| **L7** | Seam **wired** + Phase 1: `MEASURED_POST_L6_EXIT.verified=True` (`--through level6-exit` 2/2, `selected_item=2`). `--through level7-entry` runs power-on → measured L6 exit → `level7_post_l6_overworld` **green** (`0x22→0x25`, 1577f) → fails closed at `level7_bait_purchase` (`bait_shop_geometry_unobserved`). Rupee 42→60 top-up (`SPINE_L7_RUPEE_RETOPUP`) fires. To go green: live-recon `0x25 → bait shop 0x34` geometry + buy policy, then `0x34 → pond 0x42` + Whistle drain + entry room. Do not start from `0x77`. Use `inland_then_descend` on 0x53. After shard, measure post-L7 leave (expect TF `0x7F`, Candle 2, Food consumed). | Hand the packet to L8. Map bait shop `0x34`. |
| **L8** | Consume L7 packet into `PostLevel7Handoff` (copy fields; keep local type). Reach `0x6D` from that leftover through live `0x5C`. Solve burn tile. After shard, measure post-L8 leave (expect TF `0xFF`, Magic Key). | Hand the packet to L9. |
| **L9** | Consume L8 packet. Walk to `0x05`, bomb left rock with natural bombs, settle room `0x76` with TF `0xFF`. Fixture `*ReconFixture` stays `route_eligible=false` until recomposed. | |

`complete()` stays false on a pose-only row. Filling screen/xy without
inventory still fails closed.

Did not STATUS-promote. Did not claim pond / bush / rock as spine-green.
