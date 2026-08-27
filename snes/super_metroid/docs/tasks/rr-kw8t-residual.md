## Residual — rr-kw8t Gravity on the Phantoon tip

**Continue (rr-1xc2.3):** Halt-3 is out. After a red, keep going from Next
below (a). Three of the same miss class → new probe angle, same sitting (b).
A new sitting (c) only because context died; it still reads this file.
Do not dest-hop dual this checkbox. Do not reboot the hop because the
count hit three.

**Miss class:** Pocket LEFT-shot. Last leftover `(1113, 1899)` p156 on the
left stairs, Wave crystals **still up** (now to her RIGHT). min_y still
1843 @ f61 (pit jump). Shoulder-R pose 6 slid off the lip. PLM ids never
changed this sitting (`0xc842`/`0xc848`/`0xeedb`/`0xb703` only) — the
0xD080 latch never saw a spawn. Horizontal X with CHARGE_FULL=60 never
released (leftover charge 55).

**Status:** Hop 1 dual-green. Hop 2 **grate_seat PHASE 189f ×2**. Living
tip is `--to phantoon` **195,336f** ×2 (STATUS). Do not STATUS from a pin.
Do not wire hops 1–2 onto `--to phantoon`. Phase dumps are not hop GREEN.

**Pin in:** `scratch/post_ws_basement_to_main.state` (`0xCAF6` ~(1173,1979)
p1 gs=8)
**Goal:** Attic `0xCA52` gs=8. Full hop GREEN is that leave only.

### Already green (do not re-prove)

| Layer | Dual | Leave |
|-------|-----:|-------|
| Power-on Phantoon | **195,336f** ×2 | `0xCC6F` (1240,139) p10 gs=8 |
| Hop 1 Basement → Main | **1,579f** ×2 | `0xCAF6` (1173,1979) p1 gs=8 |
| Hop 2 grate_seat (phase) | **189f** ×2 | `0xCAF6` (1177,1883) p2 gs=8 |

### Hop 2 seams (HARD_ROOM_SPLITS)

Controller kept: `ws_main_climb.py` / `ws_main_shaft.py` / `ws_main_actions.py`
/ `ws_main_ice.py` / `ws_main_grate.py`. Seats: `ws_main_phases.py`.
Six chunks, not one climb. `ws_main_shaft.py` is **500 lines** — split
before a shaft knob.

| Phase | From | Held exit | Status |
|-------|------|-----------|--------|
| 1 pit_shot | pin (1173,1979) | 3-shot, still Main, not Basement | PARTIAL — Wave+Spazer **opens the grate** |
| 2 grate_seat | pin | right hatch-lip ~(1177, 1883) p2 | **PHASE 189f ×2** |
| 3 west_super | lip | y~1675 in shaft, not 0xCDA8 | **RED** living |
| 4 mid_climb | 1675 | y~680 in shaft | not started |
| 5 attic_seat | 680 | ~(1135, ≤160) stand | not started |
| 6 attic_door | door | Attic `0xCA52` gs=8 | hop GREEN only |

Do not open mid_climb while west_super is red. Phase dumps are not hop GREEN.

### Tape scan (take02 policy; 03/04/05 agree)

| Take | Lip fire | First `0xD080` | Morph (not lip) | First y≤1675 |
|------|----------|----------------|-----------------|--------------|
| 02 | (1223,1860) p3 UP+X ~7f then UP | f306 | (1189,1785) p56 | f717 (1099,1675) |
| 03 | (1227,1856) p3 UP+X | f139 | (1189,1785) p56 | f547 |
| 04 | **(1195,1883) p3 UP+X** then RIGHT | f215 | (1214,1801) p56 | f1039 |
| 05 | (1221,1862) p4 UP+X | f296 | (1189,1785) p56 | f974 |

No take fires LEFT+X from the hatch-lip pocket ~(1177, 1883). Take04 is
the only y=1883 fire, from x=1195 — 14px right of the bot seat. Walk
RIGHT from grate_seat sticks at x=1181 (wall). After spawn, take02
LEFT+A from ~(1231,1852) through the hole to morph ~(1189,1785).

### west_super duals — previous sitting (stop: walk/jump-RIGHT)

Leftover always `0xCAF6` (1181, 1883). 3691f timeout.

| # | Knob | Leftover | Why |
|---|------|----------|-----|
| 1 | hold UP+X until spawn | p3 charge 120 | X held; never releases; UP misses pocket blocks |
| 2 | walk RIGHT to take02 column | p1 charge 0 | wall; x stuck 1181 |
| 3 | RIGHT+A onto save-ledge | p1 charge 0 | same leftover; ceiling/wall bonk |

### west_super duals — this sitting (LEFT-shot pocket)

All from hop-1 pin `--stop-at west_super`. Stopped dualing R-aim after
the stairs leftover. Do not 4th-dual walk/jump-RIGHT.

| # | Knob | Leftover | Why |
|---|------|----------|-----|
| 1 | LEFT+X charge-walk | (1169, 1883) p38 face R | walked 1px off LIP_SHOT_X=1170; west_super stole RIGHT |
| 2 | stationary LEFT+X until CHARGE_FULL | (1179, 1883) p2 face L charge **55** | never hit 60, never released; crystals up |
| 3–5 | R+X tap (release at 8) | **(1202, 1854) p77** ×3 | pose 6 not a seat → RIGHT-A to save-column. Peak (1224, 1820) p26. PLM ids **never changed** (no 0xD080) |
| 6 | pose 6 stays a lip seat + R+X | **(1113, 1899) p156** stairs | slid LEFT off the lip; crystals still up (now to her RIGHT). min_y 1843 pit jump |

PNG #6: Samus on the left stairs, hatch below, Wave crystals intact.

### Taught this sitting (kept; last recipe not dualed)

`ws_main_grate.py` / `ws_main_actions.py` / `plm.shot_block_spawns`:

- Hatch-lip x<1216: face LEFT, **X in place**, release at `POCKET_RELEASE_CHARGE=8` (empty buttons). No LEFT walk, no shoulder R.
- Take02 column still UP+X.
- LIP_SHOT_X (1164, 1223); turning 37/38 and aim 5–8 still count as the seat.
- After a 0xD080-family spawn (new slot **or same-slot id change**): LEFT+A through the hole. Morph only at ~(1189, 1785).
- `shot_block_spawns` latches id 0xC842→0xD080 on the same `i` (take02 adds slots 30/31; live Main Shaft reuses).

ROM-free: `uv run pytest snes/super_metroid/tests/test_ws_main_climb.py snes/super_metroid/tests/test_practice_takes.py -q` → 30 passed.

### Next — one seam only

Dual the **stationary X tap** (release at 8, no R, no LEFT walk). Do not
dual shoulder-R. Do not 4th-dual walk/jump-RIGHT. Morph only after a
0xD080-family spawn, and not on the lip. Do not dest-hop. Do not STATUS.
Do not wire `--to phantoon`.

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  snes/super_metroid/scripts/probe/ws_main_climb.py --stop-at west_super --no-video --dual
uv run python snes/super_metroid/scripts/probe/kpdr.py pure ws-main-to-attic \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ws_basement_to_main.state \
  --headed
```

Watch leftover PNG: crystals gone? Then lip_hit should latch. If she
stays on (1177, 1883) with crystals up after firing, the pocket angle
cannot hit — next is a different first-jump (do not retouch grate_seat
until that is proven).

### Non-claims

- Did not STATUS-promote Gravity
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not wire hops 1–2 onto `--to phantoon`
- Did not dual the full hop this wrap
- Did not pin out `post_ws_main_to_attic.state`
- Did not treat a phase dump as hop GREEN
- Did not 4th-dual the RIGHT-into-wall arc
- Did not compose hop 1 + hop 2 onto `--to phantoon`
