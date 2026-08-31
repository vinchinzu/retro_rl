## Residual — rr-kw8t Gravity on the Phantoon tip

**Continue (rr-kw8t):** Main Shaft hop 2, Attic→West Ocean, and West
Ocean→Pancakes are complete. Do not reopen the grate, west-super, mid-climb,
upper-ladder, Attic door, Attic kill-all, or West Ocean maze. The next
boundary is Pancakes→Homing Geemer from
`scratch/post_west_ocean_to_pancakes.state`. The inherited body reaches
Homing Geemer deterministically, but its 120f generic settle is RED at
`0x968F` `(39,139) p11` gs=11 dt=1 after 536f.

**Closed miss:** the 523→443 handoff now latches only after the real
block-clearing jump, then uses the take02 RIGHT+A drift. The upper ladder is
the tape-observed 443→363→267→171→91 sequence. The short Attic shot/jump
cycle centers the blocked first jump before the door opens. Full natural
entry is exact twice; no phase dump is being promoted as hop evidence.

**Status:** Hop 1 dual-green. Hop 2 Main Shaft→Attic is natural-entry
**2,745f** ×2, exact at `0xCA52` `(1133,203) p1` gs=8 dt=0. Its
`attic_seat` phase is **2,044f** ×2 to `(1115,97) p9`; the isolated door
is **296f** ×2 to the same Attic leave. Living tip `--to phantoon`
is **195,336f** ×2 (STATUS). Scratch `--to gravity` is wired (parent
phantoon; hops include s23 Attic→Bowling tapes +
`controller:gravity_collect`). Gravity
collect pin dual **132f** ×2 from
`f022887_enter_0xCE40_0xCE40.state`, items `0x3125` `(127, 135)` p46 gs=8.
s23 hop 08 recording dry-resolve GREEN. Not STATUS. Attic→West Ocean is
reactive dual-exact **2,618f** ×2 from the natural Main Shaft pin, settled at
`0x93FE` `(2008,139) p2` gs=8 dt=0. It follows `gravity_path_v2`
choreography: one Power Bomb at the right entry, then live Engage choices
for Kihunter bodies/wings and Atomics. Both runs observed
`enemies_killed=8` and zero door-counted enemies remaining before the left
gray door opened. Report: `scratch/attic_to_west_ocean_dual.json`; pin out:
`scratch/post_attic_to_west_ocean.state`.
West Ocean→Pancakes restores the human anchor's missing entry momentum with
8f `B+LEFT`, then follows the Gravity-v2 maze body. It is dual-exact
**1,476f** ×2 at `0x9461` `(39,139) p9` gs=8 dt=0. Report:
`scratch/west_ocean_to_pancakes_dual.json`; pin out:
`scratch/post_west_ocean_to_pancakes.state`.
Phase dumps are named scratch pins.

**Pin in:** `scratch/post_west_ocean_to_pancakes.state` (`0x9461`
`(39,139) p9` gs=8 dt=0)
**Goal:** Homing Geemer `0x968F` gs=8 dt=0.
**Living checkbox:** `pancakes_to_homing_geemer` (**RED settle**).
Natural Attic entry remains `scratch/post_ws_main_to_attic.state`.

### Already green (do not re-prove)

| Layer | Dual | Leave |
|-------|-----:|-------|
| Power-on Phantoon | **195,336f** ×2 | `0xCC6F` (1240,139) p10 gs=8 |
| Hop 1 Basement → Main | **1,579f** ×2 | `0xCAF6` (1173,1979) p1 gs=8 |
| Hop 2 grate_seat (usable) | **118f** ×2 | `0xCAF6` (1217,1867) p9 gs=8 |
| Hop 2 natural west_super | **580f** ×2 | `0xCAF6` (1094,1700) p48 gs=8 |
| Hop 2 mid_climb | **1,035f** ×2 | `0xCAF6` (1101,651) p9 gs=8 |
| Hop 2 attic_seat | **2,044f** ×2 | `0xCAF6` (1115,97) p9 gs=8 |
| Hop 2 Main Shaft → Attic | **2,745f** ×2 | `0xCA52` (1133,203) p1 gs=8 |
| Hop 3 Attic → West Ocean | **2,618f** ×2 | `0x93FE` (2008,139) p2 gs=8; 8 kills, 0 required left |
| Hop 4 West Ocean → Pancakes | **1,476f** ×2 | `0x9461` (39,139) p9 gs=8 dt=0 |

Observable `(1189, 1883) p2` and take04 `~(1195, 1883)` are not that pin.

### Hop 2 seams

Controller: `routes/kpdr/wrecked_ship/` geometry + overlay + play. Unpowered
`ws_main.py` is a different hop. Phase ladder:
[`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md). Beads: `rr-1xc2.8`.

| File | Role |
|------|------|
| `leave_specs.py` | usable outgoing pin (`WS_MAIN_GRATE_SEAT`) |
| `ws_main_geometry.py` | observable land band (`GRATE_LAND_*`) + region |
| `ws_main_departure.py` | take02 LEFT+A vs take04 walk-right (data) |
| `ws_main_actions.py` | one action per region |
| `ws_main_shaft.py` | overlay loop |
| `ws_main_ice.py` | ice overlay |
| `ws_main_climb.py` | play |

| Phase | From | Held exit | Status |
|-------|------|-----------|--------|
| 1 pit_shot | pin (1173,1979) | 3-shot, still Main, not Basement | PARTIAL — Wave+Spazer **opens the grate** |
| 2 grate_seat | pin | usable take02 fire slope | **118f** ×2 (1217,1867) p9 |
| 3 west_super | usable fire slope | y~1675 in shaft, not 0xCDA8 | **580f** ×2 natural entry, `(1094,1700) p48` |
| 4 mid_climb | 1675 | planted `(1101, 651) p9` | **1035f** ×2 natural west-super. 1019, 827, and 651 are slope-runs (B+RIGHT to 1243/907, B+LEFT to 1061/763, B+RIGHT to 1243/587) |
| 5 attic_seat | 651 | `(1115,97) p9` stand | **2,044f** ×2 natural mid-climb entry |
| 6 attic_door | door | Attic `0xCA52` gs=8 | **296f** ×2 phase pin; full hop **2,745f** ×2 |

All Main Shaft phases are green. 1675 takeoff is
`TakeoffWindow((1054, 1074), "RIGHT")` gun-jump RIGHT+A (tape; no air-B).
1543 takeoff is far-right `(1248, 1260) LEFT` (takes 02–05), not guessed
`(1120, 1180)`. 1019 takeoff is right wall `(1240, 1246) LEFT`. 827
takeoff is left wall `(1058, 1064) RIGHT`. 651 takeoff is `(1228, 1234)
LEFT` after 5f UP+X / 6f UP opens three near-Samus 572 blocks
`(1224,552) (1208,520) (1208,488)`; spin-jump from the plant leftover
`(1061, 752) p77` is the 827 wall. `$1C87` PLM coords divide the
byte offset by two. Live wall pixels are those triples, not tape 904/1112. Phase dumps are not hop GREEN. Do not `--source`
`post_ws_main_grate_land.state` (or the 1189 hash) as if it were
grate_seat. Do not boot `post_ws_main_shaft_1543.state`, leftover
`(1099, 1095) p38`, leftover `(1045, 1066) p48`, leftover `(1187, 817)
p48`, leftover `(1061, 752) p77`, leftover `(1240, 587) p38`, leftover
`(1224, 587) p38`, leftover `(1234, 587) p3`, leftover `(1082, 523) p12`,
leftover `(1061, 358) p82`, or `post_ws_main_shaft_1083.state` as
if they were a planted attic_seat pin. The 1083 dump is p138 wall-contact,
not hop GREEN.

### Departure windows (rr-1xc2.8.2 locked)

Living policy is take02 slope LEFT+A. Window: `SLOPE_LEFT_A` x
(1227, 1231) × y (1852, 1856). Data:
`routes/kpdr/wrecked_ship/ws_main_departure.py`. Not `climb_action`.

| Take | Policy | Lip fire | Grounded LEFT+A | Peak y |
|------|--------|----------|-----------------|--------|
| 02 | **slope_left_a (living)** | (1223,1860) p3 UP+X, 0xD080 f306 | (1231,1852) p3, then LEFT+A | 1763 |
| 03 | slope_left_a (agrees) | (1227,1856) p3 UP+X, f139 | (1227,1856) p1, A then LEFT+A | 1763 |
| 04 | walk_right_alcove | **(1195,1883) p3 UP+X then RIGHT**, f215 | (1242,1851) p1 save ledge | 1795 |
| 05 | walk_right_alcove | (1243,1851) p6 X+R, f296 | (1243,1851) p2 | 1795 |

take02 after fire walks RIGHT 8px on the slope to 1231 (still inside
`WS_MAIN_GRATE_SEAT`). x≈1221–1227 during that LEFT+A is airborne
y~1800, not the takeoff. take05 (1221,1862) p4 UP+X is a later fire
(f539), not the first 0xD080. Take04/05 alcove x≥1242 is outside the
glance spec. No take fires LEFT+X from the hatch-lip pocket ~(1177, 1883).
Take02 two-hop: short A from ~(1166,1979) that **fails**, land, walk LEFT
to 1156, committed A, RIGHT+A at y~1920, land (1208,1875) p9, walk to
(1223,1860). Human never hops LEFT off 1166 toward the stairs.

### Next — one seam

Settle `play_pancakes_to_homing_geemer` from the natural
`post_west_ocean_to_pancakes.state` pin. The current s23 body already crosses
the right door; the live miss is only destination settle (`gs=11`, `dt=1`),
not Pancakes geometry. Keep all earlier rooms closed.

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  snes/super_metroid/scripts/probe/kpdr.py pure pancakes-to-homing-geemer \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_west_ocean_to_pancakes.state \
  --expect-room 0x9461 --no-red-diag
```

### Non-claims

- Did not STATUS-promote Gravity
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not treat scratch `--to gravity` as living tip / Survival / Finish
- Did not wire hops 1–2 onto `--to phantoon`
- Did not treat a phase dump as hop GREEN
- Did not treat leftover `(1099, 1095) p38` or min_y 1082 as mid_climb green
- Did not treat leftover `(1045, 1066) p48` or min_y 1066 as mid_climb green
- Did not treat leftover `(1187, 817) p48` or min_y 792 as mid_climb green
- Did not treat leftover `(1159, 958) p47` as mid_climb green
- Did not treat `post_ws_main_shaft_1083.state` as hop GREEN or mid_climb green
- Did not treat leftover `(1154, 1561) p76` as mid_climb green
- Did not treat leftover `(1061, 752) p77` or min_y 628 as attic_seat green
- Did not treat leftover `(1234, 587) p3` or min_y 572 as attic_seat green
- Did not treat leftover `(1082, 523) p12` or min_y 499 as attic_seat green
- Did not treat leftover `(1061, 358) p82` or min_y 356 as attic_seat green
- Did not treat land_523 `(1216, 518)` as hop GREEN
- Did not treat a 443 overshoot as a planted 443 seat
- Did not treat leftover `(1117, 640) p47` overlapping Atomic as mid_climb green
- Did not treat pocket `(1177, 1883)` or land `(1189, 1883)` as fire-slope green
- Did not treat take04 alcove as the living handoff
- Did not treat the RED Pancakes→Homing Geemer settle as composed Gravity evidence
- Did not power-on / Phantoon-leave `--to gravity`
