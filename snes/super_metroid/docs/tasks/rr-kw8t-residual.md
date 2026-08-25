## Residual — rr-kw8t Gravity on the Phantoon tip

**Status:** BLOCKED on hop 1 phase B (Ice kill + east takeoff). Living
tip is `--to phantoon` **195,336f** ×2 (STATUS). Do not STATUS from a pin.

**Pin in:** `scratch/post_phantoon_leave.state` (`0xCC6F` ~(1240,139) p10
gs=8, `$D82B` bit 0)
**Goal:** Gravity Suit PLM (`0xCE40` / items bit `0x0020`), then power-on
compose onto the one tip. Shape: `tasks/gravity_path_human` /
`scratch/post_gravity_caterpillar.state` — guideline only.

### Already green (do not re-prove)

| Layer | Dual | Leave |
|-------|-----:|-------|
| Power-on Phantoon | **195,336f** ×2 | `0xCC6F` (1240,139) p10 gs=8 |
| Fight+leave pin compose | **12,455f** ×2 | same basement |
| Hop 1 phase A | tunnel LEFT | floor under hatch ~(630–690, 185) |

### Hop 1 — powered Basement → Main Shaft

Controller: `routes/kpdr/k6/ws_basement_return.py`
`play_ws_basement_to_main`. **Keep it.** Do not revert Ice keepaway or
`hatch_mount_action`. Dual:
`scripts/probe/ws_basement_return.py --dual --no-video`.
Scratch: `ws_basement_to_main_dual.json` (RED, overwrite next).

Watch (repo-wide `--headed`, not a custom pygame loop):

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure ws-basement-to-main \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_phantoon_leave.state \
  --headed
./snes/super_metroid/play post_phantoon_leave --headed --assist-full
```

`--headed` is `retro_harness.headed` (`add_headed_flag`, `attach_headed`).
Probe calls `UnlimitedResourcesAssist.attach_env` **before** the window so
the HUD reads `hp=n/max` after refill. `./play` default is ON@0.

### Phase A (reached, freeze)

Morph-roll LEFT from the Phantoon door clears the tunnel.

### Phase B (BLOCKED — halt duals)

East takeoff + Ice-until-dead are in the hop. They still do not stand on
~(657,91). Did **not** shoot the blue door.

Earlier in-band floor jumps (do not restore): (651,188) p81 `UP+A`;
(641,187) p9 into the Workrobot; (657,187) p2 standing on the robot.

This session (keep these adds):

| # | Final | Miss |
|--:|-------|------|
| 1 | (974,187) p37 mov=14 2222f | Charge held, no Ice shot; pose-37 stall |
| 2 | **(879,187) p2** **8108f** | tap-release Ice. `enemies_killed=1`. Health **299/299** |

Timeout dump (do not re-dump unless the pin moved):

| slot | id | xy | hp | freeze | what |
|-----:|----|-----|---:|-------:|------|
| 0 | `0xE8FF` | **(624,176)** | 800 | 0 | Workrobot — now on the hatch seat |
| 1 | `0xE8FF` | (384,176) | 800 | 0 | Workrobot |
| 2 | `0xE9FF` | (152,77) | 250 | 0 | Atomic, map-side — skip |
| 3 | `0xE9FF` | **(638,168)** | **250** | 0 | hatch-floor Atomic. Unfrozen. Shots from 879 miss |
| 5 | `0xEA3F` | (1145,106) | 80 | 0 | Covern — tank |

The 852 high Atomic is gone (the kill). 638 is still 250 after ~8k frames of
2f-tap / 6f-release Ice from x=879. Horizontal taps do not connect;
`ice_keepaway_action` stays on that blob and never reaches the 720 takeoff.
Workrobot 624 occupies under-hatch. Beams Ice `0x1007`.

### Next session (add, do not revert)

Keep `hatch_mount_action` (x≳720 `spin_jump("LEFT")`). Keep Ice-until-dead
around `_run_to_hatch`. Keep `attach_env` before `--headed`.

One add: **hit the 638 Atomic.** Tap Ice from 879 is not a hit. Need a
shared shoot primitive (charge-release, aim angle / jump-shot, position
under the blob, then fire) so 638 actually dies, then takeoff, then
tap-shot the blue ceiling door into ordinary `0xCAF6` gs=8.

Do not 4th-dual this miss. Overwrite `scratch/ws_basement_to_main_dual.json`
only. Glance the leave. No mid pin. No STATUS. Tip stays `phantoon`.

### Non-claims

- Did not STATUS-promote Gravity
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not treat `post_gravity_caterpillar` as power-on
- Did not concat `gravity_path_human` as the tip
- Did not wire `ws_basement_to_main` onto `--to phantoon`
- Did not treat low-WRAM `boss_bits[3]` as Phantoon (open-bus); bank 7E
  still has bit 0
- Did not revert Ice keepaway or the east takeoff
