## Residual — rr-kw8t Gravity on the Phantoon tip

**Status:** BLOCKED on hop 1 phase B (latch the east takeoff, stand on the
mid platform, then door). Ice hit on 638 is in. Living tip is
`--to phantoon` **195,336f**
×2 (STATUS). Do not STATUS from a pin.

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
| Hatch-floor Atomic 638 | **dead** | `enemies_killed=2`; slot 3 gone |

### Hop 1 — powered Basement → Main Shaft

Controller: `routes/kpdr/k6/ws_basement_return.py`
`play_ws_basement_to_main`. Ice policy in `ws_basement_ice.py`. **Keep
both.** Do not revert Ice-until-dead or `hatch_mount_action` east
takeoff (x≳720 spin-LEFT). Dual:
`scripts/probe/ws_basement_return.py --dual --no-video`.
Scratch: `ws_basement_to_main_dual.json` (still the 8108f 879-tap miss —
do not overwrite until a run actually sits on ~(657,163) p1/2).

Watch (repo-wide `--headed`, not a custom pygame loop):

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure ws-basement-to-main \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_phantoon_leave.state \
  --headed
./snes/super_metroid/play post_phantoon_leave --headed --assist-full
```

### Phase A (reached, freeze)

Morph-roll LEFT from the Phantoon door clears the tunnel.

### Ice primitive (landed this session — do not re-prove the 879 miss)

Shared shoot: `routes/skills/charge_shot.py`
`position_then_charge_action`. Charge equipped: hold X, release fires
(`$0CD0 >= 60`). Diagonal aim is shoulder **R** (never UP+LEFT).
Jump-shot holds X through the first airborne frames, then releases.

`ice_keepaway_action` walks into a seat (east of the Workrobot clamp)
then charge-releases. Horizontal taps from x=879 do **not** connect:
the hatch pillar blocks LOS to 638 at (638,168).

Movement stalls (not stun):

| name | RAM | what |
|------|-----|------|
| turning | mov=14 / pose 37–38 | firing X during the turn is the pose-37 stall |
| knockback | pose 137/138 | live Atomic contact |
| frozen_atomic | freeze_timer>0, overlap | solid; walking into it zeros vx |
| workrobot | `0xE8FF` gap<48 | solid, no damage |

Single-probe (not dual) after the primitive: **killed=2**, hatch-floor
638 gone, map-side Atomic 152 skipped, health 299/299. First Ice-only
window sat **(717,163) p48** on the right lip of the mid platform.

### Phase B (BLOCKED — hatch seat)

East takeoff + Ice-until-dead are in the hop. They still do not **stand**
on ~(657,163) p1/2. Did **not** shoot the blue door. Did **not** dual.
The takeoff seat is now farther east, x=820–880, because x~720 reached the
platform's right wall at (728,175) before clearing its top. The action also
finishes the RIGHT→LEFT turn before pressing A; a jump pressed during mov=14
does not latch.

Latest single (RED, `ws_basement_to_main.json` only):

| run | frames | final | miss |
|-----|-------:|-------|------|
| Ice primitive first window | 1882 | **(717,163) p48** | on the lip, facing right; hatch_mount was still RIGHT+B |
| latest | **2842** | (728,175) p26 mov=3 | spin-LEFT at takeoff height; not seated under the door |

Three new same-pin single windows (no dual) kept killed=2 and hp 299/299:

| window | final | signal |
|--------|-------|--------|
| baseline | (728,175) p26 mov=3 | reproduced platform-right-wall miss |
| east x=820–880 | (822,187) p76 mov=2 | jump was asked on the turn frame and did not latch |
| turn-before-jump | **(818,187) p38 mov=14** | LEFT turn drifted below x=820; stateless band then commanded RIGHT |

The last state is an input-policy oscillation, not Atomic contact: `stall=turning`,
both hatch-side Atomics dead, Workrobots at x~260/~440, hp 299/299.

Workrobots after the Ice kill walk west (~260 and ~440). Under-hatch
floor is then empty — but a gun-jump from y=187 hits the platform
**underside**, not the ceiling door. Door is from the mid platform.

### Next session (add, do not revert)

Keep Ice-until-dead + charge-release seat. Keep the farther-east
`hatch_mount_action` seat and turn-before-jump guard. Keep on-platform walk to
x=657 once y≤175 and vy≈0.

One add: latch the takeoff phase across the small x drift: once the x=820–880
seat starts turning LEFT, do not resume RIGHT approach at x=818. Finish the
turn, spin-LEFT, then **stand on the platform under the door** ~(657,163)
pose 1/2. The 717,163 lip landing is the handoff — land, walk LEFT, then
`hatch_jump_action` tap-shot the blue ceiling door into ordinary `0xCAF6`
gs=8.

Do not dual the 879-tap miss. Overwrite `scratch/ws_basement_to_main_dual.json`
only after a seated platform standing. Glance the leave. No mid pin.
No STATUS. Tip stays `phantoon`.

### Non-claims

- Did not STATUS-promote Gravity
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not treat `post_gravity_caterpillar` as power-on
- Did not concat `gravity_path_human` as the tip
- Did not wire `ws_basement_to_main` onto `--to phantoon`
- Did not treat low-WRAM `boss_bits[3]` as Phantoon (open-bus); bank 7E
  still has bit 0
- Did not revert Ice keepaway or the east takeoff
- Did not dual after the 638 hit (hatch seat still red)
- Did not overwrite the existing 879-miss dual
