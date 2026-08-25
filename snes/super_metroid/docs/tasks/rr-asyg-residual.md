## Residual — rr-asyg doppler Phantoon + loot/exit

Wire the rr-7lc5 **wiki missile doppler** as the `--to phantoon` fight body
and extend the tip through flame-drop loot + left-door exit into WS
Basement. Charge-only / charge+missiles / Ice-on X-Factor stay research.
Do **not** STATUS-promote. Default CLI stays `ice`. `--to ws` still ends
`0xCA08`.

**Pin in:** `scratch/post_ws_basement_to_phantoon.state` (`0xCD13` ~(39,124)
p81 gs=8 HP 2500)
**Pin out:** `scratch/post_phantoon_leave.state` (`0xCC6F` ~(1240,139) p10
gs=8). Did **not** clobber `post_phantoon_poweron.state` (in-room charge
kill) or `post_phantoon_defeated.state`.

### Why doppler

Same natural pin, assist ON (rr-7lc5):

| recipe | frames | vs 20537f | note |
|---|---:|---:|---|
| charge-only | 20537 ×2 | 0 | research |
| charge+missiles | 27645 ×2 | +7108 | research |
| **doppler 2-2-N** | **12118** ×2 | **−8419** | product |
| Ice-on X-Factor | window miss | n/a | research |

Doppler extras during the ~10f close still count **0** (recipe landed
2-2-1). Super finisher is gated on HP ≤ 600. Faster pin kill + next hop
(loot/exit) dual-green from that leave.

### Fight + leave (product)

| layer | frames | clock | leave |
|---|---:|---|---|
| doppler fight | **12118** ×2 | 03:21.97 | `0xCD13` (37,187) p1 HP 0 + `$D82B` bit 0 |
| loot + left-door (from charge pin) | **335** ×2 | 00:05.58 | `0xCC6F` (1240,139) p10 |
| compose fight+leave | **12455** ×2 | 03:27.58 | same basement; 12118+337 |

Loot on the charge/doppler leave pin: no remaining drops (sweep 37f then
jump LEFT+A). Floor-hug LEFT at x≤40 is wall knockback p138 — the door
slot is the enter height.

Reports: `scratch/phantoon_loot_exit_dual.json`,
`scratch/phantoon_doppler_leave_dual.json`.

### Spine

`PHANTOON_ONLY_HOPS` += `phantoon_loot_exit` (`0xCD13` → `0xCC6F`, left /
right). Fight hop still in-room. Tip `phantoon` `final_room` **0xCC6F**.

### Verify paste

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  snes/super_metroid/tests/test_phantoon_leave.py \
  snes/super_metroid/tests/test_phantoon_combat.py \
  snes/super_metroid/tests/test_phantoon_doppler.py \
  snes/super_metroid/tests/test_post_ice_spine.py \
  snes/super_metroid/tests/test_continuous_tips.py \
  snes/super_metroid/tests/test_k4_speed_branches.py \
  snes/super_metroid/tests/test_progression.py -q
# → unit GREEN; DEFAULT_CONTINUOUS_TIP still ice; --to phantoon hops
#    … fight, phantoon_loot_exit; final_room 0xCC6F
```

### Acceptance

- [x] Spine fight is `play_phantoon_doppler_fight`
- [x] Charge / charge+missiles / xfactor modules remain, unused by spine
- [x] `phantoon_loot_exit` dual 335f ×2 from in-room kill pin
- [x] Compose doppler+leave dual **12455f** ×2 at `0xCC6F` (1240,139) p10
- [x] `--to ws` / `WS_ONLY_HOPS` / `DEFAULT_CONTINUOUS_TIP` unchanged
- [ ] Power-on `--to phantoon` dual (rr-8g2u)

### Next action (required)

- **One change:** `rr-8g2u` — pin compose `ws-to-phantoon` from
  `scratch/post_ws_poweron.state` with doppler + loot-exit (expect
  ~14784f pin / ~190925f power-on). Scratch only. Do not STATUS-promote.
  Do not write `recordings/phantoon.json` on a red run.

### Non-claims

- Did not STATUS-promote past Ice
- Did not change `DEFAULT_CONTINUOUS_TIP` or `WS_ONLY_HOPS`
- Did not write `recordings/phantoon.json`
- Did not run power-on `--to phantoon`
- Did not clobber `post_phantoon_poweron.state` / `post_phantoon_defeated.state`
- Did not claim true wiki doppler extras or a 2-round X-Factor
- Did not start Gravity / Leave
