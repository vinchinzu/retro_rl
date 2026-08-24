## Residual — rr-gyla `--to phantoon` wire (catalog + fight wrapper)

### Intent
Wire unpowered WS interior hops + the existing assist fight as continuous tip
`--to phantoon` (parent `ws`). Do **not** STATUS-promote. Default CLI stays
`ice`. Do **not** append these hops to `--to ws` / `WS_ONLY_HOPS`. Tip `ws`
still ends at `0xCA08`.

### One change
This card: catalog + SpineHops + thin `play_phantoon_room_fight` wrapper.
Did **not** rewrite Entrance / Main / Basement / fight bodies. Did **not**
run power-on dual (rr-8g2u).

### Source state
`scratch/post_ws_poweron.state` (WS Entrance `(57,139)` p1 gs=8) → interior
hops already dual-green (rr-ahjo / rr-4btp / rr-cjpp). Fight pin
`scratch/post_ws_basement_to_phantoon.state` (rr-tlaq assist kill **20537f** ×2).

### Verify paste
```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  snes/super_metroid/tests/test_continuous_tips.py \
  snes/super_metroid/tests/test_post_ice_spine.py \
  snes/super_metroid/tests/test_phantoon_combat.py \
  snes/super_metroid/tests/test_k4_speed_branches.py \
  snes/super_metroid/tests/test_progression.py -q
# →  unit GREEN; DEFAULT_CONTINUOUS_TIP still ice; WS_ONLY_HOPS still
#    ["west_ocean_to_ws"]; --to phantoon hops
#    entrance→main→basement→room→fight; fight emits no DoorEdge
```

### Acceptance
- [x] `--to phantoon` registered (parent `ws`; aliases `phan` / `k6_phantoon`)
- [x] `PHANTOON_ONLY_HOPS` = entrance / main / basement / fight
- [x] `WS_ONLY_HOPS` still `["west_ocean_to_ws"]`; `--to ws` still ends `0xCA08`
- [x] `play_ws_basement_to_phantoon` still does not fight
- [x] Wrapper uses `PhantoonStrategy(weapon=beam, shots_per_window=3)`; never Super
- [x] `after=require_phantoon_defeated` peeks `$7E:D82B` bit 0
- [x] `DEFAULT_CONTINUOUS_TIP` stays `ice`
- [ ] Power-on / pin compose dual (rr-8g2u)

### Next action (required)
- **One change:** `rr-8g2u` — pin compose `ws-to-phantoon` from
  `scratch/post_ws_poweron.state` (~22866f) with the existing charge fight
  (wiki doppler 12118f is pin-only scratch; do not swap). If GREEN, power-on
  `continuous.py --to phantoon --no-video` dual (scratch only, ~199k).
  Do not STATUS-promote. Do not write `recordings/phantoon.json` on a red run.
  Do not leave the room / Gravity. Wiki benches: `docs/tasks/rr-7lc5-residual.md`.

### Non-claims
- Did not STATUS-promote past Ice
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not write `recordings/ws.json` / `recordings/phantoon.json`
- Did not append interior/fight hops to `--to ws` / `WS_ONLY_HOPS`
- Did not close `rr-g3nj`
- Did not rewrite hop bodies or `combat/phantoon.py`
- Did not clobber `post_phantoon_defeated.state`
- Did not Super-spray
- Did not run the emulator compose
