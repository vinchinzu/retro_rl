## Residual — rr-cjpp WS Basement → Phantoon room

### Intent
Replace `play_ws_basement_to_phantoon` 240f RIGHT+B scaffold with a dual-green
walk `0xCC6F` → `0xCD13`. Do **not** STATUS-promote. Default CLI stays `ice`.
Do **not** append this hop to `POST_ICE_SPINE` / `WS_ONLY_HOPS` / `--to ws`.
Do **not** start the Phantoon fight.

### One change
This hop: unpowered basement hallway, land from p24 fall, skip left map,
bomb morph-tunnel obstruction, Super Gadora then blue shell, ordinary
Phantoon-room settle. Default CLI tip stays `ice`. Do **not** STATUS-promote.

### Source state
`scratch/post_ws_main_to_basement.state` (WS Basement `(657,92)` p24 gs=8).
Leave pin: `scratch/post_ws_basement_to_phantoon.state`.

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/ws_basement.py pure --dual
# → GREEN dual=True 0xCD13 (39,124) p81 gs=8 dt=0 frames=718 ×2 exact
#    items 0x3105 beams 0x1007 selected=2 max PB 5
#    leave pin scratch/post_ws_basement_to_phantoon.state
```

### Acceptance
- [x] `play_ws_basement_to_phantoon` not `_scaffold_exit` (RAM-driven land +
      morph-tunnel bomb + Gadora Super + `wait_ordinary_room` `0xCD13` gs=8)
- [x] Public policy in docstring (unpowered basement, skip map LEFT, bomb X
      while morph, Super eye door, do not fight)
- [x] Phantoon *fight* not implemented (no HP loop / farm / kill)
- [x] Main / Entrance / attic not rewritten
- [x] BEFORE scaffold timeout **240f** still `0xCC6F` `(863,187)` p137
- [x] AFTER dual GREEN **718f** ×2 `0xCD13` `(39,124)` p81 gs=8 dt=0
- [x] Unit tests without emulator (`test_wrecked_ship_scaffold.py`)
- [x] `WS_ONLY_HOPS` still `["west_ocean_to_ws"]`; `--to ws` still ends `0xCA08`
- [x] `DEFAULT_CONTINUOUS_TIP` stays `ice`

| | frames | seconds | clock |
|---|---:|---:|---|
| BEFORE scaffold (timeout, still `0xCC6F`) | 240 | 3.993 | 00:04.00 |
| AFTER product dual GREEN | 718 | 11.947 | 00:11.97 |
| Δ | +478 | +7.954 | +00:07.97 |

Success vs the timeout scaffold; negative Δ is faster — this hop is slower
because it actually bombs the tunnel, Super the Gadora, and settles.

### Next action (required)
- **One change:** `rr-tlaq` — Phantoon *fight* from `scratch/post_ws_basement_to_phantoon.state`
  `0xCD13` `(39,124)` p81 gs=8. Do not rewrite attic / basement hop. Do not
  append to `--to ws`. Planner STATUS for `moat` is `rr-g3nj`. Do not
  STATUS-promote.
- Mid→thin is still the 2974f period WJ.
- Default CLI stays `ice`.

### Non-claims
- Did not STATUS-promote past Ice
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not write `recordings/ws.json` / `recordings/moat.json` (scratch only)
- Did not append Basement→Phantoon to `POST_ICE_SPINE` / `WS_ONLY_HOPS` / `--to ws`
- Did not close `rr-g3nj`
- Did not replace mid→thin period WJ
- Did not rewrite `play_ws_entrance_to_main` / `play_ws_main_to_basement`
- Did not start the Phantoon fight
