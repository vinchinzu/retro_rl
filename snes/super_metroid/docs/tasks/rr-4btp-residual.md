## Residual — rr-4btp WS Main Shaft → basement

### Intent
Replace `play_ws_main_to_basement` 240f RIGHT+B scaffold with a dual-green
walk `0xCAF6` → `0xCC6F`. Do **not** STATUS-promote. Default CLI stays `ice`.
Do **not** append this hop to `POST_ICE_SPINE` / `WS_ONLY_HOPS` / `--to ws`.

### One change
This hop: unpowered first-visit descent, morph-stair / dash switchbacks,
PB the floor pipes, jump-down Super the green hatch, ordinary Basement
settle. Default CLI tip stays `ice`. Do **not** STATUS-promote.

### Source state
`scratch/post_ws_entrance_to_main.state` (WS Main Shaft `(1063,907)` p9 gs=8).
Leave pin: `scratch/post_ws_main_to_basement.state`.

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/ws_main.py pure --dual
# → GREEN dual=True 0xCC6F (657,92) p24 gs=8 dt=0 frames=1208 ×2 exact
#    items 0x3105 beams 0x1007 selected=2 max PB 5
#    leave pin scratch/post_ws_main_to_basement.state
```

### Acceptance
- [x] `play_ws_main_to_basement` not `_scaffold_exit` (s21 human RLE 1091f
      + `wait_ordinary_room` `0xCC6F` gs=8)
- [x] Public policy in docstring (unpowered first visit, not post-Phantoon
      climb; skip attic / save / left missile; tank Coverns)
- [x] Phantoon still `_scaffold_exit` (240f)
- [x] BEFORE scaffold timeout **240f** still `0xCAF6` `(1243,907)` p137
- [x] AFTER dual GREEN **1208f** ×2 `0xCC6F` `(657,92)` p24 gs=8 dt=0
- [x] Unit tests without emulator (`test_wrecked_ship_scaffold.py`)
- [x] `WS_ONLY_HOPS` still `["west_ocean_to_ws"]`; `--to ws` still ends `0xCA08`
- [x] `DEFAULT_CONTINUOUS_TIP` stays `ice`

| | frames | seconds | clock |
|---|---:|---:|---|
| BEFORE scaffold (timeout, still `0xCAF6`) | 240 | 3.993 | 00:04.00 |
| AFTER product dual GREEN | 1208 | 20.100 | 00:20.13 |
| Δ | +968 | +16.107 | +00:16.13 |

Success vs the timeout scaffold; negative Δ is faster — this hop is slower
because it actually opens the floor hatch and settles.

### Next action (required)
- **One change:** `rr-cjpp` — replace `play_ws_basement_to_phantoon` scaffold
  with a dual-green walk `0xCC6F` `(657,92)` p24 → `0xCD13` Phantoon
  ordinary gs=8 from `scratch/post_ws_main_to_basement.state`. Do not
  rewrite attic / Phantoon fight. Do not append to `--to ws`. Planner STATUS
  for `moat` is `rr-g3nj`. Do not STATUS-promote.
- Mid→thin is still the 2974f period WJ.
- Default CLI stays `ice`.

### Non-claims
- Did not STATUS-promote past Ice
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not write `recordings/ws.json` / `recordings/moat.json` (scratch only)
- Did not append Main→basement to `POST_ICE_SPINE` / `WS_ONLY_HOPS` / `--to ws`
- Did not close `rr-g3nj`
- Did not replace mid→thin period WJ
- Did not rewrite `play_ws_entrance_to_main` / `play_ws_basement_to_phantoon`
