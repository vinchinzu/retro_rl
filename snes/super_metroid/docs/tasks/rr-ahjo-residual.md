## Residual — rr-ahjo WS Entrance → Main Shaft

### Intent
Replace `play_ws_entrance_to_main` 240f RIGHT+B scaffold with a dual-green
walk `0xCA08` → `0xCAF6`. Do **not** STATUS-promote. Default CLI stays `ice`.
Do **not** append this hop to `POST_ICE_SPINE` / `WS_ONLY_HOPS` / `--to ws`.

### One change
This hop: unpowered 4-screen hallway walk/run right, beam the blue door,
ordinary Main Shaft settle. Default CLI tip stays `ice`. Do **not**
STATUS-promote.

### Source state
`scratch/post_ws_poweron.state` (WS Entrance `(57,139)` p1 gs=8).
Leave pin: `scratch/post_ws_entrance_to_main.state`.

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/ws_entrance.py pure --dual
# → GREEN dual=True 0xCAF6 (1063,907) p9 gs=8 dt=0 frames=403 ×2 exact
#    items 0x3105 beams 0x1007 selected=0 max PB 5
#    leave pin scratch/post_ws_entrance_to_main.state
```

### Acceptance
- [x] `play_ws_entrance_to_main` RAM-driven (select beam, `hold_until` x≥900
      or `0xCAF6`, then `play_run_shoot_exit` blue door + `wait_ordinary_room`)
- [x] Public policy in docstring (unpowered 4-screen, Covern only, blue door)
- [x] Basement / Phantoon still `_scaffold_exit` (240f)
- [x] BEFORE scaffold timeout **240f** still `0xCA08` `(987,139)` p137
- [x] AFTER dual GREEN **403f** ×2 `0xCAF6` `(1063,907)` p9 gs=8 dt=0
- [x] Unit tests without emulator (`test_wrecked_ship_scaffold.py`)
- [x] `WS_ONLY_HOPS` still `["west_ocean_to_ws"]`; `--to ws` still ends `0xCA08`
- [x] `DEFAULT_CONTINUOUS_TIP` stays `ice`

| | frames | seconds | clock |
|---|---:|---:|---|
| BEFORE scaffold (timeout, still `0xCA08`) | 240 | 3.993 | 00:04.00 |
| AFTER product dual GREEN | 403 | 6.706 | 00:06.72 |
| Δ | +163 | +2.712 | +00:02.72 |

Success vs the timeout scaffold; negative Δ is faster — this hop is slower
because it actually opens the door and settles.

### Next action (required)
- **One change:** `rr-4btp` — replace `play_ws_main_to_basement` scaffold
  with a dual-green walk `0xCAF6` `(1063,907)` p9 → `0xCC6F` Basement
  ordinary gs=8 from `scratch/post_ws_entrance_to_main.state`. Do not
  rewrite attic / Phantoon. Do not append to `--to ws`. Planner STATUS
  for `moat` is `rr-g3nj`. Do not STATUS-promote.
- Mid→thin is still the 2974f period WJ.
- Default CLI stays `ice`.

### Non-claims
- Did not STATUS-promote past Ice
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not write `recordings/ws.json` / `recordings/moat.json` (scratch only)
- Did not append Entrance→Main to `POST_ICE_SPINE` / `WS_ONLY_HOPS` / `--to ws`
- Did not close `rr-g3nj`
- Did not replace mid→thin period WJ
- Did not rewrite `play_ws_main_to_basement` / `play_ws_basement_to_phantoon`
