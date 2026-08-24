## Residual — rr-p2bw `--to ws` (over-ocean spark)

### Intent
Wire West Ocean → Wrecked Ship Entrance as continuous tip `--to ws`
(`play_west_ocean_over_ocean_spark`, parent `moat`). Do **not**
STATUS-promote. Default CLI stays `ice`.

### One change
This hop: power-on `--to ws` dual continuous (scratch). Default CLI
tip stays `ice`. Do **not** STATUS-promote.

### Source state
`scratch/post_moat_poweron.state` (West Ocean ~(49,1163) p1).
Leave pin: `scratch/post_ws_poweron.state`.

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py compose moat-to-ws \
  --source snes/super_metroid/scratch/post_moat_poweron.state \
  --output snes/super_metroid/scratch/post_ws_from_moat_poweron.state \
  --no-red-diag
# → GREEN 0xCA08 (57,139) p1 frames=615 ×2 exact dual

uv run python snes/super_metroid/scripts/record/continuous.py --to ws --no-video \
  --report snes/super_metroid/scratch/ws_poweron.json \
  --state-output snes/super_metroid/scratch/post_ws_poweron.state
# → GREEN 0xCA08 (57,139) p1 frames=176141 ×2 exact dual; max PB 5
```

### Acceptance
- [x] `--to ws` registered (parent moat → ws; aliases wrecked_ship / ws_entrance / k6_ws; `west_ocean` stays on moat)
- [x] Spine hop `west_ocean_to_ws` = `play_west_ocean_over_ocean_spark`
- [x] Compose `moat-to-ws` dual GREEN **615f** ×2 `0xCA08` `(57,139)` p1
- [x] Ice-pin compose `ice-to-ws` GREEN **29212f** `0xCA08` `(57,139)` p1 max PB 5
- [x] Power-on `--to ws` dual GREEN **176141f** ×2 `0xCA08` `(57,139)` p1 gs=8 items `0x3105` beams `0x1007` max PB 5; integrity green (loads/prog/deaths 0)

| | frames | seconds | clock |
|---|---:|---:|---|
| Spine hop West Ocean → WS | 615 | 10.233 | 00:10.25 |
| Ice-pin compose Ice → WS | 29212 | 486.066 | 08:06.87 |
| Power-on `--to ws` (scratch dual) | 176141 | 2930.857 | 48:55.68 |
| Power-on `--to moat` prefix | 175526 | 2920.624 | 48:45.43 |

### Next action (required)
- **One change:** `rr-ahjo` — replace `play_ws_entrance_to_main` scaffold
  with a dual-green walk `0xCA08` `(57,139)` p1 → `0xCAF6` Main Shaft
  ordinary gs=8 from `scratch/post_ws_poweron.state` (or
  `post_moat_poweron_wo_to_ws.state`). Do not rewrite attic / basement /
  Phantoon. Planner STATUS for `moat` is `rr-g3nj`. Do not STATUS-promote.
- Mid→thin is still the 2974f period WJ.
- Default CLI stays `ice`.

### Non-claims
- Did not STATUS-promote past Ice
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not write `recordings/ws.json` / `recordings/moat.json` (scratch only)
- Did not close `rr-g3nj`
- Did not replace mid→thin period WJ
- Probe spark 627f vs spine hop 615f is session accounting, not a mismatch
