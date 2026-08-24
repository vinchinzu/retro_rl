# rr-tlaq Phantoon fight — status

**Bead:** rr-tlaq CLOSED. Assist dual-green kill **20537f** ×2. Do not
STATUS-promote. Do not append to `--to ws`.

## Assist kill

| run | frames | windows | HP | `$D82B` | gs | health |
|-----|-------:|--------:|---:|--------:|---:|-------:|
| 1 | 20537 | 9×300 | 0 | bit 0 | 8 | 299 |
| 2 | 20537 | 9×300 | 0 | bit 0 | 8 | 299 |

`strategy --assist --weapon beam` from `scratch/post_ws_basement_to_phantoon.state`.
Same skip/jump policy. Super unused. Pin `scratch/post_phantoon_poweron.state`
(did not clobber `post_phantoon_defeated.state`). Reports
`scratch/phantoon_assist_kill.json` + `_dual.json`.

## No-assist ceiling (do not re-prove)

W1 fig-8 (104, 149) vs (120, 108); rain (48, 96) (37, 148) p21. Skip x=219,
(128, 96), (88, 64), (53, 82), (56, 113), (83, 64). Jump only legal (48, 96).
Best W1–W6 2500→700 then `$D82A` halt. 54–59 HP cannot tank `$D82A`.

## Next

Leave Phantoon's Room / WS power-on from the new pin. Residual
`docs/tasks/rr-tlaq-residual.md`. Default CLI stays `ice`.
