## Residual — rr-8g2u power-on `--to phantoon` dual

**Status:** OPEN. `--to phantoon` is wired (rr-gyla) with wiki doppler +
loot/exit (rr-asyg). Pin compose and power-on dual are this card.
**Pin in:** `scratch/post_ws_poweron.state` (`0xCA08` ~(57,139) p1 gs=8)
**Pin out:** `scratch/post_phantoon_leave.state` (`0xCC6F` ~(1240,139) p10
gs=8, `$D82B` bit 0). Do not clobber `post_phantoon_poweron.state` /
`post_phantoon_defeated.state`.

Do **not** STATUS-promote. Default CLI stays `ice`. `--to ws` still ends
`0xCA08`. Glance leave with `super_metroid.hop_glance` — no MP4.

### Already green (do not re-prove)

| Layer | Dual | Leave |
|-------|-----:|-------|
| Entrance → Main | **403f** ×2 | `0xCAF6` (1063,907) p9 gs=8 |
| Main → basement | **1208f** ×2 | `0xCC6F` (657,92) p24 gs=8 |
| Basement → room | **718f** ×2 | `0xCD13` (39,124) p81 gs=8 |
| Doppler fight | **12118f** ×2 | `0xCD13` (37,187) p1 HP 0 + bit 0 |
| Loot + left-door | **337f** ×2 | `0xCC6F` (1240,139) p10 gs=8 |
| Fight+leave compose | **12455f** ×2 | same basement |

Charge-only / charge+missiles / Ice-on X-Factor stay research.

### Next action

- **One change:** pin compose `ws-to-phantoon` from
  `scratch/post_ws_poweron.state` with doppler + loot-exit, then power-on
  `--to phantoon --no-video` dual. Scratch only.
- **Glance:** final room `0xCC6F`, gs=8, stand pose, x∈[1200,1280],
  y∈[120,160], boss bit 1, health ≥1.

### Non-claims

- Did not STATUS-promote past Ice
- Did not change `DEFAULT_CONTINUOUS_TIP` or `WS_ONLY_HOPS`
- Did not write `recordings/phantoon.json`
- Did not rewrite Entrance / Main / Basement / fight bodies
