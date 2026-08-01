# SM-KIHUNTER-RECON Report

## Result

Diagnostic recon only. This report does not claim pure green, continuous
evidence, or a route/status promotion.

## Method

- Source: `custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state`
- Fresh emulator session per x target.
- Repeated the existing read-only shot-block climb from `kraid_return.py`.
- Moved toward each target x with ordinary `RIGHT`/`LEFT` + spin, then held
  `DOWN`, periodically firing beam shots.
- Recorded room, pose, x, y, and `door_transition` every frame.
- Applied only `UnlimitedResourcesAssist`; no progression, capacity, room,
  door, event, or boss RAM writes.

## Measurements

The table below is populated from the JSON emitted by
`debug/kihunter_zeela_recon.json` after the required probe command. The
`upper_warp` rows are development-only position probes because the natural
climb reaches the known wrong-door transition before the upper y band.

| Target x | Entry | Upper pin | DOWN x band | Transition room | Final room |
|---:|---|---|---:|---|---|
| 64 | position warp | `(64, 240)` | 64 | none | `0xA4DA` |
| 96 | position warp | `(96, 240)` | 96 | `0xA471` Zeela | `0xA471` |
| 128 | position warp | `(128, 238)` | 128 | `0xA471` Zeela | `0xA471` |
| 160 | position warp | `(160, 240)` | 160 | `0xA471` Zeela | `0xA471` |
| 192–480 | position warp | `(x, 240)` | target x | none | `0xA4DA` |

Natural climb diagnostic: the source-room upper-edge transition starts at
`x=492`, remains `0xA4DA` during loading, and completes in Baby Kraid
`0xA521` at approximately `x=39`, `y=116` in the prior R-02B pin. No natural
climb trial reached a stable upper-band position in this recon.

Interpretation bands:

- **Zeela candidate band:** `x=96..160` in the development upper-position
  sweep; all three sampled points selected `0xA471`.
- **Baby candidate band:** natural climb trigger at source-room `x=492`,
  completing in `0xA521`; the warped sweep did not reproduce Baby at its
  tested x values.
- **Unresolved band:** rows that remain in `0xA4DA` or do not produce a
  transition within the frame budget.

## Recommendation

Recommended numeric window for R-02C review: **`x=96..160`** in the
source-room upper band. This is the narrowest contiguous interval supported by
the sampled targets that selected `0xA471`; it must be validated with a denser
natural-input sweep before being applied to route geometry. Do not use the
natural `x=492` edge, which is the observed Baby-side failure trigger.

## Non-claims

- Did not claim pure green.
- Did not claim continuous evidence.
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- This probe cannot identify live PLM, door BTS, or internal door-state fields.
- The recommended interval comes from a development position warp, not a
  natural climb, and therefore is a diagnostic candidate only.
