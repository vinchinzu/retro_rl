# Status: TMNT IV: Turtles in Time

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M8 |
| Best verified result | Clean Hard credits from power-on, Leonardo |
| Last verification | 2026-11-14 |
| Runtime class | Bronze |
| Intervention class | Clean |

One continuous emulator session from power-on through Super Shredder,
Hard staff/cast credits, and the final Splinter scene. Completion requires
1,200 final-scene settle frames followed by a further 600-frame capture
hold. The published video is a normal input-tape replay with no policy,
lookahead, recovery, or state loads.

## Verified Clean capture

| Evidence | Result |
|----------|--------|
| Power-on to completed credits | **01:13:47.558** (266,091 frames) |
| Complete tape/video | 266,691 frames, including the capture hold |
| Damage / minimum gameplay HP | 74 / 60 |
| Life losses / lives | **0** / 2 → 7 |
| Health assists / iframe assists | **0 / 0**, both disabled |
| RAM writes / state loads | **0 / 0** |
| A-special / post-boot START | **0 / 0** |
| Replay CRC checks / desyncs | **445 / 0** |
| Lookahead restores in published replay | **0** |
| In-game pizza pickups | 4 |
| Video | 768×720, 60 fps, stereo game audio |

Artifacts (local, gitignored):

- [Video](../recordings/tmnt_iv_full_hard_clean.mp4)
- [Published replay report](../recordings/tmnt_iv_full_hard_clean.json)
- [Complete tape](../recordings/tmnt_iv_full_hard_clean.tape.npz)
- [Dry verification](../recordings/tmnt_iv_full_hard_clean_dry_run.json)
- [Machine-readable integrity checks](../recordings/tmnt_iv_full_hard_clean_verification.json)
- [Final Splinter scene](../recordings/clean_final_splinter.png)
- [Completion card](../recordings/clean_complete_card.png)

## Provenance and verifier repair

Leo J was killed before saving its final tape. The preserved Leo I tape
contained all gameplay and the start of the ending, but its old verifier
failed when the cast roll reset stage byte 10 to 0. Recovery replayed all
253,714 original inputs from power-on, verified all 423 source CRCs, and
added **12,977 neutral-input frames only during the ending**. The complete
tape then passed the independent normal replay and video capture above.
The tape and reports retain the source path and SHA-256.

Input generation used the disclosed ROM lookahead contract. The original
search reported 290,318 rollout restores; the published replay uses none.
No source gameplay inputs were edited, no stage pins were stitched, and
no health or protection writes were used.

The cast demonstrations reuse stages 0–9, player HP, and lives. The
verifier now ends gameplay damage/healing/life accounting at the ending
transition while preserving write, state-load, input, and CRC audits
through the full roll. Recovery cannot fill in missing gameplay. Replay
also rejects A-special and post-boot START before committing them.

Validation: **81 unit tests**, **3 ROM tests**, full power-on dry replay,
and the separate full video replay. The final Splinter scene and
completion card were visually inspected; video frame count, audio
streams, tape hash, and video hash passed the artifact checks.

## Previous assisted baseline

The earlier M8 Raphael result remains available under
`recordings/tmnt_iv_full_hard_credits*` and
`recordings/tmnt_iv_full_hard_dry_run.json`: 00:57:19.635, 65 emergency HP
restores, and 4,635 iframe-guard frames. It is not the Clean result.
Its old damage meter included stage-clear entity resets as damage.

No completion blocker remains. Future speed work is in [plan.md](plan.md).
