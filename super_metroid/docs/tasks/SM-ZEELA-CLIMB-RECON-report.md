# SM-ZEELA-CLIMB-RECON Report

## Scope

Diagnostic-only input recon from `scratch/post_kihunter_to_zeela_return.state`
(`0xA471` Zeela), not a pure-green or continuous run. Each trial booted a
fresh emulator state, used the resource-only assist, attempted a natural
morph-roll setup toward `x=60`, `x=120`, or `x=190`, and then ran one of eight
input classes for up to 360 frames. The setup did not reach the requested
bands: every trial settled at `strategy_start_x=281`. This is itself evidence
that the lower-floor geometry, rather than target selection, dominated this
grid.

## Trial Table

| Target x | Strategy | Strategy start x | Min y | End room | End x/y/pose | Door transition | Frames |
|---:|---|---:|---:|---|---|---|---:|
| 60 | standing A spam | 281 | 398 | 0xA471 | 275/409/65 | no | 360 |
| 60 | crouch-load A | 281 | 398 | 0xA471 | 278/409/65 | no | 360 |
| 60 | left wall-run + Hi-Jump | 281 | 398 | 0xA471 | 117/409/31 | no | 360 |
| 60 | right wall-run + Hi-Jump | 281 | 398 | 0xA471 | 411/409/30 | no | 360 |
| 60 | morph bomb cycle | 281 | 398 | 0xA471 | 278/409/65 | no | 360 |
| 60 | morph-left bomb cycle | 281 | 334 | 0xA4B1 | 20/371/82 | yes | 304 |
| 60 | forward-drop reverse shot | 281 | 331 | 0xA471 | 32/382/82 | no | 360 |
| 60 | wall-run crouch-load | 281 | 398 | 0xA471 | 117/409/31 | no | 360 |
| 120 | standing A spam | 281 | 398 | 0xA471 | 275/409/65 | no | 360 |
| 120 | crouch-load A | 281 | 398 | 0xA471 | 278/409/65 | no | 360 |
| 120 | left wall-run + Hi-Jump | 281 | 398 | 0xA471 | 117/409/31 | no | 360 |
| 120 | right wall-run + Hi-Jump | 281 | 398 | 0xA471 | 411/409/30 | no | 360 |
| 120 | morph bomb cycle | 281 | 398 | 0xA471 | 278/409/65 | no | 360 |
| 120 | morph-left bomb cycle | 281 | 334 | 0xA4B1 | 20/371/82 | yes | 304 |
| 120 | forward-drop reverse shot | 281 | 331 | 0xA471 | 32/382/82 | no | 360 |
| 120 | wall-run crouch-load | 281 | 398 | 0xA471 | 117/409/31 | no | 360 |
| 190 | standing A spam | 281 | 398 | 0xA471 | 275/409/65 | no | 360 |
| 190 | crouch-load A | 281 | 398 | 0xA471 | 278/409/65 | no | 360 |
| 190 | left wall-run + Hi-Jump | 281 | 398 | 0xA471 | 117/409/31 | no | 360 |
| 190 | right wall-run + Hi-Jump | 281 | 398 | 0xA471 | 411/409/30 | no | 360 |
| 190 | morph bomb cycle | 281 | 398 | 0xA471 | 278/409/65 | no | 360 |
| 190 | morph-left bomb cycle | 281 | 334 | 0xA4B1 | 20/371/82 | yes | 304 |
| 190 | forward-drop reverse shot | 281 | 331 | 0xA471 | 32/382/82 | no | 360 |
| 190 | wall-run crouch-load | 281 | 398 | 0xA471 | 117/409/31 | no | 360 |

## Findings

- `best_min_y=331`, from `forward_drop_reverse_shot`, repeated identically
  for all three requested target bands. This breaks the `y=395` floor pin but
  does not reach the forward middle-band target of `y<=325`.
- `morph_left_bomb_cycle` also breaks the floor pin at `min_y=334`, but exits
  into `0xA4B1` with `door_transition=1`; it is not a usable reverse-climb
  result without separately understanding that neighboring-room transition.
- Standing, crouch-load, both wall-run variants, morph bomb, and wall-run
  crouch-load stayed floor-pinned at `min_y=398` and ended near `y=409`.
- The nominal x-band sweep was ineffective: natural setup ended at `x=281`
  for every trial, so no conclusion about x-specific ledges is justified.

## Recommendation

`forward_drop_reverse_shot` is the one maneuver class worth carrying into
`SM-K4-R-03D`: open the reverse drop with `UP+X`, then use a jump/leftward
shot cadence rather than standing A spam or a wall-run hold. R-03D must first
re-test its setup geometry because this recon only reached `y=331`, not the
`y<=325` middle band, and never reached the warehouse door band (`y<=200`).

**Next card ID:** `SM-K4-R-03D`  
**One change:** test the forward-drop reverse-shot class with one named setup
geometry adjustment that can move the launch from the natural `x=281` floor
position into the `x≈105` middle-drop lane.  
**Source state:** `scratch/post_kihunter_to_zeela_return.state`

## Non-claims

- Did not claim pure-green.
- Did not claim continuous evidence or promote STATUS.
- Did not forge progression, capacity, door, event, or boss RAM.
- The probe used only ordinary inputs and the resource-only assist; its
  generated machine report is `debug/zeela_climb_recon.json`.

## Probe Pin

Best stayed in room `0xA471`: `pose=82`, `x=32`, `y=382`,
`door_transition=0`. The alternate `min_y=334` result ended in room `0xA4B1`:
`pose=82`, `x=20`, `y=371`, `door_transition=1`.
