# SM-KIHUNTER-CLIMB-RECON Report

## Result

RED diagnostic result. The natural-input sweep did not reach the upper band
(`y < 280`) in any trial. All 108 trials remained in Kihunter room `0xA4DA`
with `door_transition=0`; none entered Baby Kraid.

## Setup

- Source: `custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state`
- Room: `0xA4DA`; source pin was approximately `x=465, y=378`
- Budget: 480 frames per trial
- Grid: left launch frames `0, 4, ..., 32`; right caps `450, 460, 470, 475`
- Shot patterns: `up_shot`, `alternating_shot`, `held_up_shot`
- Inputs: ordinary controller actions only; no `place_samus` or room/door/RAM
  positioning

`min_y` is the minimum sampled Y in the Kihunter room. `x-range` is the
minimum and maximum sampled X. `final` is final `x/y`; all rows have
`upper=false`, meaning no sample met `room=0xA4DA`, `y<280`, and
`door_transition=0`.

## Launch Table

Each pattern cell is `min_y; x-range; final x/y; upper`.

| Left | Cap | `up_shot` | `alternating_shot` | `held_up_shot` |
|---:|---:|---|---|---|
| 0 | 450 | 371; 448..464; 455/395; false | 371; 446..464; 449/395; false | 371; 448..464; 450/395; false |
| 0 | 460 | 371; 450..471; 470/393; false | 371; 450..464; 462/395; false | 371; 450..464; 461/395; false |
| 0 | 470 | 371; 450..481; 475/371; false | 371; 450..477; 469/395; false | 371; 450..473; 471/395; false |
| 0 | 475 | 371; 450..486; 474/395; false | 371; 450..482; 475/395; false | 371; 450..475; 475/395; false |
| 4 | 450 | 371; 445..464; 452/386; false | 371; 446..464; 449/395; false | 371; 445..464; 450/395; false |
| 4 | 460 | 371; 445..471; 467/385; false | 371; 446..467; 460/395; false | 371; 445..464; 460/395; false |
| 4 | 470 | 371; 445..481; 470/395; false | 371; 446..477; 469/395; false | 371; 445..473; 470/395; false |
| 4 | 475 | 371; 445..486; 478/395; false | 371; 446..482; 475/395; false | 371; 445..477; 476/395; false |
| 8 | 450 | 371; 440..464; 452/371; false | 371; 441..464; 450/395; false | 371; 440..464; 449/395; false |
| 8 | 460 | 371; 440..471; 464/374; false | 371; 441..466; 460/395; false | 371; 440..464; 459/395; false |
| 8 | 470 | 371; 440..481; 469/395; false | 371; 441..477; 472/395; false | 371; 440..472; 472/395; false |
| 8 | 475 | 371; 440..486; 476/395; false | 371; 441..475; 473/395; false | 371; 440..477; 475/395; false |
| 12 | 450 | 371; 431..464; 453/395; false | 371; 432..464; 451/395; false | 371; 431..464; 449/395; false |
| 12 | 460 | 371; 431..471; 468/395; false | 371; 432..464; 459/395; false | 371; 431..464; 461/395; false |
| 12 | 470 | 371; 431..481; 479/393; false | 371; 432..477; 470/395; false | 371; 431..475; 470/395; false |
| 12 | 475 | 371; 431..486; 476/395; false | 371; 432..480; 475/395; false | 371; 431..477; 476/395; false |
| 16 | 450 | 371; 420..464; 457/372; false | 371; 421..464; 449/395; false | 371; 420..464; 450/395; false |
| 16 | 460 | 371; 420..471; 460/395; false | 371; 421..467; 460/395; false | 371; 420..467; 461/395; false |
| 16 | 470 | 371; 420..481; 474/379; false | 371; 421..475; 469/395; false | 371; 420..475; 471/395; false |
| 16 | 475 | 371; 420..486; 476/395; false | 371; 421..480; 474/395; false | 371; 420..477; 475/395; false |
| 20 | 450 | 371; 409..464; 456/395; false | 371; 410..464; 451/395; false | 371; 409..464; 448/395; false |
| 20 | 460 | 371; 409..475; 463/395; false | 371; 410..468; 460/395; false | 371; 409..475; 461/395; false |
| 20 | 470 | 371; 409..481; 475/395; false | 371; 410..477; 472/395; false | 371; 409..475; 469/395; false |
| 20 | 475 | 371; 409..486; 476/395; false | 371; 410..482; 477/395; false | 371; 409..477; 477/395; false |
| 24 | 450 | 371; 398..467; 452/395; false | 371; 399..464; 447/395; false | 371; 398..465; 450/395; false |
| 24 | 460 | 371; 398..477; 464/395; false | 371; 399..475; 461/395; false | 371; 398..475; 460/395; false |
| 24 | 470 | 371; 398..481; 474/395; false | 371; 399..476; 470/395; false | 371; 398..475; 468/395; false |
| 24 | 475 | 371; 398..486; 475/395; false | 371; 399..480; 475/395; false | 371; 398..475; 475/395; false |
| 28 | 450 | 371; 387..470; 454/374; false | 371; 388..464; 449/395; false | 371; 387..466; 452/395; false |
| 28 | 460 | 371; 387..480; 466/372; false | 371; 388..469; 459/395; false | 371; 387..475; 460/395; false |
| 28 | 470 | 371; 387..481; 470/395; false | 371; 388..475; 468/395; false | 371; 387..475; 470/395; false |
| 28 | 475 | 371; 387..486; 478/395; false | 371; 388..480; 472/395; false | 371; 387..477; 477/395; false |
| 32 | 450 | 371; 376..477; 451/395; false | 371; 377..472; 450/395; false | 371; 376..470; 450/395; false |
| 32 | 460 | 371; 376..477; 469/385; false | 371; 377..475; 461/395; false | 371; 376..475; 460/395; false |
| 32 | 470 | 371; 376..481; 476/395; false | 371; 377..475; 469/395; false | 371; 376..475; 470/395; false |
| 32 | 475 | 371; 376..486; 476/395; false | 371; 377..480; 474/395; false | 371; 376..477; 476/395; false |

## Recommendation

There is no successful launch recipe in this sweep. For the next residual
consumer, the least-bad numeric launch is:

- `left_frames=32`
- `right_cap=450`
- `shot_pattern=up_shot`
- observed `min_y=371`, `x=376..477`, final `x=451, y=395`,
  `door_transition=0`, `upper=false`

This reaches the requested lower launch band near `x=376`, but momentum still
peaks at `x=477`, and the height is 91 pixels short of the `y<280` criterion.
It should be treated as a diagnostic baseline only, not as a controller
recipe or pure-green result.

## Non-claims

- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, boss, or room state.
- Not pure-green evidence.
- Not continuous evidence.
