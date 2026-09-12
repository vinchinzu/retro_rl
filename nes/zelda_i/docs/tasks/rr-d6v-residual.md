# rr-d6v residual — Clean L6 Entrance→TF (no Survival)

**BLOCKED** (3/3 0x59 y-band chase). East-waist LEFT/spacing also
**blocked 3/3** (v4–v6). Stopped at fixture-live. Not a STATUS claim.
`route_eligible=false`. Do not close the bead. Do not spawn another
0x59-chase / east-waist try.

## Pin glance

`Level6Entrance` play **0x79** `(120,205)` mode **5**. Isolated pin: TF
`0x00`, keys **0**, bombs **0**, bow **0**, arrows **0**, rod **0**, raft
**0**, ladder **0**, health **0x2F** (3 HC, lo nibble `0xF` display — not
`0x22` lo==hi). No L5 inventory.

Runner: `scripts/run_level6_entrance_tf.py --from-state Level6Entrance
--no-infinite-life --no-video --trials 1 --tag l6_entrance_tf_clean_v9`.
`make_env` + `reset_obs` + `resync_custom_state`. Assist None.
`poke_arrows=False`. `allow_pokes=False`.

## Landed this sitting (south hold + same-x sidestep; still red → blocked)

Hypothesis: commit south of y=157 (hold; no UP-flip from 149 onto 141).
If 0x24 shares Link's x within 16px, LEFT/RIGHT off that column first.

Unit: leftover `(160,149)` + `0x24@(160,141)` + `0x59` y=141 →
`wizzrobe_sidestep` LEFT/RIGHT, not UP. `QT_QPA_PLATFORM=offscreen uv run
pytest nes/zelda_i/tests/test_level6*.py -q` → 123 passed.

v4–v8 evidence kept. This trial is `l6_entrance_tf_clean_v9`.

## Stages until first unfixed red

| stage | frames | ok |
|-------|--------|----|
| `level6_right_0x7a` | 374 | yes |
| `level6_east_key_0x7a` | 1084 | yes (keys 0→1, bombs 0→4 natural) |
| `level6_return_0x79` | 305 | yes |
| `level6_west_key_0x78` | 381 | yes (key spent, keys 1→0) |
| `level6_west_clear_0x78` | 300 | **no** |

## Leftover (stop)

room **0x78**, mode **17** (death / CONTINUE), xy **(144,141)**, TF
**0x00**, keys **0**, bombs **4**, health **0x20** (3 HC, lo=0), deaths
**1**, bow **0**, arrows **0**, rod **0**. PNG
`recordings/l6_entrance_tf_clean_v9_final.png`. Total 2444f.

Sidestep on the waist walked back onto y=141 at x=144 (same 300f death).
0x59@(144,152) and @(120,141) share that pose.

## Object census (last play frame, f299)

shot_types: **0x59**. Notes: `shot_type_0x59_s8_f101`.

| slot | type | xy | hp |
|------|------|----|----|
| 1 | 0x24 | (192,141) | 64 |
| 2 | 0x24 | (96,141) | 64 |
| 3 | 0x24 | (192,141) | 64 |
| 4 | 0x24 | (144,173) | 64 |
| 5 | 0x24 | (208,141) | 64 |
| 8 | 0x59 | (171,141) | 128 |
| 9 | 0x59 | (144,152) | 128 |
| 10 | 0x59 | (187,141) | 128 |
| 11 | 0x59 | (120,141) | 128 |

## Class split

| tag | xy | west_clear f | class |
|-----|----|--------------|-------|
| v4 | (189,141) | 300 | east-waist **blocked 3/3** |
| v5 | (192,140) | 301 | east-waist |
| v6 | (176,141) | 300 | east-waist |
| v7 | (120,125) | 302 | 0x59 chase |
| v8 | (160,149) | 92 | 0x59 chase |
| v9 | **(144,141)** | 300 | 0x59 chase **blocked 3/3** |

## Next hop

**Blocked.** Do not resume 0x59-chase or east-waist LEFT. New geometry
only if a later sitting fights from a hold that is not y=141 and does
not sidestep *on* the waist. Do not poke arrows. Do not reopen Survival
Gohma. `rr-d6v` stays OPEN. `route_eligible=false`.
