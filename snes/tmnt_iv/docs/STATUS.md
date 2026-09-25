# Status: TMNT IV: Turtles in Time

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M8 |
| Best verified result | Low-assist hard credits from power-on |
| Last verification | 2026-07-25 |
| Runtime class | Bronze |
| Intervention class | Resource-assisted + Protection-assisted |

One emulator session from power-on through hard-mode staff and cast credits.
Zero life losses, zero state loads, zero stage or lives writes, no A-special.
The assist contract is emergency HP (HP at or below 16 restored to 80) plus a
form-2 Super Shredder iframe hold at 1. Both are counted. See
`docs/ASSIST_CONTRACT.md`.

Published baseline on this date: 00:57:19.635, 4,667 damage, 65 emergency HP
restores, 4,635 iframe frames, zero life losses. Video
`recordings/tmnt_iv_full_hard_credits.mp4`. Dry-run manifest
`recordings/tmnt_iv_full_hard_dry_run.json`. Later scratch reports are not
this gate. Maturity stays M8 until a program decision. Clean would be the
same clear with both assists at zero. That clear is not verified.

Production boot selects Raphael. Stage byte `0x0082` value 0 is Big Apple
(human stage 1).

## Clean track (not the gate)

Stage 1 pizza-only suite is green (`probe_clean --stage 1 --suite`,
`recordings/stage1_clean_track/clean_suite.json`):

| Entry | Frames | Damage | Min HP |
|-------|--------|--------|--------|
| `Stage1` | 15,237 | 108 | 30 |
| power-on | 15,046 | 138 | 10 |
| Baxter | 5,323 | 40 | 44 |

Alleycat suite is 2/4. Boss and late pre-boss entries cleared. The full
`Stage2` checkpoint lost a life. The stage-1-clear bridge timed out. That is
not a Clean stage clear.

Sewer `LiveHardStage3` with heal=none lost a life. `Stage3` and `Boss3` are
last-life checkpoints and can die on the post-kill fade. Not green.

No Clean claim for Technodrome through Starbase, and no Clean power-on credits.

## Not done

- Power-on hard credits with zero emergency HP writes, zero form-2 iframe
  writes, and zero life losses.
- A sticky camera word. Progress still uses `0x003A`.
