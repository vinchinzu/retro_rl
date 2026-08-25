## Residual — rr-tne2 L6 Survival compose (stairs / Gohma / TF `0x20`)

**Status:** OPEN. Bead `rr-tne2`.
**Pin:** after `--through level6-clear3a` 1/1 leftover play `0x3A`
`(144,141)` rod=1 keys=4 bow=0 arrows=0 TF=`0x1F` map=`0x0A`; center 0x68
unpushed. Next: stairs / Gohma / TF `0x20`. Do not grant Map. Do not poke
doors/keys/arrows. Do not invent Gohma room id.

Do **not** STATUS-promote. Do not overwrite Clean M5. Glance leave with
`zelda_i.screen_glance` — no MP4. `--no-video` on spine CLIs. Occupancy
halt at first miss ([predict-path](../../../../.grok/skills/predict-path/SKILL.md)).

### Already green (do not re-prove)

Closed L1–L5 Survival spine. L6 hops through `--through level6-clear3a` are
historical 1/1 (quoted from AGENTS Next; not new greens):
`l6_clear3a_continuous_v1` leftover play `0x3A` `(144,141)` hop 1,857f tape
219,649f rod=1 keys=4 bow=0 arrows=0. Prefix: `level6-clear09` 210,699f hop
1,419f leftover `(112,173)` keys=4 bombs=8 TF=`0x1F` map=`0x0A`;
`level6-room09` v2 play `0x09` `(120,205)` keys 5→4; `level6-stairs09` mode 9
`0x75` `(208,93)`; `level6-rod` `l6_rod_continuous_v15` 627f `(136,141)`
rod=1; `level6-exit75` `l6_exit75_continuous_v3` 1,525f / 213,054f play
`0x09` `(192,141)`; `level6-south09` 251f / 213,305f play `0x19` `(120,77)`;
`level6-south19` 663f / 213,968f play `0x29` `(120,77)` keys 4→3;
`level6-clear29` `l6_clear29_continuous_v2` 1,406f / 215,534f leftover
`(55,133)` keys 3→4; `level6-south29` `l6_south29_continuous_v4` 288f /
215,822f play `0x39` `(120,93)`; `level6-settle39` 5× Vire `0x12`;
`level6-clear39` 1,330f leftover `(136,173)`; `level6-east39`
`l6_east39_continuous_v3` 320f / 217,632f play `0x3A` `(16,141)`;
`level6-settle3a` 3×`0x17`+2×`0x23`+2×`0x24`+0x68. `--through level6-east29`
stayed red (sealed). Skip Map. Bow=0 arrows=0 (L1 bow skipped).

### Next action

- **One change:** the Gohma/stairs/TF `0x20` checkbox. Do not invent Gohma
  room id. Do not grant Map. Do not poke doors/keys/arrows.
- **Glance:** `zelda_i.screen_glance` — room hex, mode, x/y band, TF bits,
  earned keys/bombs, hearts lo==hi. `--no-video`. Halt after 3 serial reds
  on the same checkbox.
- Occupancy halt at first miss (`predict-path`). Do not batch exploration.

### Non-claims

- Did not STATUS-promote
- Did not overwrite Clean M5
- Did not poke doors/keys/arrows/undiscovered items
- Did not grant Map/Whistle
