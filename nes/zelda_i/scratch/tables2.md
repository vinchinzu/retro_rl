### Walk bill (tables2)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 183 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | cleared |
| `0x78` | 114 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 3 | 0->0 | open |
| `0x68` | 789 | 531 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 1 | 4 | 4 | 0->4 | cleared |
| `0x58` | 1407 | 310 | 3.00/3 -> 2.50/3 | 0.50 | 1 | octorok_Ex1 | 0 | 4 | 4 | 4->2 (1 reset) | cleared |
| `0x59` | 214 | 0 | 2.50/3 -> 2.50/3 | 0.00 | 0 | - | 0 | 1 | 5 | 2->2 | transit |
| `0x49` | 431 | 172 | 2.50/3 -> 0.49/3 | 2.50 | 5 | octorok_fast_Ex1 octorok_fast_Nx1 octorok_fast_Sx1 octorok_fast_Wx2 | 0 | 2 | 6 | 2->0 (1 reset) | open |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 3 | 0 | - | 0 | 0 | 0 | 0 | - | 0 |
| `0x68` | 4 | 0 | - | 4 | 0 | 0 | 0 | 1Rx1 heartx1 | 1 |
| `0x58` | 4 | 0 | - | 4 | 0 | 0 | 0 | - | 0 |
| `0x59` | 0 | 0 | peahatx4 zorax1 | 0 | 0 | 1 | 3 | - | 0 |
| `0x49` | 5 | 1 | - | 2 | 0 | 0 | 0,2 | - | 0 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 10 | 1.56 |
| 3 | D / D | 0.406 | 0.081 | 1 | 0.08 |

11 kills -> **2 floor drops** (18% against 32% billed), E[R] 1.64.

### Run totals

- kills 11 census / 10 ROM counters
- rupees banked 1, final 1
- damage 3.00 hearts over 6 hits (2 iframe arms)
- streak best 6, resets 2
- hits by cause: {'octorok_E': 1, 'octorok_fast_N': 1, 'octorok_fast_W': 2, 'octorok_fast_S': 1, 'octorok_fast_E': 1}
- prey passed on value: {}
- transit screens: ['0x59'] (128f)
- duck frames: 18, shield 41
- stages: sword_cave 749f ok=True, bomb_walk 3602f ok=False
- last stage notes: ['hop_3_59', 'hop_4_49', 'link_death']
- result ok=False failed=bomb_walk
