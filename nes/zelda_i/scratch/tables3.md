### Walk bill (tables3)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 183 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | cleared |
| `0x78` | 114 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 3 | 0->0 | open |
| `0x68` | 566 | 322 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 2 | 4 | 0->2 | cleared |
| `0x58` | 646 | 397 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 3 | 4 | 2->5 | open |
| `0x59` | 203 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 5 | 5->5 | transit |
| `0x49` | 843 | 601 | 3.00/3 -> 1.99/3 | 1.00 | 2 | octorok_fast_Ex1 octorok_fast_Sx1 | 1 | 4 | 6 | 5->2 (2 reset) | cleared |
| `0x4a` | 2554 | 2401 | 1.99/3 -> 0.99/3 | 1.00 | 2 | tektite_blue_Nx1 tektite_blue_Sx1 | 0 | 2 | 6 | 2->0 (1 reset) | cleared |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 3 | 0 | - | 0 | 0 | 0 | 0 | - | 0 |
| `0x68` | 4 | 0 | - | 2 | 0 | 0 | 0 | - | 0 |
| `0x58` | 4 | 0 | - | 3 | 0 | 0 | 0 | - | 0 |
| `0x59` | 0 | 0 | peahatx4 zorax1 | 0 | 0 | 0 | 3 | - | 0 |
| `0x49` | 5 | 1 | - | 4 | 0 | 0 | 0,2 | 1Rx1 | 1 |
| `0x4a` | 0 | 0 | tektite_bluex6 | 0 | 0 | 2 | 1 | 1Rx1 | 0 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 9 | 1.41 |
| 1 | B / C | 0.594 | 0.891 | 2 | 1.78 |

11 kills -> **2 floor drops** (18% against 36% billed), E[R] 3.19.

### Run totals

- kills 11 census / 10 ROM counters
- rupees banked 1, final 1
- damage 2.01 hearts over 4 hits (4 iframe arms)
- streak best 5, resets 3
- hits by cause: {'octorok_fast_E': 1, 'octorok_fast_S': 1, 'tektite_blue_N': 1, 'tektite_blue_S': 1}
- prey passed on value: {}
- transit screens: ['0x59'] (132f)
- duck frames: 22, shield 78
- stages: sword_cave 749f ok=True, bomb_walk 5676f ok=True, bomb_buy 1f ok=False
- last stage notes: ['farm_need_20_have_1', 'shop_need_20_have_1']
- result ok=False failed=bomb_buy
