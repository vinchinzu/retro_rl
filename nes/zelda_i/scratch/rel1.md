### Walk bill (rel1)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 183 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | cleared |
| `0x78` | 114 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 3 | 0->0 | open |
| `0x68` | 566 | 322 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 2 | 4 | 0->2 | cleared |
| `0x58` | 646 | 397 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 3 | 4 | 2->5 | open |
| `0x59` | 203 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 5 | 5->5 | transit |
| `0x49` | 907 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 2 | 6 | 6 | 5->11 | cleared |
| `0x4a` | 2190 | 1978 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 6 | 6 | 6 | 11->17 | cleared |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 3 | 0 | - | 0 | 0 | 0 | 0 | - | 0 |
| `0x68` | 4 | 0 | - | 2 | 0 | 0 | 0 | - | 0 |
| `0x58` | 4 | 0 | - | 3 | 0 | 0 | 0 | - | 0 |
| `0x59` | 0 | 0 | peahatx4 zorax1 | 0 | 0 | 0 | 3 | - | 0 |
| `0x49` | 5 | 1 | - | 5 | 1 | 0 | 0,2 | 1Rx2 5Rx1 | 2 |
| `0x4a` | 0 | 0 | tektite_bluex6 | 0 | 0 | 6 | 1 | 1Rx2 5Rx1 fairyx1 heartx1 | 6 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 10 | 1.56 |
| 1 | B / C | 0.594 | 0.891 | 6 | 5.34 |
| 2 | C / B | 0.406 | 0.122 | 1 | 0.12 |

17 kills -> **8 floor drops** (47% against 42% billed), E[R] 7.03.

### Run totals

- kills 17 census / 17 ROM counters
- rupees banked 8, final 8
- damage 0.00 hearts over 0 hits (0 iframe arms)
- streak best 17, resets 0
- hits by cause: {}
- prey passed on value: {}
- transit screens: ['0x59'] (132f)
- duck frames: 18, shield 64
- stages: sword_cave 749f ok=True, bomb_walk 5376f ok=True, bomb_buy 1f ok=False
- last stage notes: ['farm_need_20_have_8', 'shop_need_20_have_8']
- result ok=False failed=bomb_buy
