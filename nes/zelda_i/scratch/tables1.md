### Walk bill (tables1)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | cleared |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 183 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | yes |
| `0x78` | 114 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 3 | 0->0 | no |
| `0x68` | 566 | 322 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 2 | 4 | 0->2 | yes |
| `0x58` | 646 | 397 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 3 | 4 | 2->5 | no |
| `0x59` | 739 | 533 | 3.00/3 -> 1.99/3 | 1.00 | 2 | fireball_or_statue_projectile_Ex1 peahat_Sx1 | 0 | 1 | 5 | 5->0 (1 reset) | no |
| `0x49` | 684 | 434 | 1.99/3 -> 1.49/3 | 0.50 | 1 | octorok_blue_Wx1 | 1 | 6 | 6 | 0->4 (1 reset) | yes |
| `0x4a` | 2618 | 2401 | 1.49/3 -> 1.99/3 | 0.50 | 1 | tektite_blue_Ex1 | 1 | 4 | 6 | 4->1 (1 reset) | yes |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 3 | 0 | - | 0 | 0 | 0 | 0 | - | 0 |
| `0x68` | 4 | 0 | - | 2 | 0 | 0 | 0 | - | 0 |
| `0x58` | 4 | 0 | - | 3 | 0 | 0 | 0 | - | 0 |
| `0x59` | 0 | 0 | peahatx4 zorax1 | 0 | 0 | 1 | 3 | - | 0 |
| `0x49` | 5 | 1 | - | 5 | 1 | 0 | 0,2 | 1Rx1 | 1 |
| `0x4a` | 0 | 0 | tektite_bluex6 | 0 | 0 | 4 | 1 | 1Rx1 heartx1 | 1 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 10 | 1.56 |
| 1 | B / C | 0.594 | 0.891 | 4 | 3.56 |
| 2 | C / B | 0.406 | 0.122 | 1 | 0.12 |
| 3 | D / D | 0.406 | 0.081 | 1 | 0.08 |

16 kills -> **3 floor drops** (19% against 39% billed), E[R] 5.33.

### Run totals

- kills 16 census / 15 ROM counters
- rupees banked 2, final 2
- damage 2.01 hearts over 4 hits (4 iframe arms)
- streak best 7, resets 3
- hits by cause: {'fireball_or_statue_projectile_E': 1, 'peahat_S': 1, 'octorok_blue_W': 1, 'tektite_blue_E': 1}
- stages: sword_cave 749f, bomb_walk 6117f, bomb_buy 1f
- result ok=False failed=bomb_buy
