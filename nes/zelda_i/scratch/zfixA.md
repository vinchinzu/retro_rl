### Walk bill (zfixA)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 177 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | cleared |
| `0x78` | 1221 | 238 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 4 | 4 | 0->4 | cleared |
| `0x79` | 925 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 3 | 5 | 4->7 | open |
| `0x7a` | 883 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 5 | 3 | 4 | 7->10 | open |
| `0x7b` | 995 | 60 | 3.00/3 -> 0.99/3 | 2.01 | 4 | leever_Ex4 | 5 | 7 | 7 | 10->2 (4 reset) | transit |
| `0x7c` | 875 | 601 | 0.99/3 -> 0.49/3 | 0.50 | 1 | fireball_or_statue_projectile_Ex1 | 1 | 2 | 7 | 2->2 (1 reset) | open |
| `0x7d` | 633 | 0 | 0.49/3 -> 0.49/3 | 0.00 | 0 | - | 0 | 0 | 5 | 2->2 | transit |
| `0x7e` | 234 | 140 | 0.49/3 -> 0.49/3 | 0.49 | 1 | octorok_fast_Nx1 | 0 | 0 | 5 | 2->2 (1 reset) | open |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 4 | 0 | - | 4 | 0 | 0 | 0 | fairyx1 | 0 |
| `0x79` | 0 | 0 | tektite_bluex5 | 0 | 0 | 3 | 1 | clockx1 heartx1 | 0 |
| `0x7a` | 0 | 0 | tektite_bluex4 | 0 | 0 | 3 | 1 | 1Rx1 5Rx1 | 5 |
| `0x7b` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 7 | 1,3 | 5Rx1 | 5 |
| `0x7c` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 2 | 1,3 | 1Rx1 clockx1 | 1 |
| `0x7d` | 0 | 4 | zorax1 | 0 | 0 | 0 | 2,3 | - | 0 |
| `0x7e` | 4 | 0 | zorax1 | 0 | 0 | 0 | 0,3 | - | 0 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 4 | 0.62 |
| 1 | B / C | 0.594 | 0.891 | 13 | 11.58 |
| 3 | D / D | 0.406 | 0.081 | 2 | 0.16 |

19 kills -> **8 floor drops** (42% against 51% billed), E[R] 12.37.

### Run totals

- kills 19 census / 17 ROM counters
- rupees banked 11, final 11
- damage 3.00 hearts over 6 hits (25 iframe arms)
- streak best 10, resets 6
- hits by cause: {'leever_E': 4, 'fireball_or_statue_projectile_E': 1, 'octorok_fast_N': 1}
- prey passed on value: {}
- transit screens: ['0x7b', '0x7d'] (442f)
- duck frames: 110, shield 113
- stages: sword_cave 749f ok=True, bomb_walk 6672f ok=False
- last stage notes: ['lane_nogain_6_7d', 'hop_6_7e', 'link_death']
- result ok=False failed=bomb_walk
