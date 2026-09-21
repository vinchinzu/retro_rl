### Walk bill (zfixB)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 177 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | cleared |
| `0x78` | 1221 | 238 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 4 | 4 | 0->4 | cleared |
| `0x79` | 925 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 3 | 5 | 4->7 | open |
| `0x7a` | 883 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 5 | 3 | 4 | 7->10 | open |
| `0x7b` | 841 | 145 | 3.00/3 -> 0.48/3 | 4.00 | 8 | fireball_or_statue_projectile_Ex2 leever_Ex2 leever_Nx1 leever_Sx2 leever_Wx1 | 3 | 5 | 7 | 10->1 (5 reset) | transit |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 4 | 0 | - | 4 | 0 | 0 | 0 | fairyx1 | 0 |
| `0x79` | 0 | 0 | tektite_bluex5 | 0 | 0 | 3 | 1 | clockx1 heartx1 | 0 |
| `0x7a` | 0 | 0 | tektite_bluex4 | 0 | 0 | 3 | 1 | 1Rx1 5Rx1 | 5 |
| `0x7b` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 5 | 1,3 | 1Rx2 5Rx1 heartx1 | 3 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 4 | 0.62 |
| 1 | B / C | 0.594 | 0.891 | 10 | 8.91 |
| 3 | D / D | 0.406 | 0.081 | 1 | 0.08 |

15 kills -> **9 floor drops** (60% against 51% billed), E[R] 9.61.

### Run totals

- kills 15 census / 14 ROM counters
- rupees banked 8, final 8
- damage 4.00 hearts over 8 hits (8 iframe arms)
- streak best 10, resets 5
- hits by cause: {'leever_E': 2, 'fireball_or_statue_projectile_E': 2, 'leever_N': 1, 'leever_W': 1, 'leever_S': 2}
- prey passed on value: {}
- transit screens: ['0x7b', '0x7d'] (396f)
- duck frames: 68, shield 68
- stages: sword_cave 749f ok=True, bomb_walk 4464f ok=False
- last stage notes: ['duck_wall_7b_left', 'duck_wall_7b_right', 'link_death']
- result ok=False failed=bomb_walk
