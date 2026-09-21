### Walk bill (zoffI)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 177 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | cleared |
| `0x78` | 1221 | 238 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 4 | 4 | 0->4 | cleared |
| `0x79` | 925 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 3 | 5 | 4->7 | open |
| `0x7a` | 868 | 614 | 3.00/3 -> 2.50/3 | 0.50 | 1 | tektite_blue_Wx1 | 0 | 2 | 4 | 7->0 (1 reset) | open |
| `0x7b` | 1196 | 101 | 2.50/3 -> 1.49/3 | 2.01 | 4 | fireball_or_statue_projectile_Ex1 leever_Nx2 leever_Wx1 | 5 | 5 | 7 | 0->1 (2 reset) | transit |
| `0x7c` | 964 | 626 | 1.49/3 -> 0.48/3 | 1.00 | 2 | fireball_or_statue_projectile_Ex1 leever_Wx1 | 1 | 6 | 7 | 1->5 (1 reset) | open |
| `0x7d` | 186 | 102 | 0.48/3 -> 0.48/3 | 0.48 | 1 | octorok_blue_Nx1 | 0 | 0 | 5 | 5->5 (1 reset) | transit |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 4 | 0 | - | 4 | 0 | 0 | 0 | fairyx1 | 0 |
| `0x79` | 0 | 0 | tektite_bluex5 | 0 | 0 | 3 | 1 | clockx1 heartx1 | 0 |
| `0x7a` | 0 | 0 | tektite_bluex4 | 0 | 0 | 2 | 1 | 1Rx1 | 0 |
| `0x7b` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 5 | 1,3 | 1Rx1 5Rx1 heartx1 | 5 |
| `0x7c` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 6 | 1,3 | 1Rx1 clockx1 | 1 |
| `0x7d` | 0 | 4 | zorax1 | 0 | 0 | 0 | 2,3 | - | 0 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 4 | 0.62 |
| 1 | B / C | 0.594 | 0.891 | 15 | 13.36 |
| 3 | D / D | 0.406 | 0.081 | 1 | 0.08 |

20 kills -> **9 floor drops** (45% against 53% billed), E[R] 14.07.

### Run totals

- kills 20 census / 19 ROM counters
- rupees banked 6, final 6
- damage 4.00 hearts over 8 hits (12 iframe arms)
- streak best 9, resets 5
- hits by cause: {'tektite_blue_W': 1, 'fireball_or_statue_projectile_E': 2, 'leever_N': 2, 'leever_W': 2, 'octorok_blue_N': 1}
- prey passed on value: {'zora': 9}
- transit screens: ['0x7b', '0x7d'] (315f)
- duck frames: 118, shield 118
- blade presses 54, off-face 3, turn frames 56
- stages: sword_cave 749f ok=True, bomb_walk 6162f ok=False
- last stage notes: ['lane_nogain_5_7c', 'hop_5_7d', 'link_death']
- result ok=False failed=bomb_walk
