### Walk bill (zoffJ)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 177 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | cleared |
| `0x78` | 1221 | 238 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 4 | 4 | 0->4 | cleared |
| `0x79` | 925 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 3 | 5 | 4->7 | open |
| `0x7a` | 883 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 5 | 3 | 4 | 7->10 | open |
| `0x7b` | 943 | 156 | 3.00/3 -> 2.49/3 | 1.50 | 3 | leever_Ex2 leever_Sx1 | 7 | 7 | 7 | 10->4 (2 reset) | transit |
| `0x7c` | 1379 | 686 | 2.49/3 -> 0.98/3 | 1.51 | 3 | fireball_or_statue_projectile_Ex1 fireball_or_statue_projectile_Sx1 leever_Ex1 | 7 | 6 | 7 | 4->3 (2 reset) | open |
| `0x7d` | 617 | 121 | 0.98/3 -> 0.48/3 | 0.98 | 2 | octorok_blue_Ex1 rock_projectile_Ex1 | 0 | 0 | 5 | 3->0 (1 reset) | transit |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 4 | 0 | - | 4 | 0 | 0 | 0 | fairyx1 | 0 |
| `0x79` | 0 | 0 | tektite_bluex5 | 0 | 0 | 3 | 1 | clockx1 heartx1 | 0 |
| `0x7a` | 0 | 0 | tektite_bluex4 | 0 | 0 | 3 | 1 | 1Rx1 5Rx1 | 5 |
| `0x7b` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 7 | 1,3 | 1Rx2 5Rx1 clockx1 heartx1 | 7 |
| `0x7c` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 6 | 1,3 | 1Rx2 5Rx1 | 7 |
| `0x7d` | 0 | 4 | zorax1 | 0 | 0 | 0 | 2,3 | - | 0 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 4 | 0.62 |
| 1 | B / C | 0.594 | 0.891 | 18 | 16.03 |
| 3 | D / D | 0.406 | 0.081 | 1 | 0.08 |

23 kills -> **13 floor drops** (57% against 54% billed), E[R] 16.74.

### Run totals

- kills 23 census / 21 ROM counters
- rupees banked 19, final 19
- damage 4.00 hearts over 8 hits (8 iframe arms)
- streak best 10, resets 5
- hits by cause: {'leever_E': 3, 'leever_S': 1, 'fireball_or_statue_projectile_E': 1, 'fireball_or_statue_projectile_S': 1, 'rock_projectile_E': 1, 'octorok_blue_E': 1}
- prey passed on value: {'zora': 79}
- transit screens: ['0x7b', '0x7d'] (536f)
- duck frames: 231, shield 261
- blade presses 81, off-face 30, turn frames 0
- stages: sword_cave 749f ok=True, bomb_walk 6770f ok=False
- last stage notes: ['duck_wall_7d_up', 'duck_wall_7d_down', 'link_death']
- result ok=False failed=bomb_walk
