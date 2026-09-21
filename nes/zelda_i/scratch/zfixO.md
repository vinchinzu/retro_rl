### Walk bill (zfixO)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 177 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | cleared |
| `0x78` | 1221 | 238 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 4 | 4 | 0->4 | cleared |
| `0x79` | 925 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 3 | 5 | 4->7 | open |
| `0x7a` | 891 | 626 | 3.00/3 -> 2.50/3 | 0.50 | 1 | tektite_blue_Wx1 | 5 | 3 | 4 | 7->1 (1 reset) | open |
| `0x7b` | 1026 | 221 | 2.50/3 -> 3.00/3 | 0.50 | 1 | leever_Ex1 | 7 | 6 | 7 | 1->2 (1 reset) | transit |
| `0x7c` | 1154 | 528 | 3.00/3 -> 1.49/3 | 1.50 | 3 | fireball_or_statue_projectile_Wx2 leever_Wx1 | 1 | 4 | 7 | 2->2 (2 reset) | open |
| `0x7d` | 977 | 38 | 1.49/3 -> 0.49/3 | 1.49 | 3 | fireball_or_statue_projectile_Wx1 rock_projectile_Ex2 | 0 | 1 | 5 | 2->0 (1 reset) | transit |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 4 | 0 | - | 4 | 0 | 0 | 0 | fairyx1 | 0 |
| `0x79` | 0 | 0 | tektite_bluex5 | 0 | 0 | 3 | 1 | clockx1 heartx1 | 0 |
| `0x7a` | 0 | 0 | tektite_bluex4 | 0 | 0 | 3 | 1 | 1Rx1 5Rx1 | 5 |
| `0x7b` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 6 | 1,3 | 1Rx2 5Rx1 clockx1 heartx2 | 7 |
| `0x7c` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 4 | 1,3 | 1Rx1 | 1 |
| `0x7d` | 0 | 4 | zorax1 | 0 | 1 | 0 | 2,3 | - | 0 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 4 | 0.62 |
| 1 | B / C | 0.594 | 0.891 | 16 | 14.25 |
| 2 | C / B | 0.406 | 0.122 | 1 | 0.12 |

21 kills -> **12 floor drops** (57% against 53% billed), E[R] 15.00.

### Run totals

- kills 21 census / 20 ROM counters
- rupees banked 13, final 13
- damage 4.00 hearts over 8 hits (8 iframe arms)
- streak best 9, resets 5
- hits by cause: {'tektite_blue_W': 1, 'leever_E': 1, 'leever_W': 1, 'fireball_or_statue_projectile_W': 3, 'rock_projectile_E': 2}
- prey passed on value: {'zora': 20}
- transit screens: ['0x7b', '0x7d'] (664f)
- duck frames: 230, shield 281
- blade presses 51, off-face 1, turn frames 37
- stages: sword_cave 749f ok=True, bomb_walk 6996f ok=False
- last stage notes: ['duck_wall_7d_left', 'lane_nogain_6_7d', 'link_death']
- result ok=False failed=bomb_walk
