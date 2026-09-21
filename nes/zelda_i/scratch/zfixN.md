### Walk bill (zfixN)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 177 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | cleared |
| `0x78` | 1221 | 238 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 4 | 4 | 0->4 | cleared |
| `0x79` | 925 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 3 | 5 | 4->7 | open |
| `0x7a` | 891 | 626 | 3.00/3 -> 2.50/3 | 0.50 | 1 | tektite_blue_Wx1 | 5 | 3 | 4 | 7->1 (1 reset) | open |
| `0x7b` | 1011 | 221 | 2.50/3 -> 3.00/3 | 0.50 | 1 | fireball_or_statue_projectile_Wx1 | 2 | 6 | 7 | 1->2 (1 reset) | transit |
| `0x7c` | 1425 | 163 | 3.00/3 -> 1.99/3 | 2.01 | 4 | fireball_or_statue_projectile_Ex2 leever_Nx1 leever_Sx1 | 2 | 6 | 7 | 2->3 (3 reset) | transit |
| `0x7d` | 676 | 0 | 1.99/3 -> 0.48/3 | 1.99 | 4 | fireball_or_statue_projectile_Sx1 unk_0x63_Nx3 | 0 | 1 | 5 | 3->1 (2 reset) | transit |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 4 | 0 | - | 4 | 0 | 0 | 0 | fairyx1 | 0 |
| `0x79` | 0 | 0 | tektite_bluex5 | 0 | 0 | 3 | 1 | clockx1 heartx1 | 0 |
| `0x7a` | 0 | 0 | tektite_bluex4 | 0 | 0 | 3 | 1 | 1Rx1 5Rx1 | 5 |
| `0x7b` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 6 | 1,3 | 1Rx2 clockx1 heartx2 | 2 |
| `0x7c` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 6 | 1,3 | 1Rx2 heartx1 | 2 |
| `0x7d` | 0 | 4 | zorax1 | 0 | 1 | 0 | 2,3 | - | 0 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 4 | 0.62 |
| 1 | B / C | 0.594 | 0.891 | 18 | 16.03 |
| 2 | C / B | 0.406 | 0.122 | 1 | 0.12 |

23 kills -> **13 floor drops** (57% against 54% billed), E[R] 16.78.

### Run totals

- kills 23 census / 23 ROM counters
- rupees banked 9, final 9
- damage 5.00 hearts over 10 hits (10 iframe arms)
- streak best 9, resets 7
- hits by cause: {'tektite_blue_W': 1, 'fireball_or_statue_projectile_W': 1, 'fireball_or_statue_projectile_E': 2, 'leever_S': 1, 'leever_N': 1, 'unk_0x63_N': 3, 'fireball_or_statue_projectile_S': 1}
- prey passed on value: {}
- transit screens: ['0x7b', '0x7c', '0x7d'] (954f)
- duck frames: 160, shield 160
- blade presses 56, off-face 1, turn frames 30
- stages: sword_cave 749f ok=True, bomb_walk 6951f ok=False
- last stage notes: ['duck_wall_7d_up', 'stall_escape_7d_96_85', 'link_death']
- result ok=False failed=bomb_walk
