### Walk bill (zfixK)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 177 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | cleared |
| `0x78` | 1221 | 238 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 4 | 4 | 0->4 | cleared |
| `0x79` | 925 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 3 | 5 | 4->7 | open |
| `0x7a` | 891 | 626 | 3.00/3 -> 2.50/3 | 0.50 | 1 | tektite_blue_Wx1 | 5 | 3 | 4 | 7->1 (1 reset) | open |
| `0x7b` | 1152 | 234 | 2.50/3 -> 1.49/3 | 2.01 | 4 | fireball_or_statue_projectile_Ex1 leever_Nx2 leever_Sx1 | 6 | 6 | 7 | 1->2 (3 reset) | transit |
| `0x7c` | 417 | 233 | 1.49/3 -> 0.48/3 | 1.49 | 3 | fireball_or_statue_projectile_Sx1 leever_Ex1 leever_Sx1 | 4 | 5 | 7 | 2->2 (3 reset) | open |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 4 | 0 | - | 4 | 0 | 0 | 0 | fairyx1 | 0 |
| `0x79` | 0 | 0 | tektite_bluex5 | 0 | 0 | 3 | 1 | clockx1 heartx1 | 0 |
| `0x7a` | 0 | 0 | tektite_bluex4 | 0 | 0 | 3 | 1 | 1Rx1 5Rx1 | 5 |
| `0x7b` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 6 | 1,3 | 1Rx1 5Rx1 heartx1 | 6 |
| `0x7c` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 5 | 1,3 | 1Rx3 5Rx1 | 4 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 4 | 0.62 |
| 1 | B / C | 0.594 | 0.891 | 17 | 15.14 |

21 kills -> **12 floor drops** (57% against 54% billed), E[R] 15.77.

### Run totals

- kills 21 census / 21 ROM counters
- rupees banked 15, final 15
- damage 4.00 hearts over 8 hits (7 iframe arms)
- streak best 9, resets 7
- hits by cause: {'tektite_blue_W': 1, 'fireball_or_statue_projectile_E': 1, 'leever_S': 2, 'leever_N': 2, 'fireball_or_statue_projectile_S': 1, 'leever_E': 1}
- prey passed on value: {}
- transit screens: ['0x7b', '0x7d'] (386f)
- duck frames: 74, shield 74
- blade presses 45, off-face 5, turn frames 44
- stages: sword_cave 749f ok=True, bomb_walk 5304f ok=False
- last stage notes: ['hop_4_7c', 'duck_wall_7c_down', 'link_death']
- result ok=False failed=bomb_walk
