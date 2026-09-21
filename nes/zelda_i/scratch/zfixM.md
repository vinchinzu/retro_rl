### Walk bill (zfixM)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 177 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | cleared |
| `0x78` | 1221 | 238 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 4 | 4 | 0->4 | cleared |
| `0x79` | 925 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 3 | 5 | 4->7 | open |
| `0x7a` | 891 | 626 | 3.00/3 -> 2.50/3 | 0.50 | 1 | tektite_blue_Wx1 | 5 | 3 | 4 | 7->1 (1 reset) | open |
| `0x7b` | 1011 | 221 | 2.50/3 -> 3.00/3 | 0.50 | 1 | fireball_or_statue_projectile_Wx1 | 2 | 6 | 7 | 1->2 (1 reset) | transit |
| `0x7c` | 2185 | 616 | 3.00/3 -> 0.49/3 | 2.51 | 5 | fireball_or_statue_projectile_Ex1 fireball_or_statue_projectile_Sx1 fireball_or_statue_projectile_Wx2 leever_Ex1 | 6 | 4 | 7 | 2->0 (4 reset) | open |
| `0x7d` | 527 | 77 | 0.49/3 -> 0.49/3 | 0.49 | 1 | rock_projectile_Ex1 | 0 | 2 | 5 | 0->2 (1 reset) | transit |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 4 | 0 | - | 4 | 0 | 0 | 0 | fairyx1 | 0 |
| `0x79` | 0 | 0 | tektite_bluex5 | 0 | 0 | 3 | 1 | clockx1 heartx1 | 0 |
| `0x7a` | 0 | 0 | tektite_bluex4 | 0 | 0 | 3 | 1 | 1Rx1 5Rx1 | 5 |
| `0x7b` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 6 | 1,3 | 1Rx2 clockx1 heartx2 | 2 |
| `0x7c` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 4 | 1,3 | 1Rx1 5Rx1 | 6 |
| `0x7d` | 0 | 4 | zorax1 | 0 | 2 | 0 | 2,3 | 1Rx1 bombx1 | 0 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 4 | 0.62 |
| 1 | B / C | 0.594 | 0.891 | 16 | 14.25 |
| 2 | C / B | 0.406 | 0.122 | 2 | 0.24 |

22 kills -> **14 floor drops** (64% against 53% billed), E[R] 15.12.

### Run totals

- kills 22 census / 22 ROM counters
- rupees banked 13, final 13
- damage 4.00 hearts over 8 hits (8 iframe arms)
- streak best 9, resets 7
- hits by cause: {'tektite_blue_W': 1, 'fireball_or_statue_projectile_W': 3, 'fireball_or_statue_projectile_E': 1, 'fireball_or_statue_projectile_S': 1, 'leever_E': 1, 'rock_projectile_E': 1}
- prey passed on value: {'zora': 78}
- transit screens: ['0x7b', '0x7d'] (480f)
- duck frames: 230, shield 252
- blade presses 52, off-face 3, turn frames 52
- stages: sword_cave 749f ok=True, bomb_walk 7562f ok=False
- last stage notes: ['hop_5_7d', 'duck_wall_7d_up', 'link_death']
- result ok=False failed=bomb_walk
