### Walk bill (zbase1)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 177 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | cleared |
| `0x78` | 1221 | 238 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 4 | 4 | 0->4 | cleared |
| `0x79` | 925 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 3 | 5 | 4->7 | open |
| `0x7a` | 883 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 5 | 3 | 4 | 7->10 | open |
| `0x7b` | 945 | 148 | 3.00/3 -> 1.99/3 | 1.50 | 3 | fireball_or_statue_projectile_Sx1 leever_Ex2 | 0 | 6 | 7 | 10->2 (3 reset) | transit |
| `0x7c` | 819 | 613 | 1.99/3 -> 0.99/3 | 1.00 | 2 | fireball_or_statue_projectile_Ex1 leever_Sx1 | 7 | 5 | 7 | 2->3 (2 reset) | open |
| `0x7d` | 546 | 148 | 0.99/3 -> 0.49/3 | 0.50 | 1 | octorok_blue_Ex1 | 0 | 2 | 5 | 3->1 (1 reset) | transit |
| `0x7e` | 139 | 98 | 0.49/3 -> 0.49/3 | 0.49 | 1 | fireball_or_statue_projectile_Ex1 | 0 | 3 | 5 | 1->4 (1 reset) | open |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 4 | 0 | - | 4 | 0 | 0 | 0 | fairyx1 | 0 |
| `0x79` | 0 | 0 | tektite_bluex5 | 0 | 0 | 3 | 1 | clockx1 heartx1 | 0 |
| `0x7a` | 0 | 0 | tektite_bluex4 | 0 | 0 | 3 | 1 | 1Rx1 5Rx1 | 5 |
| `0x7b` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 6 | 1,3 | heartx1 | 0 |
| `0x7c` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 5 | 1,3 | 1Rx3 5Rx1 clockx1 | 7 |
| `0x7d` | 0 | 4 | zorax1 | 0 | 2 | 0 | 2,3 | - | 0 |
| `0x7e` | 4 | 0 | zorax1 | 3 | 0 | 0 | 0,3 | - | 0 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 7 | 1.09 |
| 1 | B / C | 0.594 | 0.891 | 15 | 13.36 |
| 2 | C / B | 0.406 | 0.122 | 2 | 0.24 |
| 3 | D / D | 0.406 | 0.081 | 2 | 0.16 |

26 kills -> **11 floor drops** (42% against 49% billed), E[R] 14.86.

### Run totals

- kills 26 census / 23 ROM counters
- rupees banked 12, final 12
- damage 3.50 hearts over 7 hits (17 iframe arms)
- streak best 10, resets 7
- hits by cause: {'leever_E': 2, 'fireball_or_statue_projectile_S': 1, 'leever_S': 1, 'fireball_or_statue_projectile_E': 2, 'octorok_blue_E': 1}
- prey passed on value: {'zora': 43}
- transit screens: ['0x7b', '0x7d'] (790f)
- duck frames: 79, shield 116
- stages: sword_cave 749f ok=True, bomb_walk 6384f ok=False
- last stage notes: ['hop_5_7d', 'hop_6_7e', 'link_death']
- result ok=False failed=bomb_walk
