### Walk bill (zfixD)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 177 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | cleared |
| `0x78` | 1221 | 238 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 4 | 4 | 0->4 | cleared |
| `0x79` | 925 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 3 | 5 | 4->7 | open |
| `0x7a` | 883 | 601 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 5 | 3 | 4 | 7->10 | open |
| `0x7b` | 2156 | 156 | 3.00/3 -> 0.98/3 | 3.01 | 6 | fireball_or_statue_projectile_Ex1 fireball_or_statue_projectile_Sx2 leever_Ex3 | 6 | 9 | 7 | 10->1 (4 reset) | transit |
| `0x7c` | 1365 | 642 | 0.98/3 -> 1.48/3 | 0.50 | 1 | fireball_or_statue_projectile_Sx1 | 0 | 7 | 7 | 1->6 (1 reset) | open |
| `0x7d` | 719 | 251 | 1.48/3 -> 0.98/3 | 0.50 | 1 | rock_projectile_Ex1 | 1 | 2 | 5 | 6->0 (1 reset) | transit |
| `0x7e` | 249 | 189 | 0.98/3 -> 0.48/3 | 0.50 | 1 | octorok_fast_Nx1 | 0 | 3 | 5 | 0->0 (1 reset) | open |
| `0x7f` | 353 | 172 | 0.48/3 -> 0.48/3 | 0.00 | 0 | - | 0 | 1 | 2 | 0->1 | open |
| `0x6f` | 1 | 0 | 0.48/3 -> 0.48/3 | 0.00 | 0 | - | 0 | 0 | 0 | 1->1 | open |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 4 | 0 | - | 4 | 0 | 0 | 0 | fairyx1 | 0 |
| `0x79` | 0 | 0 | tektite_bluex5 | 0 | 0 | 3 | 1 | clockx1 heartx1 | 0 |
| `0x7a` | 0 | 0 | tektite_bluex4 | 0 | 0 | 3 | 1 | 1Rx1 5Rx1 | 5 |
| `0x7b` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 9 | 1,3 | 1Rx2 5Rx1 heartx1 | 6 |
| `0x7c` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 7 | 1,3 | heartx1 | 0 |
| `0x7d` | 0 | 4 | zorax1 | 0 | 2 | 0 | 2,3 | 1Rx1 | 1 |
| `0x7e` | 4 | 0 | zorax1 | 3 | 0 | 0 | 0,3 | - | 0 |
| `0x7f` | 0 | 1 | zorax1 | 0 | 1 | 0 | 2,3 | - | 0 |
| `0x6f` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 7 | 1.09 |
| 1 | B / C | 0.594 | 0.891 | 18 | 16.03 |
| 2 | C / B | 0.406 | 0.122 | 3 | 0.37 |
| 3 | D / D | 0.406 | 0.081 | 4 | 0.33 |

32 kills -> **11 floor drops** (34% against 49% billed), E[R] 17.82.

### Run totals

- kills 32 census / 28 ROM counters
- rupees banked 12, final 12
- damage 4.52 hearts over 9 hits (8 iframe arms)
- streak best 10, resets 7
- hits by cause: {'leever_E': 3, 'fireball_or_statue_projectile_E': 1, 'fireball_or_statue_projectile_S': 3, 'rock_projectile_E': 1, 'octorok_fast_N': 1}
- prey passed on value: {}
- transit screens: ['0x7b', '0x7d'] (1114f)
- duck frames: 269, shield 295
- stages: sword_cave 749f ok=True, bomb_walk 8967f ok=True, bomb_topup 95f ok=False
- last stage notes: ['link_death']
- result ok=False failed=bomb_topup
