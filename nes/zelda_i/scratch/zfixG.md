### Walk bill (zfixG)

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 177 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | cleared |
| `0x78` | 902 | 0 | 3.00/3 -> 2.50/3 | 0.50 | 1 | rock_projectile_Ex1 | 0 | 4 | 4 | 0->0 (1 reset) | cleared |
| `0x79` | 765 | 601 | 2.50/3 -> 2.50/3 | 0.00 | 0 | - | 0 | 1 | 5 | 0->1 | open |
| `0x7a` | 810 | 616 | 2.50/3 -> 2.50/3 | 0.00 | 0 | - | 1 | 2 | 4 | 1->3 | open |
| `0x7b` | 892 | 262 | 2.50/3 -> 0.49/3 | 2.50 | 5 | fireball_or_statue_projectile_Ex2 fireball_or_statue_projectile_Sx1 leever_Nx1 leever_Wx1 | 1 | 5 | 7 | 3->1 (4 reset) | transit |

### The wave, red against blue

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 0 | 0 | - | 0 | 0 | 0 | - | - | 0 |
| `0x78` | 4 | 0 | - | 4 | 0 | 0 | 0 | - | 0 |
| `0x79` | 0 | 0 | tektite_bluex5 | 0 | 0 | 1 | 1 | heartx1 | 0 |
| `0x7a` | 0 | 0 | tektite_bluex4 | 0 | 0 | 2 | 1 | 1Rx1 clockx1 | 1 |
| `0x7b` | 0 | 0 | leeverx6 zorax1 | 0 | 0 | 5 | 1,3 | 1Rx1 | 1 |

### Drop rate against the ROM rows

| ROM row | letters (Baxter/locations) | billed P(drop) | E[R]/kill | kills measured | E[R] |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 4 | 0.62 |
| 1 | B / C | 0.594 | 0.891 | 6 | 5.34 |
| 3 | D / D | 0.406 | 0.081 | 2 | 0.16 |

12 kills -> **4 floor drops** (33% against 47% billed), E[R] 6.13.

### Run totals

- kills 12 census / 10 ROM counters
- rupees banked 2, final 2
- damage 3.00 hearts over 6 hits (6 iframe arms)
- streak best 4, resets 5
- hits by cause: {'rock_projectile_E': 1, 'fireball_or_statue_projectile_E': 2, 'fireball_or_statue_projectile_S': 1, 'leever_N': 1, 'leever_W': 1}
- prey passed on value: {'tektite_blue': 118}
- transit screens: ['0x7b', '0x7d'] (140f)
- duck frames: 25, shield 25
- stages: sword_cave 749f ok=True, bomb_walk 3963f ok=False
- last stage notes: ['duck_wall_7b_left', 'stall_escape_7b_120_149', 'link_death']
- result ok=False failed=bomb_walk
