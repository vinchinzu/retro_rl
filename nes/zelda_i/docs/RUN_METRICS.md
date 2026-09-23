# Run metrics — milestones toward a clean credits video

One row per measured continuous run, newest last. The row comes from
`scripts/run_metrics.py <report.json>`; the ledger behind it is
`spine/ledger.py` (wraps `env.step`, observation only).

- **frames**: power-on to the end of the last stage.
- **flutter**: Link's position reversing on one axis within 8 frames outside
  knockback, meaning a visible left-right or up-down twitch. Lower is better to watch.
- **drops picked**: enemy floor drops picked up / seen. The rest expired or
  were left behind on a room change.
- **hearts missed**: heart/fairy drops not picked up. Under the refill each one
  is healing the assist paid for instead.
- **hits / refills**: assist damage events / refill writes. Under `--engage-hearts 1`
  (last-heart) each refill is a death the assist prevented.
- **poked b/k/R**: bombs, keys and rupees granted by Survival top-ups.
- **slowest**: the longest single room visit, which is the first stall to look at.

A deterministic emulator gives one outcome per config, so one run is the
measurement. Compare a row against the row above it only when the code changed.

| run | result (assist) | through | frames | flutter | drops picked | hearts missed | hits / refills | poked b/k/R | slowest |
|---|---|---|---|---|---|---|---|---|---|
| full_poweron12 (92e7a315, = poweron11 code) | ok (unlimited_health) | level9-credits | 292742 | 17942 | 67/176 | 32 | 441 / 667 | 82 / 6 / 9 | 9:03 12842f |
| **full_poweron27 (b0a328e9)** | **ok (unlimited_health)** | level9-credits | **277687** | **8178** | 66/163 | 42 | 371 / 596 | 67 / 6 / 0 | 7:0d 14024f |
| full_poweron26 (3ce0a056), stopped in L9 | level9_natural_silver_arrows (unlimited_health) | level9-credits | 270230 | 6529 | 68/177 | 47 | 324 / 533 | 55 / 5 / 0 | 9:4f 13993f |

## Stabilization loop after the lattice walkers (2026-09-23)

Each continuous run failed one stage later than the last; each stall was
fixed from its `R<n>_<stage>` save point and the next run started from
power-on. Frames and flutter count only up to the failure.

| run | failed stage | TF | frames | flutter | slowest visit | fix |
|---|---|---|---|---|---|---|
| full_poweron13 | north | 0x00 | 39136 | 716 | 1:74 3552f | L1 return-west on the lattice; dungeon door lanes |
| full_poweron14 | enter_6f_key | 0x01 | 62619 | 3033 | 2:6e 5806f | 0x6E key door lattice-only |
| full_poweron16 | level5_whistle_stairs_64 | 0x0F | 129348 | 5823 | 5:64 7131f | stairs_step first; wait for the wave; door retry |
| full_poweron17 | level7_room19_east_bomb | 0x3F | 195890 | 11521 | 0:52 7194f | bomb re-place (3x) |
| full_poweron18 | level5 west_did_not_enter_24 | 0x0F | 140771 | 5293 | 2:6e 5118f | L5 west hops = exit_door |
| full_poweron19 | level7_pond_approach | 0x3F | 209046 | 37106 | 0:55 29890f | raft 0x61 not a combatant |
| full_poweron20 | enter_level4 | 0x07 | 132867 | 1889 | 0:45 37835f | L4 mouth approach row south of the mouth |
| full_poweron21 | level5 west_did_not_enter_25 | 0x0F | - | - | 5:26 | moat: ladder_release; west hops retry then clear |
| full_poweron22 | level4_west_0x31 | 0x07 | - | - | 4:32 | L4 west door lattice; maze ladder release |
| full_poweron23 | level6_clear_0x29 | 0x1F | - | - | 6:29 | latched ladder crossing heading |
| full_poweron25 | level6_clear_0x29 | 0x1F | - | - | 6:29 | engine backs off a ladder that goes nowhere |
| full_poweron26 | level9_natural_silver_arrows | 0xFF | 270230 | 6529 | 9:4f 13993f | passes on later code from the same pose |

## Last-heart refill (`--engage-hearts 1`)

The refill writes only at the last heart, so `refills` counts deaths the
assist prevented. This is the number potions and better combat must drive to zero.

| run | through | result | frames | refills (deaths prevented) | hits | note |
|---|---|---|---|---|---|---|
| lasth_l1 (fed68421) | level1 | ok | 48874 | 2 | 11 | gather chain keeps its own last-heart refill |

## Pre-l1 prefix (assist off)

| run | result | frames | flutter | note |
|---|---|---|---|---|
| baseline (92e7a315) | ok | 13133 | 3100 | 0x79/0x7A 2px bounce, ~2600 of the flutters |
| lattice walker (257dfeef) | ok | 6985 | 493 | pixel BFS obeys the ROM turn lattice |
