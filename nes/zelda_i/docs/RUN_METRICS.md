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

## Pre-l1 prefix (assist off)

| run | result | frames | flutter | note |
|---|---|---|---|---|
| baseline (92e7a315) | ok | 13133 | 3100 | 0x79/0x7A 2px bounce, ~2600 of the flutters |
| lattice walker (257dfeef) | ok | 6985 | 493 | pixel BFS obeys the ROM turn lattice |
