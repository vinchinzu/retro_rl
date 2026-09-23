> Historical lane note. The live sitting is the gathering prefix in [PRE_L1.md](../PRE_L1.md). This file stays because the clean-tip ladder or a route doc still cites it. It is not the current plan.

# rr-npv.3 — Clean L7 Entrance→TF natural bait

Fixture-live only. Do not STATUS. Do not poke `ADDR_FOOD`.

## This sitting (2026-09-10)

Clean path: `survival=False` / `allow_pokes=False` never writes `ADDR_FOOD`.
Hungry Goriya fail-closes with leftover `hungry_goriya_requires_food` when
Food is 0. Forced Digdogger is a leftover-relative `HopController`: dest is
RAM (type `0x38` gone / shrunk dead / play `0x0C`), not 12×B or
`STAND_SETTLE` idle.

### Pin

`Level7Entrance` (whistle-poke recon, `natural_entry=false`):

```text
L7 play 0x79 (120,205) mode 5 TF 0x00 Food 0 bombs 0 keys 0
rupees 0 ladder 0 whistle 1 health 0x22 (3 HC) 
```

Provenance stays fixture-only. Cannot buy Bait from this pin.

### ONE ROM trial (no assist, no pokes, `--no-video`)

`make_env(..., "Level7Entrance")`, red-candle + complete chapters,
`assist=None`. Stop at first red.

1. `level7_entry_first_door` green 251f → play `0x69` `(120,205)`
2. **first red** `level7_room69_west_bomb` timeout 22000f

Leftover glance:

```text
L7 0x69 (34,165) mode 8 TF 0x00 Food 0 bombs 0 keys 0
health 0x20 (lo nibble 0) notes=['timeout']
food_poked=false route_eligible=false deaths(mode 17)=0
```

Mode 8 + empty hearts: combat drained the 3-heart pin before PLACE.
`no_bombs` is the honest gate (pin bombs=0) but the hop kill-clears first
and only notes `timeout`. Hungry (`hungry_goriya_requires_food`) is
downstream and unreached.

TF `0x40` not landed. Do not retry this pin without bombs+Food.

## Next

Need a pin that already holds Food (bought, not poked) and bombs, or wait
on `rr-8t4.4` / `.5` natural bait. Interior recon fixtures that already
carry Food (`Level7Interior28ReconFixture`) are a later isolated hop, not
Entrance→TF. Do not STATUS.
