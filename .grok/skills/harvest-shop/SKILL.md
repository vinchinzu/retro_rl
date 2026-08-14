---
name: harvest-shop
description: >
  Close a Harvest Moon seed/tool shop buy from RAM and a door landmark, not a
  full-day tape. Reject CrossMap "returned to origin" unless the shop interior
  and wallet/stock actually changed. Use when the user says "buy potato",
  "missed the shop", "seed store", "flower shop door", "buy_potato_seeds",
  or runs /harvest-shop.
---

# Harvest shop (door + wallet, not a day tape)

Picks are [harvest-interact](../harvest-interact/SKILL.md). Path/cliff hops are
[harvest-route](../harvest-route/SKILL.md).

`BUY_SEEDS` today is `CrossMapRecordedTask`: walk west off the farm, replay
`buy_potato_seeds` from `recording_start`, succeed on **farm tilemap again**
or tape end. That success is a lie if town `0x04` / shop `0x1C` never happened.

## This turn

1. **Analyze the tape you have.** Do not re-record first.
   ```bash
   uv run python -m harvest.runtime.task_recorder analyze buy_potato_seeds_d2
   ```
   Need: town `0x04` → shop `0x1C` → town, potato stock 0→1, money −200.
   Landmark is `shop_door` in `map_config` (do not invent a new pixel).
2. **Reject false returns.** `returned to origin map` with no `0x1C` and no
   potato delta is a miss. End-on-path (`0x0C`) is also unfinished — the
   day plan then thinks the shop is done while still off the farm.
3. **Prefer nav-to-door + RAM buy** over replaying the whole house→town
   walk. CrossMap should start *after* the live farm→path exit (slice at
   first `0x0C`) and must finish back on farm `0x00`.
4. **Record last, and only the gap.** If MultNav cannot enter the door from
   the live west-gate / town-square pin, record **door approach + buy +
   exit**, not farm→house preamble. F5 only when standing on the farm with
   `potato_seeds≥1` (or the seasonal bag) and money down by the bag price.
5. **Corners.** Shop-door and town-gate stasis (B+dir, tile unchanged) is
   [harvest-route](../harvest-route/SKILL.md) — stand on the open face
   (`shop_door` face up), do not hug the doorframe.

## Green

Shop interior seen, stock increased, wallet decreased, player back on farm.
Then wire `recording_name` / `recording_start` (or replace CrossMap with the
nav+buy skill). One grape is enough until this hop is green.

**Wired 2026-08-13:** spring `BUY_SEEDS` is `BuySeedsTask` (`shop_buy`).
Probe `recordings/buy_seeds_d2_probe.json`: `0x1C`, potato 0→1, money
300→100, farm `0x00`. Do not replay `buy_potato_seeds` from frame 483.
