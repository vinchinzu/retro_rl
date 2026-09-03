# L7-B side rooms — 0x68 DOWN + 0x58 north (2026-09-03)

Bead `rr-8t4.2`. Did not STATUS. Did not poke `ADDR_CANDLE` / `ADDR_LADDER`
/ `ADDR_MAX_BOMBS` / TF / doors. Did not walk `0x49` UP. Assist:
`--infinite-life`, `progression_writes=0`, `capacity_writes=0`.

Start fixtures: `Level7Interior68ReconFixture` (0x68 `~(208,93)`, keys 4,
bombs 7, Food 1) and `Level7Interior58ReconFixture` (0x58 `(120,205)`,
keys 4, bombs 7, Food 1).

## Recon table

| walk | dest `$EB` | mode | entry xy | keys/bombs | census | notes | evidence |
|------|-----------|------|----------|------------|--------|-------|----------|
| `0x68` DOWN OPEN | `0x78` | 5 | `(120,77)` N mouth | 4 / 7 (`max_bombs` 8) | ropes `0x28`; floor `small_key 0x19` | dead-end; key not auto-picked | **2/2** `68_down_v3/v4` + `68_ctl_v1/v2` (frame 297/298) |
| `0x58` UP KEY | `0x48` | 5 | `(120,205)` S mouth | 4→3 / 7 (`max_bombs` 8) | bubble `0x40` + `0x4f` | old-man "I BET YOU'D LIKE TO HAVE -100"; 100-rupee bomb-capacity dead-end; do not write `max_bombs` | **2/2** `58_north_v2/v3` + `58_ctl_v1/v2` (frame 337/338) |

`route_eligible=false` on every fixture. Dest pins:
`Level7Interior78ReconFixture`, `Level7Interior48ReconFixture`
(`development_only` / `fixture_only` / `natural_entry=false`).

## Geometry

**0x68 (KEESE_TRAPS):** dark, 4 blade traps `0x49` in the corners + 4 keese
`0x1b`. OPEN south door. OccupancyWalker miss-blocked trap knockback and
stood at `(174,149)` (v2, 12 misses). Waypoint: peel `x=160`, drop `y=141`,
align `x=120`, push DOWN; if knocked onto the `y~189` trap row off-x, rise
first.

**0x58 (DODONGOS_UPGRADE):** 3× invuln `0x31` (hp 240) — dodge. Central
2-block mass walls the `x=120` column around `y=141`. OccupancyWalker to
`(120,93)` boxed at `(122,165)` (v1, 25 misses). East-around: `(120,165)`
→ `(160,165)` → `y=93` → `x=120` → push UP. KEY door spends one key.

## Wired

`Room68DownController` / `make_room68_down_controller`,
`Room58NorthController` / `make_room58_north_controller`. Graph:
`ROPES_KEY ram_id=0x78`, `BOMB_UPGRADE ram_id=0x48`, both
`evidence=fixture-live`. Not on the executable chapter chain.

## Leftover

Candle mainline still blocked at `0x49` pending Stepladder (sibling owns
`0x49` UP). These two side rooms are done. Do not pay the 100-rupee bomb
upgrade (0 rupees on the recon pin; capacity writes stay 0).
