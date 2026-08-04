# Status — Mortal Kombat II (SNES)

## Program gate

| Field | Value |
|-------|-------|
| States | **134 extracted** (CharSelect + 12 Fight_* + tournament stages) |
| Training | Multi-char PPO via `retro_harness.fighters` (in progress) |
| Integration | MK2 custom integrations under this game dir |

## State inventory

- `CharSelect_MortalKombatII.state`
- `Fight_{Char}.state` for all 12 characters (Match 1)
- Tournament: Match 2–8, ShangTsung, Kintaro, ShaoKahn × 12 chars

## RAM

### data.json (&lt; 0x2000)

| Variable | Address | Hex |
|----------|---------|-----|
| fatality_timer | 562 | 0x0232 |

### High WRAM (DirectRAMReader; get_ram = WRAM + 0x2001)

| Variable | WRAM | get_ram | Notes |
|----------|------|---------|-------|
| health (P1) | 0x2EFC | 0x4EFD | Starts 161 |
| enemy_health (P2) | 0x30AA | 0x50AB | P1/P2 structs 0x1AE apart |

0x020A / 0x020E are transitional state values — **not** health.
