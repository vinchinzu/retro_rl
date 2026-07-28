# RAM map — Mike Tyson's Punch-Out!! (NES)

Sources: Data Crystal, TASVideos MTPO RAM map, in-ring probes (Glass Joe).

## Fight / opponent

| Addr | Name | Notes |
|------|------|-------|
| `0x0000` | fight started | 1 in bout |
| `0x0001` | opp id | |
| `0x0002` | opp type | 0 = Glass Joe |
| `0x0004` | fight flag | `0xFF` in ring, `0x01` between rounds |
| `0x0005` | knockdown | non-zero while a fighter is down |
| `0x0006` | round | 1–3 |
| `0x0039` | opp pattern timer | next action at 0 |
| `0x003A` | opp action id | attack phase |
| `0x003B` | opp pattern set | 115 open, **150 backup/taunt**, 185 attack |

## Clock / meters

| Addr | Name | Notes |
|------|------|-------|
| `0x0300` | clock on | 1 when round clock runs |
| `0x0302` | clock minutes | |
| `0x0304` / `0x0305` | clock display digits | used as tenths/seconds fields |
| `0x0323` / `0x0324` | hearts tens/ones | |
| `0x034A` | stars raw | value = stars + `0x40` |

## Health

| Addr | Name | Notes |
|------|------|-------|
| `0x0390` | Mac health init | 1 when fight HP live |
| `0x0391` | Mac health | max `0x60` (96) |
| `0x0398` / `0x0399` | opp health next/cur | |

## Helpers (`ram.py`)

- `is_level1_ready` — both health bars live (M1 boot).
- `is_match_live` — ready + clock on + in-fight flag.
- `is_taunt_window` — Glass Joe Vive La France backup (`pattern_set == 150`).
