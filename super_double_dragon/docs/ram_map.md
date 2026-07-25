# RAM map — Super Double Dragon

Addresses are offsets in the `stable-retro` WRAM array for the USA ROM.

## Globals

| Address | Meaning | Notes |
|---------|---------|-------|
| `0x001C` | internal area | Mission starts: `10,14,17,1C,1D,1F,20` |
| `0x00D9` | credits | observed as 5 in development runs |
| `0x00DC` | lives | zero is the last playable life |
| `0x1CF9` | P1 actor page | high byte; `FF` during transitions |

## Actor pages

Actors occupy pages `0x0600` through `0x1600`.

| Offset | Meaning |
|--------|---------|
| `+0x00` | status (`03` drawn/active) |
| `+0x02` | kind / character |
| `+0x0C` | world X (little-endian word) |
| `+0x10` | logical lane Y |
| `+0x27` | HP |
| `+0x72` | rendered Y helper |
| `+0x74` | rendered screen X |

P1 must be selected through `0x1CF9`; actor kind is not stable across
scenes. HP-zero fighters remain targetable while status is `03` and disappear
only after their final knockdown.

The normalized parser uses `255 - logical_y` because pressing UP increases the
raw logical-Y byte in this game.
