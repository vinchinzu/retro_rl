# RAM map — Super Double Dragon

Addresses are offsets in the `stable-retro` WRAM array for the USA ROM.

## Globals

| Address | Meaning | Notes |
|---------|---------|-------|
| `0x0018` | scene lock | `1` on `Area19_Clear`, `0` on the `Area1A` clone |
| `0x0019` | scene sub | `6` on `0x19`, `8` on `0x1A` |
| `0x001C` | internal area | Mission starts: `10,14,17,1C,1D,1F,20` |
| `0x00D9` | credits | observed as 5 in development runs |
| `0x00DC` | lives | zero is the last playable life |
| `0x00DE` | scene byte | `20` on `Area19_Clear`, `50` on the `Area1A` clone, `10` on the natural `0x1A` fade |
| `0x1CF9` | P1 actor page | high byte; `FF` during transitions |

## Actor pages

Actors occupy pages `0x0600` through `0x1700`.  Page `0x1700` is a live
kind-`07` fighter on `Area19_Clear` (the gym-stairs leftover) and a third
drawn enemy on the `Area1A` clone.  Camera X is unused (`0`); world X is
the actor word at `+0x0C`.

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
