# Final Fight RAM map

Addresses are WRAM offsets for stable-retro `get_ram()` / `data.json`
(USA). Primary source: [TCRF Notes: Final Fight (SNES)](https://tcrf.net/Notes:Final_Fight_(SNES)).
Confirm with a probe before trusting a new combat rule.

## Game and stage

| Name | Addr | Type | Notes |
|------|------|------|-------|
| `game_status` | `0x0CA0` | u8 | `00` char select, `02`/`04` open stage, `06` play, `08` clear area, `0A` clear round, `0E` Break Car bonus |
| `round` | `0x0CB0` | u8 | `00` Slum, `01` Subway, `02` West Side, `03` Industrial, `04` Bay, `05` Up Town, `06` Break Car, `07` Break Glass |
| `area` | `0x0CB1` | u8 | Sub-area within the round |
| `rounds_cleared` | `0x0CB2` | u8 | Incremented on the clear-round handler (3 at West Side entry) |
| `level_end` | `0x0CD0` | u8 | Write `01` forces level end (stuck if the boss is not dead). TCRF |
| `boss_dead_flag` | `0x0CD2` | u8 | `01` when certain bosses die. Ends the level script. TCRF. Damnd and Sodom underflow left this at 0 |
| `go_flashing` | `0x0CD7` | u8 | `01` when the GO arrow flashes |
| `char_select` | `0x008F` | u8 | `00` Cody, `01` Haggar |
| `camera_x` | `0x0E07` | u16 LE | Scroll X in the round |

Writing `game_status` to `0x08` is a development bridge, not a Clean clear.

## Player 1 (base `0x0D00`)

| Name | Addr | Type | Notes |
|------|------|------|-------|
| `player_active` | `0x0D00` | u8 | `00` inactive, `01` active |
| `player_x` | `0x0D07` | u16 LE | World X |
| `player_y` | `0x0D0D` | u16 LE | Ground Y (`0x0D0A` is jump Y). Up increases Y |
| `player_hp` | `0x0D14` | u8 | Current HP (max typically `0x80` at `0x0D18`) |
| `player_lives` | `0x0D6E` | u8 | Lives remaining (HUD is one less) |

## Enemies and boss

The entity layout matches the player. Slots:

| Slot | Base | Notes |
|------|------|-------|
| Enemy 0 | `0x1000` | |
| Enemy 1 | `0x10B0` | stride `0xB0` |
| Enemy 2 | `0x1140` | stride `0xB0` |
| Boss | `0x11E0` | status `00` none, `01` present undrawn, `03` drawn |

Per-slot fields: `+0x00` status, `+0x07` X u16, `+0x0D` Y u16, `+0x14` HP.
Boss HP is also at `0x11F4`.

## Parsing

- Living fighters use status `0x03`. Status `0x02` is junk. Status `0x01` is spawn or despawn, except a living HP near the camera, which still chips.
- Regular thug HP is at most `0x80`. Subway tough HP can be 148 (living max 224). West Side Andore is about 216. Area 1 peaks near 250. Underflow ghosts are at or above 253 and still chip until planted.
- Camera 1536 is the alley lock, not the Damnd door. Boss status `01` means door entry. Damnd draws (`03`) only after the door thugs. Do not treat clear-area `0x08` as that spawn.
