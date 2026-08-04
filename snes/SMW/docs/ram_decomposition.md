# RAM Decomposition

## Rule

Every RAM claim should include:

- WRAM address, usually `$7E:xxxx`
- stable-retro RAM offset, usually `0xxxxx`
- type and endian
- source reference
- local verification command or trace once available

The initial map comes from SMWDisX `rammap.asm`, SMW Central's RAM map, and
stable-retro's bundled SMW integration. Emulator traces become the final local
truth once a ROM is installed.

## Autoplay-Critical Fields

| Field | WRAM | Offset | Type | Purpose |
| --- | ---: | ---: | --- | --- |
| `true_frame` | `$7E:0013` | `0x0013` | `u8` | lag-sensitive frame counter |
| `effective_frame` | `$7E:0014` | `0x0014` | `u8` | gameplay frame counter |
| `powerup` | `$7E:0019` | `0x0019` | `u8` | small/big/cape/fire |
| `camera_x` | `$7E:001A` | `0x001A` | `u16` | horizontal progress |
| `camera_y` | `$7E:001C` | `0x001C` | `u16` | vertical progress |
| `player_animation` | `$7E:0071` | `0x0071` | `u8` | death, pipe, door, frozen states |
| `player_in_air` | `$7E:0072` | `0x0072` | `u8` | jump/fall state |
| `player_direction` | `$7E:0076` | `0x0076` | `u8` | facing left/right |
| `player_blocked_dir` | `$7E:0077` | `0x0077` | `u8` | collision flags |
| `player_x_speed` | `$7E:007A` | `0x007A` | `s16` | 4.12 fixed-point X speed |
| `player_y_speed` | `$7E:007C` | `0x007C` | `s16` | 4.12 fixed-point Y speed |
| `player_x_next` | `$7E:0094` | `0x0094` | `u16` | next-frame X |
| `player_y_next` | `$7E:0096` | `0x0096` | `u16` | next-frame Y |
| `player_x` | `$7E:00D1` | `0x00D1` | `u16` | current-frame X |
| `player_y` | `$7E:00D3` | `0x00D3` | `u16` | current-frame Y |
| `game_mode` | `$7E:0100` | `0x0100` | `u8` | active level/overworld/menu state |
| `lives` | `$7E:0DBE` | `0x0DBE` | `s8` | death/life-loss detection |
| `coins` | `$7E:0DBF` | `0x0DBF` | `u8` | status and trace sanity |
| `item_box` | `$7E:0DC2` | `0x0DC2` | `u8` | powerup reserve |
| `level_timer_frames` | `$7E:0F30` | `0x0F30` | `u8` | timer frame subcounter |
| `level_timer_hundreds` | `$7E:0F31` | `0x0F31` | `u8` | timer hundreds digit |
| `level_timer_tens` | `$7E:0F32` | `0x0F32` | `u8` | timer tens digit |
| `level_timer_ones` | `$7E:0F33` | `0x0F33` | `u8` | timer ones digit |
| `translevel` | `$7E:13BF` | `0x13BF` | `u8` | level identity/overworld node |
| `current_submap` | `$7E:13C3` | `0x13C3` | `u8` | overworld submap |
| `midway_flag` | `$7E:13CE` | `0x13CE` | `u8` | checkpoint state |
| `p_meter` | `$7E:13E4` | `0x13E4` | `u8` | run speed/takeoff readiness |
| `on_ground` | `$7E:13EF` | `0x13EF` | `u8` | ground contact |
| `active_boss` | `$7E:13FC` | `0x13FC` | `u8` | boss context |
| `camera_scrolling` | `$7E:13FD` | `0x13FD` | `u8` | camera transition state |

## Decomposition Milestones

### Bronze

- RAM fields above are readable through `data.json`.
- `retro_harness/platformer/levels/super_mario_world.py` can evaluate progress and
  detect basic completion/death.
- One trace fixture confirms field stability during a level.

### Silver

- Named sprite slot decoder for `$7E:009E` and related 12-slot tables.
- Player state enum for animations, movement, cape flight, climbing, water,
  pipe, door, death, and boss locks.
- Route fingerprint containing translevel, game mode, player position, powerup,
  timer, and RNG-relevant bytes.

### Gold

- Full WRAM catalog for player, sprites, extended sprites, cluster sprites,
  overworld, timers, events, item memory, stripe image upload buffers, and
  camera/scroll state.
- Tests that diff states before/after exits, keyholes, switches, bosses,
  midpoint tape, and overworld movement.

### Editor-Ready

- ROM model links RAM fields to decoded level/overworld/sprite structures.
- Live emulator snapshots can be projected onto editor coordinates.
- Editor writes patch/build data, not raw emulator memory mutations.
