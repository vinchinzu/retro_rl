# Living residual

Planner owns `docs/STATUS.md`. Overwrite this file only. Leave proof is a RAM glance, not an MP4. A save-state pin is not continuous. Do not claim a new continuous tip. Do not overwrite `recordings/verified_tip_run.json` on a red.

Session: `.grok/skills/alttp-session/SKILL.md`. Claim exactly one bead: `bd ready -l alttp -l spine`.

## Spine leftover (2026-09-20)

Continuous tip is still room `0x50`. `--through room_01`, `room_72`, and `zelda` fail closed.

Graph class, not a STATUS promote:

| Hop | Class |
|-----|--------|
| `room_50_east_to_0x01` | natural_entry |
| `room_01_down_to_0x72` | natural_entry (hold UP at the north-wall well; not DOWN) |
| `room_72_north_to_0x01` | isolated reverse |
| `room_72_south_to_0x82` | isolated |
| `room_82_west_to_0x81` | isolated |
| `room_01_to_zelda_cell` | planned |
| `0x81` `west_to_0x80` | not isolated |

Open red (`rr-ccxt.19`, state-load): `CastleB1SecondKey` wraps out of the `0x71` east pocket (NW lip UP+LEFT at y=3960; LEFT at pin y=3988 is the south jamb) onto `0x81` (632, 4155) keys=1, `$F3CC==0`, layer `$00EE==1`. LEFT at (608, 4169) stays `0x81`. Key not spent. `$F366`/`$F367` still 0. Gold jail door. `landingXy` still null.

Well leftover, not continuous: `0x50` east lands `0x01` near (560, 120); walk to (760, 99) and hold UP to `0x72` near (1273, 3665). `CastleB1Key` on the north ledge spends the small key on the south key door.

Glance windows in `alttp.screen_glance` (not the continuous tip): `ROOM_50` x[436, 492] y[2668, 2692]; `ROOM_01` x[756, 764] y[97, 124]; `ROOM_72` x[1260, 1285] y[3644, 3677]. Module `0x07`, submodule `0`, sword at least 1, follower unset. `leftover_from_snapshot` still returns leftover when misses is non-empty. `0x50` east landing near (560, 120) misses the `ROOM_01` well band.

`CastleZeldaFollower` is room `0x80` (352, 4168) with `$F3CC==1` as loaded. There is no pre-rescue cell pin.

## Dungeon-start pins (dev only)

Epic `rr-ldhk`. Not power-on, not continuous, not Sanctuary. Pokes stick at `$7EFxxx` (`assign(0xF3xx)` has no mapping). Do not poke `$F3CC`.

| Bead | Pin | Open next |
|------|-----|-----------|
| `rr-y8xg` | `EasternPalaceEntry` `0xC9` | `south_to_overworld` isolated. Next: north hall / `0xB9` |
| `rr-alwe` | `DesertPalaceEntry` `0x84` | `north_to_0x74` isolated |
| `rr-17wv` | `HeraEntry` `0x77` | probe isolated `west_down_to_0x87`; `room_engine` wedges on pegs |
| `rr-9zkv` | `PalaceOfDarknessEntry` `0x4A` | `east_stairs_to_0x09` isolated; `in_room` rejects dark world |
| `rr-q41r` | `SwampPalaceEntry` `0x28` | `south_to_overworld` isolated; `north_to_0x38` not isolated |
| `rr-lvnu` | no indoor pin | dark-world overworld `0x5B`. Detail stays in [SkullWoodsEntry-residual.md](SkullWoodsEntry-residual.md). Next: overworld `0x40` entrance `0x29` to room `0x58` |
| `rr-yk6q` | `ThievesTownEntry` `0xDB` | `south_to_outcasts` isolated; `east_to_0xDC` conveyor red |
| `rr-3yxm` | `IcePalaceEntry` `0x0E` | Freezor melt + LEFT still `0x0E`. Next: Bari key / north stairs |

## Next action

One spine bead on the `0x81` west door. The small key did not spend. Big-key bytes were 0 on that load and were not tried. The follow-up named in the sitting was layer versus door type. Do not STATUS-promote. Do not call the well or this door continuous.

## Non-claims

Did not STATUS-promote. Did not treat a pin as power-on. Did not write `$F3CC`, sword, or keys. Did not delete `z3-json-data/` or re-commit `refs/z3-json-data/`. Loader prefers the gitignored refs checkout, then the committed JSON dump.
