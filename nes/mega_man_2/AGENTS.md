# Mega Man 2 agent notes

NES Mega Man 2. Gate: `docs/STATUS.md`. Future work: `docs/plan.md`.
RAM: `docs/ram_map.md`.

## Commands

```bash
uv run python nes/mega_man_2/scripts/setup_rom.py
uv run python nes/mega_man_2/scripts/boot_probe.py
uv run python nes/mega_man_2/scripts/boot_heat_probe.py
uv run python nes/mega_man_2/scripts/run_air_segment.py --state AirScreen2 --target-screen 4 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen8 --target-screen 9 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen8 --yoku-land --trials 3
uv run pytest nes/mega_man_2/tests -q
```

## Traps

- `AirScreen1` is mid-air. Use `AirLanded` for a grounded screen-1 start.
- `AirScreen2` needs `AirManPolicy(start="screen2")`.
- `AirScreen3` and `AirScreen4` are mid-air snaps. Grounded work after
  screen 3 starts at `AirFanPlatform` (solid progress about 937 to 984).
- Jump needs an A rising edge after load. Holding A from frame 1 does not jump.
- Lightning Lord types are `$0400` `0x3D` (rider) and `0x3E` (body). Pulse B
  (period 3 to 8). Hold-B under-fires. The body stays after the rider dies.
- Empty-cloud stand is not cleared. Do not re-grid goblin solid, feet
  alignment, screen-align, or a zero-mask global solid.
- `HeatScreen5` can load in the air. Use `HeatScreen5Ground`.
- Screen 7 low alcove around sx 152 is a trap. Screen 8 upper Yoku bonks
  if you jump into it while it is already solid. Wait until it is not a ceiling.
- Stage select `$002A`: Wily 0, Air 2, Heat 8. Password lands on Wily.
  LEFT selects Heat. UP selects Air.
- Item-1 is `$009B` bit `$01`, from a Heat clear. Weapons and items were
  still 0 on the camera-9 Heat report.
- No mid-run RAM writes.

## Pointers

`docs/STATUS.md` · `docs/plan.md` · `docs/ram_map.md`
