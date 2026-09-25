# SMB3 agent notes

NES Super Mario Bros. 3. Gate: `docs/STATUS.md`. Future work: `docs/plan.md`.
RAM: `docs/ram_map.md`.

## Commands

```bash
uv run python nes/smb3/scripts/setup_rom.py
uv run python nes/smb3/scripts/boot_probe.py
uv run python nes/smb3/scripts/run_level1.py
uv run python nes/smb3/scripts/run_level1.py --from-state Level1_1
uv run python nes/smb3/scripts/run_level1.py --level 1-2
uv run python nes/smb3/scripts/run_level1.py --level 1-2 --from-state Level1_2
uv run pytest nes/smb3/tests -q
```

The 1-2 command exists. `docs/STATUS.md` still records only the World 1-1
clear. Do not treat a 1-2 run as verified unless that gate moves.

## Traps

- The boot map pose is not the enterable 1-1 node. Walk RIGHT, then UP, then A.
- `AfterLevel1` is not controllable immediately. Wait for Map_Operation `$0D`.
- 1-1 to 1-2 is two RIGHT hops: a T-junction tile, then the 1-2 panel `$04`.
- 1-1 and 1-2 policies are frame-synced to natural entry. A map-walk or
  level-load settle change desyncs them.
- NES actions use `retro_harness.nes` (9-button fceumm). B runs, A jumps.
- Progress is `x_page` (`$75`) times 256 plus `hpos` (`$90`), and only while
  `x_page < $18`.

## Pointers

`docs/STATUS.md` · `docs/plan.md` · `docs/ram_map.md`
