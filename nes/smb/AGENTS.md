# SMB agent notes

NES Super Mario Bros. Package `smb` (disk `nes/smb/`). Repo rules:
[root AGENTS.md](../../AGENTS.md). Gate: [docs/STATUS.md](docs/STATUS.md).
Future work: [docs/plan.md](docs/plan.md). Layout rules:
[docs/HYGIENE.md](docs/HYGIENE.md).

## Commands

```bash
uv run python smb/scripts/setup_rom.py
uv run pytest nes/smb/tests -q

# Clean power-on any% warp (the M8 gate)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.run_warp_finish --mode poweron --trials 3

# Warpless #3728M extract. Not the warp seed.
uv run python -m smb.tas.fetch_refs
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.annotate_fm2 --search 2-2 --from-pred --export
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.record_warpless --to 2-1
./play smb --list
```

Parked polish and import CLIs are in [docs/plan.md](docs/plan.md).
Adapt rules: [docs/TAS_ADAPT.md](docs/TAS_ADAPT.md). RAM: [docs/ram_map.md](docs/ram_map.md).

`policy.py` replays warp seeds. `tas/stages.py` holds `StageSpec` rows.
`flag_12.py` is the 32-exit 1-2 flag body, not the World 4 warp.
`scripts/` stays thin. Soft max about 1000 lines. Merge into `StageSpec`
or `policy`, or delete ([CODING_STANDARDS.md](../../CODING_STANDARDS.md)).

## Traps

- Power-on warp: boot 350 plus settle 16. Level1_1 continuous: settle 14.
  Natural 1-1 alone: settle 1.
- World 4 is world index 3. The 32-exit clock is `$075C` LevelNumber, not
  `$0760` AreaNumber.
- Ending is world 8-4 and `oper_mode=2`, held 120 idle frames.
- Do not absolute-stitch a faster 1-1 into an old 1-2 body. Retime from
  control. Preserve Left+Right. `sanitize_action` clears that combo.
- Warpless movie is TASVideos #3728M. Warp any% is #1715M. Do not mix them.
- Pin boot order is `set_state`, then `reset()`, then `set_state`.
- RAM Y is the head. A floor stand is y=176.
- Do not promote a hybrid, a pure FCEUX replay, or a 32-exit tape to the
  Clean gate. The Clean seed is `models/smb_1_1_to_ending_natural_82.json`.
