# Agent Instructions — smb

NES Super Mario Bros. (**M8** Clean power-on → 8-4 ending). Shared:
`retro_harness.platformer` (RLE / neuro). Docs: `docs/STATUS.md`, `docs/plan.md`.

## Commands

```bash
uv run python smb/scripts/setup_rom.py
uv run python smb/scripts/boot_probe.py
uv run python -m pytest smb/tests -q

# Clean power-on → ending (3/3 baseline)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.run_warp_finish --mode poweron --trials 3
uv run python -m smb.scripts.run_warp_finish --mode poweron --record

# Continuous / segments
uv run python -m smb.scripts.run_warp_finish --mode continuous --trials 3
uv run python smb/scripts/run_1_1.py --natural-entry --trials 3
uv run python -m smb.scripts.run_1_2 --predecessor stairs --trials 3
uv run python -m smb.scripts.run_reactive_warp --retime-8-2
uv run python -m smb.scripts.fold_continuous_policy
uv run python -m smb.scripts.rle_polish --list-windows
```

## Traps

- Power-on: **exactly** 350 boot frames + **16** idle, then seed.
- Level1_1 continuous: **exactly 14** idle after `Level1_1` (different phase).
- Natural 1-1 alone: idle **1** after readiness (`NATURAL_SETTLE_FRAMES`).
- World 4 = `world` index **3**. Underground `level_id=2` ≠ completion.
- Ending = World 8-4 + `oper_mode=2`, held 120 idle frames.
- **Do not** absolute-frame stitch a faster 1-1 into old 1-2 — use
  `smb.reactive_12` control gates. **Do not** W4 idle-pad to restore phase;
  retime later legs from natural control.
- 1-1-stairs polish window = frames **1050–1311** (wall-slam), not castle idle.

## Layout (pointers)

`ram.py` / `obs.py` / `policy.py` · `reactive_12|late|route.py` ·
`scripts/run_warp_finish.py` · `rle_windows.py` ·
`retro_harness.platformer.rle_*` + `neuro/`.

## Next

Promote verified 21,643f reactive policy as default fold + capture; then 4-2
polish / optional all-32. Evidence: `docs/STATUS.md`.
