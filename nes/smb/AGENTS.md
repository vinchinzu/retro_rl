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
uv run python -m smb.scripts.run_reactive_warp --retime-4-1 --retime-4-2
uv run python -m smb.scripts.run_reactive_warp --retime-4-1 --retime-4-2 --retime-8-2
uv run python -m smb.scripts.fold_continuous_policy
uv run python -m smb.scripts.rle_polish --list-windows

# 1-1 TAS polish (analyze / multi-window hillclimb / systematic delete)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.tas_1_1 analyze
uv run python -m smb.scripts.tas_1_1 optimize --window stairs,first-pipe --iters 300
uv run python -m smb.scripts.tas_1_1 verify --seed nes/smb/models/smb_1_1_tas_best.json

# 1-2 underground polish (control-relative; keeps reactive gates)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.polish_1_2_ug --windows lead,mid,slam,body
# 1-2 W4 pipe top-land suffix (no floor/face-slam)
uv run python -m smb.scripts.polish_1_2_warp_pipe
uv run python -m smb.scripts.run_1_2 --predecessor stairs --trials 2

# Human record from natural reactive handoff (no W4 pad) → skill chunks
uv run python -m smb.scripts.record_human --list
uv run python -m smb.scripts.record_human --from 4-1 --name late_v1
# bot drives retimed W4+; press ~ to take over anytime
uv run python -m smb.scripts.record_human --from auto --name pickup_v1
uv run python -m smb.scripts.parse_human_recording \
  nes/smb/recordings/human/late_v1.json --export-skills --list-jumps
```

## Traps

- Power-on: **exactly** 350 boot frames + **16** idle, then seed.
- Level1_1 continuous: **exactly 14** idle after `Level1_1` (different phase).
- Natural 1-1 alone: idle **1** after readiness (`NATURAL_SETTLE_FRAMES`).
- World 4 = `world` index **3**. Underground `level_id=2` ≠ completion.
- Ending = World 8-4 + `oper_mode=2`, held 120 idle frames.
- **Do not** absolute-frame stitch a faster 1-1 into old 1-2 — use
  `smb.reactive_12` control gates. **Do not** W4 idle-pad to restore phase;
  retime later legs from natural control (`--retime-4-1` → cont index 218;
  `--retime-4-2` freezes source at 4-1 score/load and resumes at cont 2487;
  `--retime-8-2` → +1 lead at cont 8917, then late 8-3 +2 lead + patches).
  Goal is **trim time**, never pad macros to fit an old phase. Old drop-5
  @12,898 is stale after 1-2 −97f.
- 1-1-stairs polish window = frames **1050–1311** (wall-slam), not castle idle.
- 1-2 polish mutates only `underground_from_control` in
  `smb_1_2_reactive_fragments.json` (surface stays reactive RIGHT/DOWN).

## Layout (pointers)

`ram.py` / `obs.py` / `policy.py` · `reactive_12|late|route.py` ·
`scripts/run_warp_finish.py` · `rle_windows.py` · `tas/` (1-1 TAS toolkit) ·
`scripts/tas_1_1.py` · `scripts/polish_1_2_ug.py` ·
`scripts/polish_1_2_warp_pipe.py` · `scripts/record_human.py` ·
`scripts/parse_human_recording.py` · `retro_harness.platformer.rle_*` + `neuro/`.

## Next

Best Clean power-on: **21,559f** 3/3
(`smb_1_1_to_ending_natural_82.json`). Next: fold promote + re-record MP4.
4-2 still 2599f vine (sub-5 needs route structure). Evidence: `docs/STATUS.md`.
