# Agent Instructions — smb

NES Super Mario Bros. (**M8** Clean power-on → 8-4 ending). Shared:
`retro_harness.platformer` (RLE / neuro). Docs: `docs/STATUS.md`, `docs/plan.md`.

## Commands

```bash
uv run python smb/scripts/setup_rom.py
uv run python smb/scripts/boot_probe.py
uv run python -m pytest smb/tests -q

# 32-exit human tape (Super Metroid-style ./play)
./play smb                         # power-on → all_exits_v1
./play smb 4-1 all_exits_v1        # continue from stage pin
./play smb --list                  # F5=save  F6=pin  ESC=cancel

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
# Prefer HappyLee FM2 slice when available (~1733f isolated)
uv run python -m smb.scripts.tas_1_1 verify \
  --seed nes/smb/models/smb_1_1_happylee_slice.json
# Import / verify public FM2 (do not sanitize L+R)
uv run python -m smb.scripts.import_fm2 --summary-only
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.import_fm2 --verify
# Pure HappyLee track 3 (no hybrid/natural/skills; 8-4 blocked until 8-3 leave)
uv run python -m smb.scripts.pure_hl status
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.pure_hl verify-to-83
# uv run python -m smb.scripts.pure_hl search-83 --with-continuous
# Control-relative HappyLee 1-2 → W4 (after HL 1-1 natural predecessor)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.import_fm2 --verify-1-2-slice
# Control-relative HappyLee 4-1 + 4-2 → W8 (after HL W4)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.import_fm2 --verify-4-1-4-2-slice
# Record HL / hybrid MP4 (HUD+audio; Level1_1; not raw power-on)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.record_happylee --to ending   # hybrid v2 axe ~5:00.02
# uv run python -m smb.scripts.record_happylee --to w8
# uv run python -m smb.scripts.import_fm2 --verify-8-1-8-2-slice

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
- World 4 = `world` index **3**. Underground `level_id=2` ≠ completion
  (`$0760` AreaNumber; 32-exit clock uses `$075C` LevelNumber so 1-2 UG
  is not 1-3).
- Ending = World 8-4 + `oper_mode=2`, held 120 idle frames (success gate).
  Recordings hold **780f** post-ending through Peach + thank-you text
  (`--peach-hold-frames`; do not cut on Bowser-drop alone).
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

`./play smb` (`scripts/play.py`, 32-exit tape) · `ram.py` / `obs.py` /
`policy.py` · `reactive_12|late|route.py` · `scripts/run_warp_finish.py` ·
`rle_windows.py` ·
`tas/` (adapt: `stages` StageSpec table, `slice` probe/export, `chain` reach/verify,
`replay` to_action9/idle_until, `fm2` import; residual `pipeline` 1-1 hill-climb) ·
`scripts/import_fm2.py` · `scripts/tas_1_1.py` · `scripts/polish_1_2_ug.py` ·
`scripts/polish_1_2_warp_pipe.py` · `scripts/record_human.py` ·
`scripts/parse_human_recording.py` · `retro_harness.platformer.rle_*` + `neuro/`.

## Next

Best Clean power-on: **21,559f** 3/3
(`smb_1_1_to_ending_natural_82.json`). Published continuous MP4:
`recordings/fullgame_replays/smb_warp_any_percent_poweron.mp4`.

**Prefer TAS adapt over hill-climb** (`docs/TAS_ADAPT.md`): HappyLee through
8-2 verified + exported; **hybrid v2 ending** (HL→8-2 + natural 8-3 bridge +
**flamexx 8-4@15210**) **18,031f / 5:00.02** Level1_1→axe (**−3,528** vs n82;
**+1** vs 18,030 sub-5 budget). Seeds: `smb_*_happylee_slice.json` (1-1…8-2) +
`smb_8_3_natural_for_hl_hybrid.json` + `smb_8_4_flamexx_slice.json` +
`smb_happylee_hybrid_v2_fx84.json` (v1 `smb_happylee_hybrid_ending.json` =
18,769f natural 8-4 kept). Record: `record_happylee --to ending`.
**8-3 stitchless leave 2374f 2/2** (`smb_8_3_stitchless_skills_leave.json`;
no natural_82 splice). Full HL…8-2+skills8-3+FX8-4 chain / Clean power-on
still open. Pure continuous FM2 8-3 still phase-blocked. Full power-on FM2
desyncs on fceumm.

### Oracle early-8-3 probe (rr-34v / FCEUX 2.6.6)

Entry **exact**; first `y`/`ys` break @ body **101** (not 8-2→8-3). Gate:
first obstacle → x900 → x1600 → flag/8-4 (max_x alone ≠ success).

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.tas.oracle.probe_early_8_3
# bounded local-search v3 (exact first-obstacle pose first; distinct artifacts):
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.tas.oracle.probe_early_8_3 --search-v3 --export
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run pytest nes/smb/tests/test_oracle.py -q
```

Evidence: `recordings/tas_import/oracle_happylee_8_3/` (distinct candidates;
do not overwrite shared seeds / natural_82). v3 →
`early83_local_search_v3_evidence.json` + `…_candidate_v3.json`.
