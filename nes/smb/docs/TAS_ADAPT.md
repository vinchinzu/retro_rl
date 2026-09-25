# TAS adaptation

Prefer importing a public movie over hill-climbing holds. The Clean gate
stays in `docs/STATUS.md` until a full adapted seed has its own Clean
power-on 3/3. Do not promote a slice, a hybrid, or a native-emulator
replay onto that gate.

## Why raw frames mislead

Levels 1-1 through 8-3 wait on a 21-frame boundary. Saving 1 to 20 frames
mid-level often does not change total time. Saving a whole rule saves 21
frames. World 8-4 ends on the axe, so raw frames matter there.

HappyLee #1715 is about 17,868 power-on frames (04:57.31). The Clean seed
is 21,559 RTA frames. That gap is route structure (flagpole glitch, bullet
bill glitch, fast accel), not one missed hold.

## Sources

| Source | Role |
|--------|------|
| `tas/ref/happylee_warps_1715M.fm2` | Warp any%. [tasvideos.org/1715M](https://tasvideos.org/1715M). Preserve Left+Right. |
| `tas/ref/happylee_mars608_warpless_3728M.fm2` | 32-exit, no warp. [tasvideos.org/3728M](https://tasvideos.org/3728M). Do not fold into warp slices. |
| `tas/ref/flamexx_warps_rta_4_54_099.fm2` | Later community 8-2 / 8-4 ideas. Different pad. Compare, do not splice blind. |

Trick list: [TASVideos SMB resources](https://tasvideos.org/GameResources/NES/SuperMarioBros).

## Pipeline

1. Import FM2 to nes9 (`smb.tas.fm2`, `smb.tas.slice`).
2. Verify under stable-retro. Do not sanitize Left+Right.
3. On desync, phase-align boot or split at control and retime.
4. Replace a `natural_82` body only when the TAS segment wins and the
   successor still matches.
5. Residual polish only after that, and only on 8-4 or a real desync.

```bash
uv run python -m smb.tas.fetch_refs
uv run python -m smb.scripts.convert_fm2
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.record_warpless --to 2-1
uv run python -m smb.scripts.pure_hl status
```

32-exit open step is in `docs/plan.md`. It is not this warp pipeline.

## Hard rules

1. Preserve Left+Right. TAS uses it to accel and brake.
2. HappyLee's first real boot input is Start around frame 41, not the
   repo's 350-frame Start spam. Compare both before blaming the stage.
3. Full power-on FM2 does not sync on fceumm. The blackout is longer than
   FCEUX. Do not pad the whole movie to hide that.
4. Do not absolute-stitch a mid-movie body into `natural_82`.
5. Natural 1-1 HappyLee clears on odd settle (default 1). Even settle dies.
6. Keep old seeds. `natural_82` remains the Clean gate until a new Clean
   power-on report exists.
7. Three tracks do not share writes. Clean uses `natural_82`. Hybrid and
   stitchless use their own seeds. Pure HappyLee writes only
   `models/pure_hl/` and `recordings/tas_import/pure_hl/`.

## FM2 to nes9

| FM2 | Button | NES index |
|-----|--------|-----------|
| R | Right | 7 |
| L | Left | 6 |
| D | Down | 5 |
| U | Up | 4 |
| T | Start | 3 |
| S | Select | 2 |
| B | B | 0 |
| A | A | 8 |

Index 1 is the stable-retro hole and stays 0.

## Documented slice board

These figures are the previous writeup (2026-08-07). This cleanup did not
re-run them. They are not the Clean gate.

- Isolated 1-1 HappyLee slice: 1733 frames (`smb_1_1_happylee_slice.json`).
  Natural entry with settle 1: 1749 frames. Even settle dies.
- Control-relative 1-2 to World 4: 1657-frame body, chain about 3555.
- 4-1 body 2062, 4-2 body 1516, chain about 7512 to World 8. The 4-2 slice
  is a glitch warp, not the natural vine. Check video and RAM before
  treating it as the route.
- 8-1 leave 2881, 8-2 leave 2209. Stitchless 8-3 leave 2374 (2/2) is a
  skill resume, not pure FM2. Pure 8-3 on fceumm was still open: same
  position, FCEUX vy -3 versus fceumm vy -5.
- Hybrid v2 (flamexx 8-4 after a natural 8-3) was documented at 18,031
  frames from Level1_1. Showcase only. Not Clean power-on.
- Pure track 8-4 stays blocked until a pure 8-3 leave gate exists.

## Pure track commands

```bash
uv run python -m smb.scripts.pure_hl status
uv run python -m smb.scripts.pure_hl verify-to-83
uv run python -m smb.scripts.pure_hl check-8-4-gate
```

Writes stay inside the pure_hl dirs. No `natural_82`, flamexx, or skill
macros on that track.
