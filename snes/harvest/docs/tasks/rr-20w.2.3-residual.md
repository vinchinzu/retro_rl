## Residual — rr-20w.2.3 D2 field clearing

**Status:** IN PROGRESS. Farm→spa from `Y1_D2_Wood_Progress` is live-green.
Leftover SE stumps from `Y1_D2_Wood_SpaReturn` is **two live reds** (different
causes). Do not third that leftover command. No `--video`. No STATUS.

### Verified this session

- Live dump `Y1_D2_Wood_Progress`: farm `0x00`, `(11,25)` / `(190,400)`,
  18:02, stamina **4/100**, axe+hoe, **5 stumps all SE**
  `(34,42) (42,44) (36,51) (60,55) (42,58)`. Rocks/weeds/stones/fences 0.
- West-gate tile dump from that pin:
  - `(0,24)=0xA1` gate wall (LEFT+B pins at x=22, no map change)
  - y=25 x=0–6 `0xA6` pond
  - `(1,26)=0xA8` open face; `(0,26)=0xC0` trigger (not farm-walkable)
  - `(1,27)=(1,28)=0xFF`; house column x=8 is A0 y=24–25 then A8 y=26
- Leftover `--section stumps --chunk se` from Wood_Progress, 3 reds (halt):
  1. leftover_exec debris stall aborted spa at 24k
  2. D2 tactic motion/goal watchdog aborted spa at 24k
  3. y=24 west-run BFS push-faced `(0,24)` at `(22,384)`
- Spa-only live from Wood_Progress:
  1. `(8,392)` `force_run` LEFT: 20k still on farm at `(22,392)` tile `(1,24)`
  2. DOWN to `(136,424)` without `force_run`: 20k at `(144,392)` (house body)
  3. `force_run` DOWN house column to `(136,424)`, west on y=26 A8 → C0:
     GREEN 3989f. Maps `0x00→0x0C→0x10→0x0C→0x00`. Soak 4→100 on 0x10.
     Saved `Y1_D2_Wood_SpaReturn`.
- Glance `Y1_D2_Wood_SpaReturn`: farm `0x00`, `(1,28)` / `(24,448)`, 18:12,
  stam **100/100**, 5 stumps. Player tile `(1,28)=0xFF` so
  `farm_map_loaded` is false.
- Leftover `--section stumps --chunk se` from SpaReturn (do not third):
  1. 24k idle, never moved, journal empty, `stale_farm_map`. Fixed:
     `yard_load_action` toward `(25,28)` on that reason.
  2. Yard walk GREEN. ENSURE_AXE. Reached stump `(34,42)` from `(33,42)`,
     4 axe hits, stam 100→90, then 24k debris stall. Stumps still 5.
     End `(33,42)` / `(536,679)`. Checkpoint `Y1_D2_Leftover_Checkpoint`.
- Unit: leftover stall skips spa; D2 tactic skips spa-child watchdogs and
  walks off west-gate FF; `_FARM_WEST_EXIT` `(40,424)` `is_exit` left
  radius 6; pinch `(136,424)` `force_run` down then `(72,424)` left.
  d2_work 998 LOC. No STATUS. No natural-entry D2 claim.

### Exact next action

Do not fourth leftover from `Y1_D2_Wood_Progress`. Do not third leftover
`--section stumps --chunk se` from `Y1_D2_Wood_SpaReturn`. Diagnose the
6-hit stall at stump `(34,42)` from checkpoint
`Y1_D2_Leftover_Checkpoint` (or a headed watch of one swing). Do not
400k `--section all`. Do not start from `Y1_D2_Morning_After_D1`.
Do not STATUS.

### Non-claims

- No STATUS promotion
- No natural power-on Day 2 farm-clear
- No D2 movie / `--video`
- 5 SE stumps remain (not live-cleared)
- `--stop-after-d2-clear` is unit-wired, not live-green
