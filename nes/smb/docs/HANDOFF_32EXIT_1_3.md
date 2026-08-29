# Handoff — SMB 32-exit track: 1-3 unlock (1-2 flag exit)

Public TAS body (do not mix with warps #1715M): HappyLee & Mars608
warpless [3728M](https://tasvideos.org/3728M). Fetch/convert/extract:
`smb.tas.fetch_refs`, `smb.scripts.convert_fm2`, `smb.scripts.annotate_fm2`.
Verified isolated: 1-1 **1754f @ FM2 190**; 1-2 flag **2544f @ 2109 → 1-3**;
1-3 **1740f @ 4653 → 1-4** (wait=0 after flag leave).

Session 2026-08-27 (continue). **1-3 TAS slice is verified into 1-4 control.**
`smb_1_3` LevelConfig uses dash-level identity. Isolated 1-3 from
`Level1_3.state` still misses TAS phase. Warp any% line still untouched.

Living 1-4…8-4 extract recipe: [`HANDOFF_32EXIT.md`](HANDOFF_32EXIT.md).

## TL;DR (1-3 unlock — done)

1. 1-3 TAS slice: `models/smb_1_3_warpless_slice.json` (**1740f** @ FM2
   **4653 → 1-4**, warpless #3728M). Control-relative after 1-1 @190/1754
   + wait 165 + 1-2 flag @2109/2544. Same-file play/record (not #1715M):
   `uv run python -m smb.scripts.record_warpless --to 1-3`
   (`--record` for MP4). FM2 replay:
   `uv run python -m smb.scripts.annotate_fm2 --verify-1-3`.
   1-4 spawn: `dash_level=3`, x=40, y=80 (castle), timer=301, ps=7.
   1-4 castle is done (1702f @6393 → 2-1). Next extract: 2-1 —
   **use `HANDOFF_32EXIT.md`**.
2. 1-2 flag TAS slice: `models/smb_1_2_warpless_flag_slice.json` (**2544f**
   @ FM2 2109 → 1-3). Hand-built `smb.flag_12` / `smb_1_2_flag.json` (2796f)
   is the prior 2/2 body. Prefer the TAS slice for 32-exit extract.
3. Isolated `Level1_3` is a **different phase** than TAS 1-3 control
   (pin ps=8/timer=300 vs TAS ps=7/timer=301). TAS body dies there
   (odd settle max_x≈2190; even dies ~x=1044). Bunny-20 still dies in
   the first pit. rr-tb15 stays open; do not fold the TAS body onto
   the human pin. Tape re-record is rr-8qpn / rr-n6sz.

## Pin audit (`recordings/human/all_exits_v1_pins/`, written 2026-08-13 16:00–16:04)

Pins were written ~15 min **before** fb4118e9, i.e. by the old control gate
that matched `$0760` AreaNumber. Verified by loading every pin and comparing
RAM + HUD renders (evidence: `recordings/segments_all_exits/evidence/`):

| Pin | Verdict | Actually is |
|-----|---------|-------------|
| 1-1 | OK | 1-1 control (x=40, dash=0) |
| 1-2 | OK | 1-2 surface control (x=40, dash=1) |
| 1-3 | **replaced 2026-08-27** | Real 1-3 control (`dash_level=2`, x=40, ps=7/8). Old bogus pin overwritten. |
| 1-4 | **BOGUS label** | 1-3 **castle tally** (HUD WORLD 1-3, `$075C=2`) — proves the human DID flag-exit 1-2 and clear 1-3 once |
| 2-1 | OK (mid-stage) | 2-1 at x=2431, lives=1 (interstitial → live) |

**No tape** `all_exits_v1.json` was saved (old session cancelled without
F5). Extract still refuses mid-stage pins (`player_x <= 80`) and the 1-4
castle tally. The 1-3 pin is now a real control spawn.

## Emulator facts measured this session

Boot recipe (works for any pin/seed state):

```python
env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")
env.em.set_state(data); env.reset(); env.em.set_state(data)  # re-apply after free frame
```

### End-of-1-2-underground map (all coordinates = absolute player_x world px)

Replay rig: HL 1-1 chain (`smb.tas.chain.reach_surface_after_hl_1_1`) →
FM2 `happylee_warps_1715M.fm2` from index **2109** (`HL_1_2_FM2_START`).
Body: surface → pipe @f334 → UG.

**W4 warp enter is already measured** in
`recordings/segment_1_2/polish_1_2_warp_pipe_report.json`
(`polish_1_2_warp_pipe`: ceiling reverse → right platform → DOWN on W4 pipe
lip in the warp room): **(player_x=2859, player_y=128, player_state=3,
world=3)**. HL replication at x=2860 y=128 state 3 is the same pose.

Two distinct rooms:

- **Warp room** ("WELCOME TO WARP ZONE!", pipes labeled **4/3/2**): reached
  by the ceiling-top route — y=64 ceiling (grounded) to x≈2855, jump to a
  y=32 ledge (~2846–2886), fall at x≈2944–2962. Mario lands in a **pocket
  at x=2962 between pipes "3" (≈2942–2978) and "2"** — blocked both sides.
  `evidence/warp_room_from_floor.png`. Those labeled pipes go to worlds
  **4/3/2**. There is **no** 1-2 flag exit on the floor past labeled pipe
  "2". Hunting 1-3 in the warp room is a dead end.
- **UG floor corridor** (different room; `world` stays 0 until an outdoor
  flag pipe): three same-height plant pipes **A/B/C**, plant enemy slots
  (type 13) at **x=2856 / 2920 / 2984**, rim-standing head-y≈152, floor
  y=176. Floor is blocked at **2898** (B's left wall).
  `evidence/plant_pipes_ABC_geometry.png` (camera screen_x=2771). Also
  `evidence/hl_end_corridor.png`.

Pipe A (x≈2856) is a **floor plant pipe**, not the W4 warp. Mid-fall DOWN
entry works; held-DOWN while *standing* on a rim (2881,152) does **not**
enter (`evidence/rim_stand_2881_no_enter.png`) — don't burn time on
rim-standing. The 1-2 flag pipe is this floor corridor after the plant
pipes, then the outdoor flag area.

### Traps for the next probe loop

- RAM y looks like **head/top** (floor stand y=176, feet=192).
- Guard every scripted loop with a frame cap — a walk-toward-x loop against
  a pipe wall spins forever (burned a 400s timeout twice).
- Plant timing: hidden when slot `y>=158` (~half the cycle); they stay in
  while Mario is adjacent; standing on a pipe with the plant cycling = death.
- `is_in_air`/`grounded` lags a frame after landing; poll until stable.
- Emulator chain boot (HL 1-1 + surface control) takes ~60–90 s per trial —
  batch parameter sweeps inside one process.

### Flag exit (verified this session)

Replay: `smb.flag_12` / `smb.scripts.run_1_2_flag` (cache
`recordings/segments_all_exits/hl_1_2_floor_corridor.state`).

- HL FM2 leaves the **floor** at the end-of-UG **lifts** (`fm2_i≈1398`,
  `x=2520 y=148`, `$001D=0`). Policy `snap.grounded` (y-speed) is a frame
  behind — jump only when `player_on_ground`.
- Jump is **A-only** (not RBA); xs is already 40. Hold 8–20 all land on the
  brick at `y=128`. HL uses 19f A then idle; land `(2620, 128)`.
- Short green pipe against the wall is the flag exit. Standing DOWN does
  not enter; walking onto it sets `player_state=2`, then a pipe load
  (`x=y=0`) into the outdoor flag area (`area_pointer=194`, world 0).
- Outdoor: emerge on the piranha pipe, stairs, flagpole, castle. 1-3
  control after the tally: `dash_level=2`, x=40, `area_pointer=38`.
- Plant pipes A/B/C at 2856/2920/2984 are **not** this exit (mid-UG /
  earlier corridor). Do not hunt 1-3 in the warp room (labeled 4/3/2).

Evidence: `corridor_from_hl.png`, `land_hold_19.png`, `on_exit_pipe.png`,
`flag_run_f0180.png`, `flag_run_f0360.png`, `1_3_control.png`.

## Landings already in place

- `smb.flag_12` — UG floor-pipe truth table + lift/pipe tail.
- `smb.scripts.extract_stage_state 1-3` wrote `Level1_3.state` (roundtrip
  needs `env.reset()`; close the pin env before the named boot — one
  fceumm instance per process).
- `smb.flag_12` + `run_1_2_flag` — 2796f body, **2/2** to 1-3 control.
- `smb_1_3` LevelConfig uses `SMB_DASH_COMPUTED` / completion `[3]`. Isolated
  clear (`run_1_3`) is not green (TAS body is control-relative only).
- Warpless 1-3: `smb.scripts.annotate_fm2 --search-1-3 --export-1-3`
  (**1740f @4653**, 2/161 starts; alt 4589/1803). `--verify-1-3` 1/1.
- Same-file chain: `smb.scripts.record_warpless --to 1-3` plays
  `smb_1_1_warpless_slice` + `smb_1_2_warpless_flag_slice` +
  `smb_1_3_warpless_slice` from Level1_1. Verified **6205f / 1:43.247**
  to 1-4 control (settle 2 + 1754 + wait 165 + 2544 + wait 0 + 1740;
  leave `dash_level=3`, x=40, y=80, timer=301, ps=7). `#1715M` warp 1-1 /
  W4 1-2 and `smb_1_2_flag.json` are a different phase — the runner
  rejects them. Human power-on record is still `./play smb`. Evidence:
  `recordings/tas_import/warpless_3728M/warpless_1_3_play.json`.
- Warp any% line untouched: no changes to `reactive_12`, `natural_82`, HL
  seeds, `smb_1_2_reactive_fragments.json`.

## Beads

- rr-s81w / rr-tq2v done. rr-xpeq: same-file #3728M record/play through
  1-3. rr-g2ht remaining extract is 1-4…8-4 (hint 6393). rr-tb15 isolated
  1-3 still open (pin phase ≠ TAS). rr-8qpn stays open: re-record
  `all_exits_v1` pins from 1-2 with the dash-level gate when the tape
  session resumes (rr-n6sz). 1-4 human pin is still not extractable.
