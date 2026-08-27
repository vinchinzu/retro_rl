# Handoff — SMB 32-exit track: 1-3 unlock (1-2 flag exit)

Session 2026-08-27. Goal was 1-3 segment work; outcome: **the 1-2 normal
(flag) exit is the real gate**, and the existing `all_exits_v1` pins are
partly mislabeled. Everything below was verified live on fceumm this session.

## TL;DR for next session

1. Find the 1-2 UG **normal exit** (outdoor flag area, `world` stays 0) by
   entering the unlabeled plant pipes B/C with the **mid-fall DOWN entry**
   (technique proven below). Skip lore — pipe truth table is what's missing.
2. Once found: standalone 1-2 flag body = HL slice prefix + scripted tail;
   run it; at 1-3 control capture a state (`dash_level==2`, `$075C==2`).
3. `uv run python -m smb.scripts.extract_stage_state 1-3` → installs
   `Level1_3.state`; `smb_1_3` LevelConfig is **already registered**
   (`nes/smb/platformer_levels.py`). Segment work starts there.

## Pin audit (`recordings/human/all_exits_v1_pins/`, written 2026-08-13 16:00–16:04)

Pins were written ~15 min **before** fb4118e9, i.e. by the old control gate
that matched `$0760` AreaNumber. Verified by loading every pin and comparing
RAM + HUD renders (evidence: `recordings/segments_all_exits/evidence/`):

| Pin | Verdict | Actually is |
|-----|---------|-------------|
| 1-1 | OK | 1-1 control (x=40, dash=0) |
| 1-2 | OK | 1-2 surface control (x=40, dash=1) |
| 1-3 | **BOGUS** | 1-2 UG pipe-entry frame (x=160 y=176 state=7 area_ptr=192); settles into 1-2 underground (HUD says WORLD 1-2, `$075C=1`) |
| 1-4 | **BOGUS label** | 1-3 **castle tally** (HUD WORLD 1-3, `$075C=2`) — proves the human DID flag-exit 1-2 and clear 1-3 once |
| 2-1 | OK (mid-stage) | 2-1 at x=2431, lives=1 (interstitial → live) |

All pins byte-match their `.json` metas. **No tape** `all_exits_v1.json` was
saved (session cancelled without F5). `smb.scripts.extract_stage_state`
correctly refuses the bogus 1-3 pin (dash-level check) — that refusal is the
audit passing, not a bug.

## Emulator facts measured this session

Boot recipe (works for any pin/seed state):

```python
env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")
env.em.set_state(data); env.reset(); env.em.set_state(data)  # re-apply after free frame
```

### End-of-1-2-underground map (all coordinates = absolute player_x world px)

Replay rig: HL 1-1 chain (`smb.tas.chain.reach_surface_after_hl_1_1`) →
FM2 `happylee_warps_1715M.fm2` from index **2109** (`HL_1_2_FM2_START`).
Body: surface → pipe @f334 → UG; **HL enters a pipe at x=2860, feet y=128,
state 3 @f1656 → `world=3` = W4-1** (re-verified live).

- Ceiling-top route: y=64 ceiling (grounded) runs to x≈2855, jumps to a y=32
  ledge (~2846–2886), then falls into the **warp room** at x≈2944–2962.
- Floor corridor: three same-height plant pipes **A/B/C**, plant enemy slots
  (type 13) at **x=2856 / 2920 / 2984**, rim-standing head-y≈152, floor y=176.
  Floor is blocked at **2898** (B's left wall). Camera-relative geometry
  capture: `evidence/plant_pipes_ABC_geometry.png` (camera screen_x=2771).
- **Pipe A (x≈2856) = HL's W4 warp.** Mid-fall entry replicated: from HL
  f1640 (2903,84, xs=−40) just **hold DOWN** → state 3 at (2860,128) →
  world 3. Held-DOWN while *standing* on a rim (2881,152) does **not**
  enter — entry seems to need the fall/tile alignment; don't burn time on
  rim-standing attempts, use the mid-fall technique.
- **Warp room** (separate sub-area, "WELCOME TO WARP ZONE!", pipes labeled
  4/3/2): reached by falling past the y32 ledge at x≈2944+; Mario lands in a
  **pocket at x=2962 between pipes "3" (≈2942–2978) and "2"** — blocked both
  sides. `evidence/warp_room_from_floor.png`.

### Traps for the next probe loop

- RAM y looks like **head/top** (floor stand y=176, feet=192).
- Guard every scripted loop with a frame cap — a walk-toward-x loop against
  a pipe wall spins forever (burned a 400s timeout twice).
- Plant timing: hidden when slot `y>=158` (~half the cycle); they stay in
  while Mario is adjacent; standing on a pipe with the plant cycling = death.
- `is_in_air`/`grounded` lags a frame after landing; poll until stable.
- Emulator chain boot (HL 1-1 + surface control) takes ~60–90 s per trial —
  batch parameter sweeps inside one process.

### Open questions (the actual next probes)

1. **Pipes B (2920) / C (2984):** mid-fall DOWN entry with plant hidden →
   does either give `world=0` (outdoor 1-2 flag area)? That's the normal
   exit candidate. Entry approach: from the y64 ceiling walk off at the pipe's
   x while holding DOWN, or drop from the y32 ledge with slight LEFT drift.
2. If B/C aren't it: from the room pocket (2962), walk-jump right over pipe
   "2" (runway ~2978–3030 is blocked; go over "3" leftward first for a long
   runway, or jump from the pocket at ≈2990 with walk-speed) and check the
   floor beyond the labeled pipes for a final exit pipe.
3. On success: save the 1-3 control state, then `extract_stage_state 1-3`.

## Landings already in place

- `smb.scripts.extract_stage_state` — pin→practice-state promotion with
  fingerprint verification + round-trip boot check (`--list` works, 1-3
  correctly rejected; report at `recordings/segments_all_exits/`).
- `smb_1_3` LevelConfig registered (`platformer_levels.py`) — state file
  pending step 3 above.
- Warp any% line untouched: no changes to `reactive_12`, `natural_82`, HL
  seeds, `smb_1_2_reactive_fragments.json`.

## Beads

- Pin-mislabel audit result: folded into the handoff issue for this track
  (create/retitle as needed); re-record pins with the fixed gate whenever the
  tape session resumes — old 1-3/1-4 pins are unusable as control entries.
