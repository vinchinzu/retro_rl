> Historical lane note. The live sitting is the gathering prefix in [PRE_L1.md](../PRE_L1.md). This file stays because the clean-tip ladder or a route doc still cites it. It is not the current plan.

# rr-bxzj residual — Clean L4 Entrance→TF heart-safe Gleeok

CLEAN RUN COMPLETE: `ok=True`, `failed=None`, `tf08=True`, `deaths=0`.
`route_eligible=false`. Do not STATUS. Do not close the bead without user request.
All 31 contiguous stages from `Level4Entrance` completed cleanly with zero RAM pokes,
zero state restores, and `--no-infinite-life`.

Isolated runner: `run_level4_entrance_tf.py --from-state Level4Entrance --no-infinite-life --no-video --trials 1`. Pin `Level4Entrance` (play 0x71).

## Landed

1. **Heart Preservation Across Dungeon Crawl:**
   - `level4_key_0x01`: corrected `KEY_01_PICKUP_XY` from `(120, 141)` to actual floor key `(120, 128)`, cutting pickup time from 638f to 121f and eliminating 2 hits.
   - `level4_stepladder`: added periodic defensive slashes `(self.frames % 6) < 3` along dock and spinning slashes on pedestal in Room 0x60, eliminating 1 hit.
   - `level4_exit_0x60`: added spinning sword slashes during `item_freeze` to prevent Keese hits on pedestal.
   - `level4_map_0x21`: increased sword slashing frequency against diving flyers to `(self.combat_frames % 4) < 2`.
   - `level4_key_up_0x20`: added defensive slashes while walking UP to key door 0x20 in Room 0x30, eliminating 1 hit.
   - `level4_gleeok_enter_0x13`: tightened push tolerance to `abs(dy) <= 1` (`143 <= y <= 145`) and verified `snap.cur_opened_doors & 0x01` before transitioning to `token_path`. Room 0x12 push opens door 100% reliably in 55 frames into Room 0x13 (`xy=[32, 141]`, `phase=DONE`).
   - Link arrives in Room 0x13 with **health = 102** (`hearts_hi=6, hearts_lo=6`), taking only 9 hits across 38,000 frames.

2. **Heart-Safe Gleeok Combat Policy (`boss_combat.py` & `dungeon/gleeok.py`):**
   - Configured `fireball_dodge_dist = 10` during mid-fight south-stand. At distance 14, Link previously chased horizontal fireballs across the room away from Gleeok into corners and died. At distance 10, Link remains anchored under Gleeok (`x=124, y=133`) continuously dealing sword damage while cleanly evading direct-threat fireballs.
   - Support `stand_dx: int = 0` and `stand_dy = 22` with optional `stand_dx` in `_south_stand_action`.
   - Gleeok killed at frame 950, Link exits UP through Room 0x13 north door into Room 0x03, and picks up Triforce piece at `(120, 141)`.

## Clean Run Metrics (Trial 0)

- **Result:** `ok=True`, `failed=None`, `tf08=True`, `deaths=0`
- **Total Frames:** 40655
- **Final RAM State:**
  - `room`: **0x03**
  - `mode`: **18** (fanfare)
  - `xy`: **(120, 149)**
  - `triforce`: **12** (`0x0C`, Level 3 bit 0x04 + Level 4 bit 0x08)
  - `health`: **102** (`hearts_hi=6, hearts_lo=6`)
  - `keys`: **1**
  - `bombs`: **5**
  - `state_restores`: **0**
  - `infinite_life`: **false**
- **Recording Artifact:** `recordings/l4_entrance_tf.json`
- **PNG Screenshot:** `recordings/l4_entrance_tf_t0_final.png`


