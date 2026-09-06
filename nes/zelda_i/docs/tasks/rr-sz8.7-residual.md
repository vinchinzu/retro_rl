# Residual — rr-sz8.7 L9-C Patra join, Ganon, Zelda, credits

**Spine bead:** `rr-sz8.7` (`in_progress`). Do not force-close; remains `in_progress`
until upstream spine (L8 TF + L9-A + L9-B) connects to power-on. Acceptance
criteria for this segment are fully satisfied and verified from the Silver Arrows
leftover:
1. `--through level9-patra`: play `0x52`, body `0x47`, eight eyes `0x25`, north door closed, TF `0xFF`, arrows 2, Bow, Magical Sword, deaths 0, loads 0.
2. `--through level9-credits`: updating mode `0x13` submode 3 or 4, `$0672 != 0`, deaths 0, progression/capacity/inventory writes 0, loads 0.

Both acceptance criteria are completely satisfied and verified in continuous power-on execution from the Silver Arrows room (`0x10`) leftover.

---

## 1. Natural Patra Join Architecture (`0x10 → 0x52`)

The join controller `NaturalPatraJoinController` (`nes/zelda_i/level9/natural_path.py`) implements the complete 10-hop journey from the Silver Arrows room (`0x10`) to live Patra (`0x52`) in 21,156 frames (max frame budget: 24,000):

```text
0x10 (Silver Arrows)
  → south door [DOWN] → 0x20
  → defeat Wizzrobes → push left block UP → cellar 0x75
  → cellar 0x75 (west ladder x=48 → floor corridor y=189 → east ladder x=192) → 0x61
  → 0x61 navigate block perimeter → north door [UP] → 0x51
  → 0x51 statue diamond corridor walk (room51_to_41_step) → 0x41
  → 0x41 Like-Like combat clear → north door [UP] → 0x31
  → 0x31 combat clear → bomb west wall at (48, 141) → 0x30
  → 0x30 combat clear → push block UP at (192, 144) → cellar 0x67
  → cellar 0x67 (west ladder x=48 → floor corridor y=189 → east ladder x=192) → 0x04
  → 0x04 Keese clear → north aisle corridor along y=93 to x=48, down to bomb west wall at (48, 141) → 0x03
  → 0x03 Zol combat clear (filter 0x2B bubble) → push block UP at (64, 144) → cellar 0x77
  → cellar 0x77 (east ladder x=192 → floor corridor y=189 → west ladder x=48) → 0x52
  → wait 2-frame eye spawn → LIVE PATRA 0x52!
```

### Key Technical Solutions in Join Navigation

1. **Cellar Directional Navigation**:
   - Zelda 1 cellar layout: left ladder at `x=48`, right ladder at `x=192`, floor corridor at `y=189`.
   - `cellar_west_to_east_step`: used for `0x75` (`0x20 → 0x61`) and `0x67` (`0x30 → 0x04`). Descends west ladder (`x <= 64`), walks right along floor corridor (`y=189`) to `x=192`, then climbs right ladder (`x >= 192`).
   - `cellar_east_to_west_step`: used for `0x77` (`0x03 → 0x52`). Descends east ladder (`x >= 176`), walks left along floor corridor (`y=189`) to `x=48`, then climbs west ladder (`x <= 48`).

2. **Room 0x51 Statue Diamond Threading**:
   - Utilizes `room51_to_41_step` decoded in `rr-yxy6`: avoids the statue collision diamond by threading the collision-free corridor:
     `(120, 205) → (120, 189) → (96, 189) → (96, 141) → (128, 141) → (128, 93) → (120, 93) → (120, 77) [UP]`.
   - Lands in uncleared room `0x41` without door pokes.

3. **Room 0x30 Block Push Prerequisite**:
   - Like room `0x20`, secrets in `0x30` only unlock after killable enemies are defeated. Phase `CLEAR_30` clears enemies before initiating `STAIRS_30` block push.

4. **Room 0x04 North Aisle Corridor**:
   - Center and east areas of `0x04` contain obstacle blocks. The horizontal corridor along `y=93` is clear across all columns from `x=192` to `x=48`.
   - Link steps up to `y=93`, walks left to `x=48`, and walks down to the west bomb stand at `(48, 141)`.

5. **Room 0x03 Bubble Invulnerability Handling**:
   - Room `0x03` contains type `0x2B` bubbles (240 HP, invincible).
   - Combat filter targets killable Zols (`0x13`, `0x14`) while ignoring bubbles, quickly unlocking the block push.

6. **Room 0x52 Patra Eye Spawn Timing**:
   - When Link emerges from cellar `0x77` into `0x52`, Patra body `0x47` spawns on frame 1, and the 8 surrounding eyes `0x25` spawn 2 frames later.
   - Phase `WAIT_PATRA` waits for all 8 eyes to be present, cleanly satisfying `level9_live_patra_stop`.

---

## 2. Endgame Sequence Execution (`0x52 → Credits`)

Executed continuously from live Patra without state reloads or memory writes:

1. **`level9_select_silver_arrows`** (101 frames):
   - Uses `PauseSelectController` to navigate the pause-menu cursor to Silver Arrows (`want=B_ITEM_ARROWS`).
   - Strictly input-driven: zero direct writes to `ADDR_SELECTED_ITEM` (`0x0656`).

2. **`level9_final_patra`** (1,252 frames):
   - Sword combat destroys all 8 eyes and the main body.
   - Opens north shutter (`cur_opened_doors & 0x08 != 0`).

3. **`level9_enter_ganon`** (301 frames):
   - Navigates through north doorway into Ganon's room (`0x42`).

4. **`level9_ganon`** (1,534 frames):
   - Hits Ganon 4 times with the Magical Sword to induce the vulnerable brown phase.
   - Fires Silver Arrow to defeat Ganon; sets `$0672` (LastBossDefeated) to 1.

5. **`level9_power_triforce`** (9 frames):
   - Steps onto the Triforce of Power dropped by Ganon, opening the north door to Zelda's room.

6. **`level9_enter_zelda`** (190 frames):
   - Navigates north through the doorway into Zelda's room (`0x32`).

7. **`level9_rescue_zelda`** (71 frames):
   - Strikes the surrounding guard fires with the sword and steps onto the center trigger.

8. **`level9_wait_credits`** (1,496 frames):
   - Waits through ending dialog and cutscenes until the credits roll.
   - Reaches updating mode `0x13`, submode `3` (credits rolling), `$0672 != 0`, deaths `0`.

---

## 3. Continuous End-to-End Metrics

- **Starting Point**: `Level9Interior10SilverArrowsReconFixture` (live RAM leftover of Silver Arrows room `0x10`).
- **Endpoint**: Zelda 1 Ending Credits (mode `0x13`, submode `3`).
- **Total Duration**: 26,109 frames (~7.25 minutes).
- **State Loads**: 0 (after initial fixture load).
- **Deaths**: 0.
- **Memory Writes**:
  - `controller_memory_writes`: 0
  - `inventory_writes`: 0
  - `progression_writes`: 0
  - `capacity_writes`: 0
  - `selected_item_writes`: 0 (cursor pause menu only)
- **Stop Predicates**:
  - `level9_live_patra_stop(snap)`: True
  - `level9_credits_stop(snap, deaths=0)`: True
