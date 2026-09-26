# Zelda I WIP handoff

Updated 2026-09-26.

## Verified and landed

The main Clean spine now clears Level 8 continuously from power-on.

- Run: `recordings/clean_poweron98.json`
- Command: `QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scripts/run_survival_spine.py --clean --through level8 --save-points C10 --no-video --trials 1 --tag clean_poweron98`
- Result: `ok=True`, 296,423 frames, `resumed_from=null`, 0 state loads, 0 deaths, 0 inventory/progression writes, Triforce `0xFF`, 15 containers, Magic Key held, settled at overworld `0x6D`.

The route changes are in the active spine path:

- Level 7 room `0x38` collects its natural 5-rupee item before the north door.
- Level 7 room `0x29` waits for the cellar wave before crossing to the east bomb wall.
- The Level 8 post-L7 route uses the natural `0x13` 30-rupee cave, Recorder travel through the Level 4 door, and the `0x64` blue potion shop. The potion returns two full refills for the Level 8 return passage.
- Recorder warps leave intermediate completed-dungeon door screens before cycling again. This is enabled only by the new Level 8 detour factories; the established Level 7 warp behavior remains unchanged.
- Level 8 Gohma uses lead-based arrow alignment across the inland lane.
- Level 8 room `0x3F` aligns to the measured y=141 CheckWarp row before idling on the stairs.
- Old unit tests for the retired 0x3E combat and Gohma dodge policies were removed or replaced with tests for the active north-bomb and lead-aim policies.

Verification after the changes:

`QT_QPA_PLATFORM=offscreen uv run pytest nes/zelda_i/tests -q` → 2052 passed, 40 deselected.

## Open handoff: Clean Level 9 / credits

Bead `rr-npv.5` remains open. A resumed probe from the verified C10 ending reached the Level 9 route but failed before the Silver Arrows chapter:

- Run: `recordings/clean_c11_bombs100.json`
- This was a resume from the C10 Level 8 ending, not a power-on credits claim.
- The route reached `level9_spectacle_rock_bomb`, then entered `level9_natural_silver_arrows` with about 0.7 hearts and died.
- With the blue potion detour, the wallet is 2R at the first Level 9 bomb shop. The current Level 9 plan still requests two 20R packs (`LEVEL9_BOMBS_WANTED=8`); do not lower that constant without a measured replacement budget.
- The main observed damage is on the long post-L8 overworld leg, especially screens `0x59` and `0x5A`. Start by replaying `C10_level8_ow_leave_settle` into `level9_post_l8_to_rock` with damage logging and several RNG offsets.

Next action: make the Level 9 approach survive naturally and fund its required bomb packs, then run a new continuous `--clean --through level9-credits` power-on tape. Do not promote a resumed run to the STATUS claim.
