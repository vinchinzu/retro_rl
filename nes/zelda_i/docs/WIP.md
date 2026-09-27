# Zelda I WIP handoff

Updated 2026-09-27.

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

## 2026-09-27: Clean post-L8 offset measurement (resumed, not a gate)

The eight initial pins `C11OWo0..7_level8_ow_leave_settle` came from
`C10_level8_ow_leave_settle` with `scratch/offset_pins.py`. Their raw runs are
`recordings/c11ow_before_o0.json` through `c11ow_before_o7.json`. All eight
were the same tape except for one fewer settle frame per offset: idle offsets
before the L8 fanfare settled did not vary the overworld wave. They remain
preserved but are **not** the scored comparison. A save point at the start of
`level9_post_l8_overworld` from offset 0 was pinned again as
`C11Evalo0..7_level9_post_l8_overworld`. Both scored sets below resume those
same eight post-settle pins with `--clean --through level9-entry --no-video
--trials 1`. `level9-entry` stops immediately before
`level9_natural_silver_arrows`; its final `level9_spectacle_rock_bomb` hearts
out are the Silver Arrows stage-entry hearts. `—` means that stage was never
reached. Each raw JSON holds every executed stage's hearts in/out and
`damage_by_room`. Every scored run has exactly one disclosed resume load,
zero health refills, and zero inventory/progression writes between frames.

The fix routes a knockback from the south pocket of `0x5A` back to its west
passage with `ow_edge_band_step`. Offset 7 previously pressed LEFT at
`(112,205)` until timeout; it now completes the first leg. On `0x5D`, the
approach banks a nearby natural four-bomb drop when below the unchanged
`LEVEL9_BOMBS_WANTED=8`. Offsets 5–7 take that drop, so both 0x4A restock
stages correctly skip with eight bombs despite the 2R wallet. The other five
arrive with four bombs and fail at `bomb_restock_l8` because a pack costs 20R.

In the table, room damage is in hearts and all rooms are overworld `0:xx`.
Damage includes every post-L8 stage that actually ran; later rooms are absent
from runs stopped at the shop. The JSON filenames are relative to
`nes/zelda_i/recordings/`.

| Offset | Before JSON | Silver entry before | Before `damage_by_room` | After JSON | Silver entry after | After `damage_by_room` |
|---:|---|---|---|---|---|---|
| 0 | `c11ow_before_o0_settled.json` | No; — | 5D .50, 5A .25, 59 .25 | `c11ow_after_o0_final.json` | No; — | 5D .50, 5A .25, 59 .25 |
| 1 | `c11ow_before_o1_settled.json` | No; — | 5D .25 | `c11ow_after_o1_final.json` | No; — | 5D .25 |
| 2 | `c11ow_before_o2_settled.json` | No; — | 5D .25, 49 .25 | `c11ow_after_o2_final.json` | No; — | 5D .25, 49 .25 |
| 3 | `c11ow_before_o3_settled.json` | No; — | 5D .75 | `c11ow_after_o3_final.json` | No; — | 5D .75 |
| 4 | `c11ow_before_o4_settled.json` | No; — | 5D .25 | `c11ow_after_o4_final.json` | No; — | 5D .25 |
| 5 | `c11ow_before_o5_settled.json` | No; — | 5D .50, 5A .25 | `c11ow_after_o5_final.json` | Yes; 8.48 | 5D .50, 59 .25, 58 .50, 28 .75, 17 .50, 07 1.00, 06 .50, 05 3.01 |
| 6 | `c11ow_before_o6_settled.json` | No; — | 5D .25 | `c11ow_after_o6_final.json` | Yes; 8.24 | 5D .25, 59 .25, 49 .25, 58 .25, 07 1.00, 06 1.00, 05 4.00 |
| 7 | `c11ow_before_o7_settled.json` | No; — | 5D .25, 5A 1.00 | `c11ow_after_o7_final.json` | Yes; 6.98 | 5D .25, 59 1.00, 58 .25, 28 .50, 17 .25, 06 2.00, 05 4.01 |
| **Total** | 8 runs | **0/8; mean —** | 5D 3.00, 5A 1.50, 59 .25, 49 .25; **5.00** | 8 runs | **3/8; mean 7.90** | 5D 3.00, 5A .25, 59 1.75, 58 1.00, 49 .50, 28 1.25, 17 .75, 07 2.00, 06 3.50, 05 11.02; **25.02** |

All starts held 15.00 hearts. Stage hearts below are `in→out`; these are
rounded to two decimals. Stage keys: A = `level9_post_l8_overworld`, B =
`bomb_restock_l8`, C = `exit_bomb_restock_l8`, D =
`bomb_restock_l8_second`, E = `exit_bomb_restock_l8_second`, F =
`level9_post_l8_to_rock`, G = `level9_white_sword`, H =
`level9_spectacle_rock_bomb`. B, C, D, E and G took zero damage in every
run; before offset 7 stopped in A. All per-stage `damage_by_room` maps are in
the named JSONs; A for offsets 0–4 is the table's entire room map.

| Offset | Before stage hearts | After stage hearts |
|---:|---|---|
| 0 | A 15→14.50; B 14.50→14.50 | Same |
| 1 | A 15→14.75; B 14.75→14.75 | Same |
| 2 | A 15→14.75; B 14.75→14.75 | Same |
| 3 | A 15→14.25; B 14.25→14.25 | Same |
| 4 | A 15→14.75; B 14.75→14.75 | Same |
| 5 | A 15→14.75; B 14.75→14.75 | A 15→15; B–E 15→15; F 15→11.49; G 11.49→11.49; H 11.49→8.48 |
| 6 | A 15→14.75; B 14.75→14.75 | A 15→14.75; B–E 14.75→14.75; F 14.75→12.25; G 12.25→12.25; H 12.25→8.24 |
| 7 | A 15→14.00 (timeout) | A 15→14.75; B–E 14.75→14.75; F 14.75→10.99; G 10.99→10.99; H 10.99→6.98 |

For the three runs that reached H, A's damage was only on 5D (.50/.25/.25
for offsets 5/6/7), H's only on 05 (3.01/4.00/4.01), and all their other
room damage in the first table belongs to F. The first leg's completion
improved from 7/8 to 8/8; its `0x5A` damage fell from 1.50 to .25 total.
The full-leg damage totals are not like-for-like because only the after runs
reached F and H.

Silver Arrows **entry** is not Silver Arrows completion. A longer Clean
probe (`recordings/c11ow_probe_silver_o5.json`) entered with 8.48 hearts but
died inside `level9_natural_silver_arrows` at room `0x05`. A separate rock
wait experiment (`c11ow_after_o0_verified.json` through
`c11ow_after_o7_verified.json`) increased `0x05` damage on offset 7 until it
died there; that experiment was reverted. Neither those probes nor the
resumed comparison moves the continuous power-on STATUS gate.

Next action: find a natural bomb or rupee source that covers offsets 0–4,
then reduce the large `0x05`/L9 interior damage and recheck the Silver
Arrows chapter before a continuous power-on credits run.
