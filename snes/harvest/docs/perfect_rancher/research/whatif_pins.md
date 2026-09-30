# What-if pins

NEVER route evidence. These saves are RAM research pins. A route must not load them and must not treat their scores as a Clean run.

Built with `harvest.tools.state_preset_builder` from the JSON under `/tmp/oracle_presets/`. Names are `LN_oracle_whatif_*`. Each output name was absent before the write.

`state_preset_builder` writes one happiness value onto every cow slot. `LN_oracle_whatif_cows_mixed_happiness` got a follow-up byte write at cow slot +4 after the builder returned. The builder was not edited.

`set_house_level` accepts 0, 1, or 2. Level 2 sets both upgrade bits `$7F1F64 & 0x40` and `& 0x80`. That is the third house size, so the married pin is named `house_l3` and was built with `flags.house_level` 2. On the power-on base the upgrade word became `0x00C3`.

## Oracle

`RanchMasteryCalculator` is `$83F26D`. After `%Set16bit(!M)`, `LDA.L !moneyL` loads `$7F1F04`. The money term is `min(78, (money_raw & 0xFFFF) >> 7)`.

Hearts and shipped crops use `(value & 0x01FF) >> 4`. Cow points walk all 12 slots at `$7EC1C6`, size `0x10`. Existence is status bit 0. Points are `min(25, happiness_byte >> 3)`. The sum is capped at 999.

Rating lines are event opcode `0x45` `Event_JumpifBetweenValue` at `$B5A90E`, tested on `$7F1F54`. Inclusive bands, bank_83 text table:

| Score | Text id | Block |
| --- | --- | --- |
| 0 to 199 | `0x0483` | `DATA16_BBB384` |
| 200 to 399 | `0x0488` | `DATA16_BBB640` |
| 400 to 599 | `0x0489` | `DATA16_BBB692` |
| 600 to 799 | `0x048A` | `DATA16_BBB700` |
| 800 to 999 | `0x048B` | `DATA16_BBB78E` |

The marriage milestone cites text `0x01C3`, pointer `DATA16_B8892A` `Text_Peddler_BlueFeather1`.

## Expected terms

Live load on 2026-09-29, one emulator, `make_harvest_env` then `reset`, matched the save-state oracle on all eight pins. `fails=0`.

Power-on base terms, before the edits below, are money 2, girls 5, happiness 1, development 1, total 9, text `0x0483`.

**LN_oracle_whatif_married_house_l3.** Base `LN_final_poweron_ending`. House 32, marriage 32, total 73, text `0x0483`. Other terms stay on the power-on base. `upgrade_flags` `0x00C3`, `marriage_flags` 1. Y3 Summer D30.

**LN_oracle_whatif_cows_mixed_happiness.** Base power-on. 12 cows, `num_cows` 12, cows score 36. Slot happiness bytes `0, 7, 8, 16, 24, 40, 80, 100, 199, 200, 201, 255`. Slot points `0, 0, 1, 2, 3, 5, 10, 12, 24, 25, 25, 25`, sum 132. Total 177, text `0x0483`.

**LN_oracle_whatif_chickens_12.** Base power-on. `count` 12, `total_slots` 13, `status_raw` 9. Chickens score 36, `num_chickens` 12, total 45, text `0x0483`.

**LN_oracle_whatif_girls_480.** Base power-on. Each of five girls at 480 hearts scores 30. Girls 150, total 154, text `0x0483`.

**LN_oracle_whatif_y3_summer_d20_near_perfect.** Base `latest_y3_ending_ranch_master_999`. Calendar year raw 2, season 1, day 20, happiness 960. Happiness score 30. Every other term is at its max: money 78, cows 36, chickens 36, stamina 50, girls 155, crops 124, house 32, marriage 32, children 32, items 43, development 50, cow happiness 300. Total 998, text `0x048B`.

**LN_oracle_whatif_farm_covered.** Base power-on. `fill_non_structure_with_grass` true, `update_visible_map` false. `ranch_development` byte 100, development score 50, total 58, text `0x0483`.

**LN_oracle_whatif_mixed_wrap.** Base `latest_y3_ending_all_unmarried`, still Y3 Summer D29 6:00. `money_raw` 66536 scores 7, because `66536 & 0xFFFF` is 1000 and `1000 >> 7` is 7. Hearts Maria 512, Ann 496, Nina 480, Ellen 495, Eve 511 score `0, 31, 30, 30, 31`. Crops tomatoes 512, corn 496, potatoes 495, turnips 1024 score `0, 31, 30, 0`. Pre-sleep total 626, text `0x048A`. Other pre-sleep terms: cows 24, chickens 24, stamina 50, happiness 31, house 32, marriage 0, children 32, items 43, development 0, cow happiness 200.

**LN_oracle_whatif_mixed_mid.** Same unmarried D29 base. `money_raw` 4000 scores 31. All five hearts 160 score 10. All four crops 64 score 4. Happiness 320 scores 10. Pre-sleep total 512, text `0x0489`. Herd and item terms match the wrap pin.

## Ending probes

`harvest.tools.ending_probe`, cwd `snes/harvest`, `HEADLESS=1`, `SDL_VIDEODRIVER=dummy`, `QT_QPA_PLATFORM=offscreen`, `--sleep-count 1`, `--stop-ending-index 0x25`, `--save-end`, `--ram-set` empty. One emulator, sequential. Oracle on the saved end RAM equals RAM `ranch_mastery` at ending index `0x25` on all six.

The first sample at index `0x20` is taken while `RanchMasteryCalculator` is still the current scene. The routine zeros `$7F1F54` on entry, then stores the sum, then the index advances. The comparison used here is the terminal `0x25` snapshot. `LN_final_poweron_ending` was already at index `0x25` when the probe loaded it, so that run has no `0x20` sample.

| State | Max frames | `0x20` frame | Score at `0x20` | `0x25` frame | RAM score | Oracle |
| --- | --- | --- | --- | --- | --- | --- |
| `LN_final_poweron_ending` | 8000 |  |  | 5000 | 9 | 9 |
| `latest_y3_ending_all_unmarried` | 45000 | 25717 | 0 | 26124 | 822 | 822 |
| `latest_y3_ending_all_married` | 45000 | 34686 | 0 | 35093 | 742 | 742 |
| `latest_y3_ending_ranch_master_999` | 45000 | 35483 | 999 | 35890 | 999 | 999 |
| `LN_oracle_whatif_mixed_wrap` | 45000 | 25717 | 0 | 26124 | 664 | 664 |
| `LN_oracle_whatif_mixed_mid` | 45000 | 12315 | 0 | 12722 | 550 | 550 |

JSON is `snes/harvest/debug_alignment/ending_probe_<state>.json`. End states are `custom_integrations/HarvestMoon-Snes/<state>_ending_probe_end.state`.

On the wrap pin the end RAM still has `money_raw` 66536 and money score 7. Happiness fell 999 to 969, score 31 to 30. `ranch_development` rose 0 to 78, score 0 to 39. `626 - 1 + 39 = 664`. The same end RAM with money scored as `min(78, 66536 >> 7)` would be 735. The ROM total is 664.

On the mid pin happiness fell 320 to 290, score 10 to 9, and development rose the same way. `512 - 1 + 39 = 550`.

The 999 showcase was already storing 999 before the calculator, which is why the `0x20` sample still reads 999. The `0x25` RAM matches the oracle at 999, text `0x048B`, every term at its max.
