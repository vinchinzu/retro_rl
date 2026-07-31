# TASK SM-BT-UNIT: Expand Bomb Torizo pure action unit tests

## Recipe step
boss tests (no production strategy rewrite unless bug blocks tests)

## Model
Luna or Flash

## Own files only
- `tests/test_bomb_torizo_strategy.py` (**create**)

Do **not** edit `combat/bomb_torizo.py` unless a clear bug makes tests impossible
(prefer testing current behavior). No continuous / STATUS.

## Context
- Living continuous boss: Bomb Torizo strategy in `combat/bomb_torizo.py`
- Existing feature tests: `tests/test_combat_features.py` (do not duplicate wholesale)
- Pattern: `tests/test_kraid_combat.py`

## Read first
- `combat/bomb_torizo.py` (full: strategy, fight_bomb_torizo_action, evidence)
- `tests/test_kraid_combat.py`
- `tests/test_combat_features.py` (what is already covered)
- `combat/features.py` (`bomb_torizo_catalog`)

## Do (thorough)
Add a dedicated strategy test module with **≥8** unit tests covering:
1. Catalog max HP / room / weapon
2. Statue / spawn spritemap → non-aggressive or activation-seeking actions
3. Active fight: distance too close → retreat direction
4. Active fight: distance too far → approach
5. Mid-range → fire present on fire_period frames
6. enemy0_hp == 0 → empty/idle actions
7. BombTorizoEvidence.to_dict keys stable
8. Strategy dataclass defaults sensible (max_fight_frames > 0)
9. Optional: wrap_bomb_torizo_as_boss_strategy import smoke

Use synthetic `parse_state` + dataclasses.replace like kraid tests.

## Do not
- Rewrite the continuous fight policy
- Emulator pure probes
- STATUS claims

## Acceptance
- [ ] `uv run pytest super_metroid/tests/test_bomb_torizo_strategy.py -q` green
- [ ] ≥8 tests
- [ ] Diff summary

## Verify commands
```bash
uv run pytest super_metroid/tests/test_bomb_torizo_strategy.py -q
```
