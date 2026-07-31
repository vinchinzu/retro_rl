# TASK SM-KRAID-UNIT: Expand Kraid combat pure unit coverage

## Recipe step
boss tests (Kraid already continuous-wired — unit only)

## Model
Luna

## Own files only
- `tests/test_kraid_combat.py` (extend only)

Do **not** edit `combat/kraid.py` unless a trivial import bug blocks a test.
No continuous re-record, no STATUS.

## Context
- Living: `combat/kraid.py` fight → rear door → Varia
- Existing tests cover entry lane, mid spray, body_hp

## Read first
- `combat/kraid.py` (fight_kraid_action, play_* evidence types, rear exit gates)
- `tests/test_kraid_combat.py` (already present)

## Do (thorough — add ≥6 new tests)
Expand coverage without emu:
1. Low HP enemy still sprays (not idle)
2. Zero HP → non-fire / exit-oriented if coded
3. VariaEvidence / KraidVariaEvidence.to_dict key stability (construct directly)
4. KraidStrategy defaults
5. Catalog primary weapon supers + max_hp 1000
6. Action when samus mid-arena y band still includes X or fire
7. Optional: wrap_kraid_as_boss_strategy boss_id smoke from protocol
8. body_hp reads only enemy0

Keep all prior tests green.

## Do not
- Change continuous fight timings for “optimization”
- Pure probe kraid fight (slow)

## Acceptance
- [ ] `uv run pytest super_metroid/tests/test_kraid_combat.py -q` green
- [ ] Net +≥6 tests
- [ ] Diff summary

## Verify commands
```bash
uv run pytest super_metroid/tests/test_kraid_combat.py -q
```
