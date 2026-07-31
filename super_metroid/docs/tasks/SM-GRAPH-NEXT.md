# TASK SM-GRAPH-NEXT: Unit-test reverse-spine ranking still blocks on unverified eye hop

## Recipe step
tests only

## Model
Luna (or Flash)

## Own files only
- `tests/test_progression.py` and/or `tests/test_k4_speed_branches.py`

Do **not** edit progression edge verification values.

## Context
GRAPH-GUARD locked `kraid_to_eye_return` as `unverified`. Extend so
`preferred` / `path_verification` / ranked next-hop APIs (whatever exists)
never treat reverse Business path as continuous-ready while eye hop is
unverified. Add 2–4 tight tests with local fixtures if needed.

## Read first
- existing GRAPH-GUARD tests in test_progression.py
- progression path_verification / next_hops helpers
- QUEUE residual promotion rules

## Do
1. Add tests covering reverse path from Varia/Kraid toward Business: blocking
   edge is eye return while unverified.
2. Ensure promoting that edge in a **fixture copy** unblocks path_verification
   (local mutation only — not production edges).
3. pytest green.

## Acceptance
- [ ] New tests green
- [ ] No production verification edits

## Verify commands
```bash
uv run pytest super_metroid/tests/test_progression.py super_metroid/tests/test_k4_speed_branches.py -q
```
