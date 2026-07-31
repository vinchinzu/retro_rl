# TASK SM-GRAPH-GUARD: Unit guards against inflated verification promotion

## Recipe step
tests / integrity contracts (unit only — **no emu continuous**)

## Model
Luna

## Own files only
- `tests/test_progression.py` and/or `tests/test_k4_speed_branches.py` (**edit/add**)
- optional tiny helper in test fixtures only

Do **not** edit progression edge verification levels to “make tests pass.”
Do **not** touch continuous.py / STATUS.

## Context
Wave-3 left `kraid_to_eye_return` **unverified** (pure still red). Cheap
executors and humans both tempt promoting graph edges after scaffold or
diagnostic success. This card hardens **unit** rejection of that inflation.

Known facts to lock:
- `varia_to_kraid` may already be `controller_dev`
- `kraid_to_eye_return` and remaining reverse hops are `unverified` until pure green
- `path_verification` / ranking must not treat `unverified` as continuous-ready

## Read first (all)
- `progression.py` `_K4_SPEED_EDGES` reverse hop section
- `tests/test_k4_speed_branches.py`
- `tests/test_progression.py` path_verification tests (SM-K4-04 style)
- `docs/tasks/QUEUE.md` residual promotion rules

## Do (aggressive contract pressure)
1. Add tests that fail if `kraid_to_eye_return.verification == "continuous"`
   or `"controller_dev"` **given current source tree** (assert present value is
   `unverified` — documents gold).
2. Add test: a synthetic path that includes an `unverified` reverse edge is
   **not** `all_continuous` / is blocking (reuse SM-K4-04 patterns).
3. Add test: ranking prefers continuous over controller_dev over unverified
   when choosing next hop (if API exists; else skip with comment).
4. Do **not** promote any edge in production code.
5. pytest green.

## Residual required
- What is locked vs what still needs pure green
- Explicit: tests do not replace pure probe or continuous integrity

## Do not
- Edit edge verification in progression.py
- continuous / STATUS
- Soft-assert / xfail the locks

## Acceptance
- [ ] New tests green and would fail if someone “helpfully” promotes the edge
- [ ] No production verification changes

## Verify commands
```bash
uv run pytest super_metroid/tests/test_progression.py super_metroid/tests/test_k4_speed_branches.py -q
```
