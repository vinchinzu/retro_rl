# Residual — SM-PRIM-01

### Result
GREEN

### Files changed
- `routes/controller_common.py` — added `settle_hold(session, frames=12, reason=...)` thin wrapper around `hold`; exported in `__all__`
- `tests/test_controller_common.py` — frame advance + reason preservation unit test

### Verify paste
```text
$ uv run pytest super_metroid/tests/test_controller_common.py -q
.............                                                            [100%]
13 passed in 0.19s
```
(12 after PRIM-01 alone; 13 after PRIM-02 landed in the same suite.)

### Acceptance
- [x] Helper exported from `controller_common`
- [x] pytest controller_common green
- [x] Residual names SM-PRIM-01B + one change
- [x] No business_climb / continuous / STATUS edits

### Residual risks
- Business climb still uses raw `_hold(..., 12, reason="…_settle")` until SM-PRIM-01B
- Not continuous evidence; pure climb not re-run this card

### Next action (required)
- **Next card ID:** SM-PRIM-01B
- **One change:** Replace eight business climb 12f settle holds with `settle_hold` (serialize on `business_climb.py`)
- **Source state:** `scratch/continuous_like_business_climb_entry.state`

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM
- Not continuous evidence

### Probe pin
N/A — primitive extract only
