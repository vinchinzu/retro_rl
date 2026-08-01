## Residual — SM-K4-R-02C

### Result
BLOCKED

### Files changed
- `routes/kpdr/kraid_return.py` — replaced the empty post-climb traverse with a bounded Zeela-side x-window setup and retained the Baby Kraid fail guard.
- `docs/tasks/SM-K4-R-02C-residual.md` — recorded the blocked verification and next action.

### Verify paste
`uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state`

Exit code: 1. The probe did not start because importing the route package hit an unrelated syntax error:
```text
File ".../super_metroid/routes/kpdr/kraid_approach.py", line 41
    _select_weapon = select_weapon
IndentationError: unexpected indent
```

`uv run pytest super_metroid/tests/test_controller_common.py -q`

Exit code: 2. Collection was blocked by the same `IndentationError` in `routes/kpdr/kraid_approach.py:41`.

`uv run python -m py_compile super_metroid/routes/kpdr/kraid_return.py`

Exit code: 0. No output.

### Acceptance
- [ ] Pure green into ordinary `0xA471` — blocked before emulator startup by unrelated syntax error.
- [x] Fail loud on `0xA521` — existing explicit Baby Kraid guard remains in the drop loop and the new positioning loop.
- [ ] `uv run pytest super_metroid/tests/test_controller_common.py -q` green — blocked during collection by `routes/kpdr/kraid_approach.py:41`.

### Residual risks
- The new x-window values (`120..180`, approach stop at `x<=180`) have not been emulator-verified because route import is blocked.
- No post-climb room/pose/x/y pin was captured in this session.
- Pure-green, continuous evidence, graph promotion, and STATUS promotion remain unavailable.
- `routes/kpdr/kraid_approach.py` is outside this card's ownership and must be repaired or reconciled by its owner before verification can run.

### Next action (required)
- **Next card ID:** SM-K4-R-02D
- **One change:** Fix climb success guard (must stay in `0xA4DA`; do not RIGHT into east Baby at x≈492) then use recon Zeela door band x∈[96,160]. Approach syntax blocker is already fixed; pure re-run still RED (`upper traverse crossed wrong door` → `0xA521`).
- **Source state:** `scratch/post_baby_to_kihunter_return.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin (if pure/geometry)
room=N/A pose=N/A x=N/A y=N/A door_transition=N/A (probe blocked before emulator startup)
