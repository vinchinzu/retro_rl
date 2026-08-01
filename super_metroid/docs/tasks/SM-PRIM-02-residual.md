# Residual — SM-PRIM-02

### Result
GREEN

### Files changed
- `routes/controller_common.py` — added `short_hop(session, direction, frames, *, buttons_extra=(), reason=...)`; exported in `__all__`
- `tests/test_controller_common.py` — unit coverage for directional hop holds

### Verify paste
```text
$ uv run pytest super_metroid/tests/test_controller_common.py -q
.............                                                            [100%]
13 passed in 0.19s
```

### Acceptance
- [x] Helper exported + tested
- [x] pytest controller_common green
- [x] Residual: next card + one change + source path
- [x] No varia_return / continuous / STATUS edits

### Residual risks
- Door pure `kraid-to-eye-return` still RED (door_transition=0); hop extract does not fix open
- `varia_return.py` not migrated until SM-PRIM-02B

### Next action (required)
- **Next card ID:** SM-PRIM-02B
- **One change:** Migrate kraid-return short-hop call site to `short_hop` without changing 24/20 timings
- **Source state:** `scratch/post_varia_to_kraid_pure.state`

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM
- Not continuous evidence / not pure-green door

### Probe pin
room=0xA59F pose=82 x=37 y=307 door_transition=0 (prior residual class; not re-probed this card)
