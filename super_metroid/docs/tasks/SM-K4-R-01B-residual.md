# SM-K4-R-01B Residual

## Result
**GREEN** (pure) · graph `controller_dev` · not continuous

## Files changed
- `routes/kpdr/kraid_return.py` — jump-left mid-room + X-only door + jump-enter
- `progression.py` — `eye_to_baby_return` → `controller_dev`
- `tests/test_progression.py` — lock controller_dev chain

## Verify paste
```text
pure eye-to-baby-return → success room=0xA521 frames=651
pytest test_controller_common + test_progression → 38 passed
```

## Next action
- **Next card ID:** SM-K4-R-01C (baby→kihunter) — **also done this session**
- **Then:** SM-K4-R-02 kihunter→zeela (RED / open)

## Non-claims
- Not continuous · no STATUS promote · no progression RAM forge
