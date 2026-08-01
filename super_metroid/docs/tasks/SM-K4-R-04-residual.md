## Residual — SM-K4-R-04

### Result
RED

### Files changed
- `scripts/probe/kpdr.py` — pure choice `warehouse-to-business` → `play_warehouse_to_business`
- `docs/tasks/SM-K4-R-04-residual.md` — this residual

### Verify paste
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure warehouse-to-business \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_zeela_to_warehouse_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_warehouse_to_business_return.state
```

Exit 1:

```text
success: false
error: warehouse_to_business: ... room_id=0xA6A1 samus_x=722 samus_y=160 ...
roomIdHex: 0xA6A1
samusX: 722
samusY: 160
```

Controller assumes elevator-side entry (x near 126). Reverse Zeela exit lands on the
**right** Zeela door ledge (x≈728, y≈139–160).

### Acceptance
- [ ] Pure green → ordinary Business `0xA7DE` — fail
- [x] Pure CLI wired
- [x] Residual with geometry + next card

### Geometry pin (warehouse reverse)

| Fact | Detail |
|------|--------|
| Reverse source | x≈728 y≈139 after zeela→warehouse pure |
| Elevator | left side, center ~x=126–145 |
| Super-block wall | blocks floor left at **x≈325** y=315 |
| Forward open | from **left** x≈75–100 y≈139, 3 supers, face RIGHT |
| Forward cross | spin RIGHT+B+A on **upper** y≈139–155 through x=300–360 |
| Floor left after open | still hard-stops at x=325 (passage is upper, not floor) |
| Open from right upper | multiple stand_x + super spam — stack does not open a left path |
| No Hi-Jump | reverse pure items `0x1005` |

### Residual risks
- `play_warehouse_to_business` is correct for elevator-side continuous; reverse needs a
  **right-ledge → open stack → upper spin-left → elevator → DOWN** approach class.
- Super-stack open may be left-side only in this room; reverse may need a different
  open geometry or a documented multi-phase controller in `warehouse.py`.
- Not pure-green, continuous, or STATUS.

### Next action (required)
- **Next card ID:** SM-K4-R-04B (planner redesign — warehouse reverse approach)
- **One change:** Replace elevator-only assumption when `samus_x > 400` with reverse
  approach: upper-left to super-stack, open (or confirmed reverse open class), spin
  cross at y≈139 to elevator, then existing DOWN hop.
- **Source state:** `scratch/post_zeela_to_warehouse_return.state` room `0xA6A1` x≈728

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Zeela pure green is separate; this card does not revoke it.

### Probe pin
```text
room=0xA6A1 pose=40 x=722 y=160 door_transition=0
```
