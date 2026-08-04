## Residual — SM-K4.4-PURE

### Result
**GREEN** (closed by R19 — full pure Bubble → ordinary Bat Cave)

Authoritative closeout residual:
[`SM-K4.4-PURE-R19-residual.md`](SM-K4.4-PURE-R19-residual.md).

### Summary

| Field | Value |
|-------|-------|
| Frames | **2012f** ×2 matching pure probes |
| Exit | ordinary `0xB07A` Bat Cave ~(39,395) p11 |
| Successor | `scratch/post_bubble_to_bat_pure.state` |
| Phase D | enemy-phase idle wait (Geruta slots 4/6 classes A/B) |
| Phase E | sticky period-10 right WJ + Super pressure |
| Graph edge | `bubble_to_bat_cave` → `controller_dev` |

### Acceptance

- [x] Source loads at `0xACB3` (CATH-04 pin band)
- [x] Ordinary `0xB07A` without warp / item grants
- [x] Successor written only on pure GREEN
- [x] Unit green (30 scaffold)
- [x] Residual + tip boards; continuous tip **not** advanced

### Probe

```text
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin_r19.json --no-red-diag
# success=true roomIdHex=0xB07A frames=2012
```

### Next action

- **Next card:** Bat → Speed Hall pure from `post_bubble_to_bat_pure`
  (`SM-K4.4-GRAPH` / `SM-K4.5-PURE` spine)
- Continuous tip remains Frog Save until planner compose/stabilize
- Do **not** re-open Bubble Phase D isolation or enemy RAM patches as product

### Non-claims

- Did not STATUS-promote continuous tip past Frog Save
- Did not graph-compose Cathedral→Bubble→Bat into continuous yet
- Enemy phase wait is read-only geometry; no enemy RAM writes
