# Note — K4 repath: Cathedral first Bubble (planner)

**Date:** 2026-08-01  
**Decision:** Option 1 after `SM-K4.2-PURE` RED.

## Evidence

- Pure Speedway→Farm without Speed stalls at Boost Blocks **x≈795**
  (`SM-K4.2-PURE-residual.md`, loadout `0x1105`).
- Frog Save→Speedway pure remains GREEN (~295f) but is **not** a path to Bubble
  without Speed.

## Product path (no-Speed first Bubble)

```text
Business 0xA7DE
  → Cathedral Entrance 0xA7B3   (blue, top-right)
  → Cathedral 0xA788            (red Super door)
  → Rising Tide 0xAFA3          (green Super door)
  → Bubble Mountain 0xACB3
  → Bat Cave → Speed Hall → Speed Room
```

## Post-Speed shortcut (parked)

```text
Frog Save → Speedway → Farm → Bubble   # requires speed_booster
```

## Graph / code landings

- `progression.py`: cathedral rooms + edges; `speedway_to_farm` /
  `farm_to_bubble` require `_K4_SPEED_CAPS`; `frog_save_to_business` reverse.
- `k4_norfair.py`: scaffold callables for cathedral chain + reverse.
- Cards: `SM-K4-CATH-01` next pure from `post_business_continuous`.

## Non-claims

- No continuous re-record, no STATUS promote, no Speed grant.
