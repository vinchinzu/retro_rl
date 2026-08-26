## Residual — rr-20w.2.3 D2 field clearing

**Status:** IN PROGRESS. Stop point prepared for handoff; the natural-entry
Day 2 rung is not green.
**Natural entry:** power-on. The named states below are diagnostic pins and do
not promote STATUS.

### Verified this session

- `CarryToPondStand` uses the verified F0 pond stand only. It waits through
  the throw input lock and does not report success until reaching the
  west-of-pond egress `(29,35)`.
- From `Y1_D2_After_Bushes`, fence continuations cleared the farm count
  `80 -> 49 -> 15 -> 0`. The post-fence diagnostic pin is
  `Y1_D2_After_Fences`.
- From that pin, the bounded stone section cleared 10 stones (`185 -> 175`)
  in 2,602 frames. The stable successor pin is `Y1_D2_After_Stones`.
- A valid stand beside the lower half of a 2x2 boulder is now recognized as
  adjacent to its whole footprint. Previously it was compared only with the
  top-left anchor, bounced back to navigation, and eventually timed out.
- Day 2's bounded field contract is 10 bushes, all fences, 10 stones,
  4 boulders, and 2 stumps. A bounded success no longer incorrectly requires
  the entire debris type to be absent from the farm.
- Focused non-ROM suite: 196 passed. Task/planner/script modules also pass
  `compileall`.

### Current red: hammer registration timing

Start the next diagnostic at `Y1_D2_After_Stones` (65 stamina). Do not use
the latest `Y1_D2_Rocks_Frontier`: it was overwritten by an experimental
0/4 run.

The 2x2 stand/pathing bug is fixed. The remaining problem is that the live
ROM registers a hammer hit one frame after the current 49-frame
face/settle/swing/cooldown queue drains. A raw replay against the first
boulder observed:

| Frame | Stamina | Tool counter | Target hits |
|---:|---:|---:|---:|
| 1262 | 63 | 1 | 2 |
| 1311 | 61 | 2 | 3 |
| 1360 | 59 | 3 | 4 |
| 1409 | 57 | 4 | 5 |
| 1458 | 55 | 5 | 6 |
| 1507 | 53 | reset | gone |

This proves that checking the counter as soon as the queue empties rejects a
real swing. That experimental rejection was reverted; the checked-in helper
retains the prior fixed-attempt behavior. Its best stable live result was 2/4
boulders before stamina/aim misses consumed the budget.

### Exact next action

Add a delayed post-swing observation seam (at least the one proven frame)
before deciding whether a swing registered. Count a hit only from a live
tool-counter, stamina, or tile-disappearance edge; on a genuine miss, retry a
different valid footprint side. Re-run only the 4-boulder section from
`Y1_D2_After_Stones`, then clear 2 stumps. If the quota cannot fit the remaining
stamina, compose the existing spa/refill path instead of weakening the quota.

After those sections are green, the remaining product proof is one Clean
power-on run through field clearing, eight wet potatoes, and shipping before
17:00.

### Non-claims

- No STATUS promotion
- No natural power-on Day 2 completion
- No claim that all rocks or stumps are gone
- No claim that `Y1_D2_Rocks_Frontier` is a valid successor
