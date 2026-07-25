# Plan — Star Fox

## Immediate bottleneck

Destroy Attack Carrier's three hatches and finish the Route 1 Corneria clear
from natural entry after the current segment policy.

## Next acceptance test

Corneria clear from `CorneriaStart.state` with hard timeout, then the same clear
from a natural-entry state captured after the Route 1 map transition.

## Next three milestones

1. M3→M4: natural-entry Corneria clear
2. M5: chain into the next Route 1 stage without state load
3. Publish evaluation contract for Route 1 ending evidence

## Deferred ideas

- Full Route 1 continuous dry run (M7)
- Silver/Gold observation experiments

## Infrastructure blockers

None currently; Super FX boot and controller injection are verified.
