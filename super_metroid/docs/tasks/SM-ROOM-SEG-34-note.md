# SM-ROOM-SEG-34 note — Red Pirate Shaft practice green

## Summary
Isolated dual-track practice for `room_b139_from_af72_to_afce` (0xB139 → 0xAFCE)
is GREEN and practice-promoted. Existing doorway fixture reused; one policy
geometry edit only.

## Failure that scaffold hit
Baseline `LEFT+A+B` approach pressed into the left mid ledge and ended at
`x=53 y=276 pose=65` still in 0xB139 (total 443f). Continuous jump-hold never
found the mid-platform right gap.

## Working geometry (one change)
1. Walk left off entry ledge (`LEFT+B` 24f) then fall left to mid platform.
2. Walk right into the shaft gap (`RIGHT+B` 25f).
3. Fall with six pulsed `DOWN+X` shots (4f / 20f gap) plus a final door pulse.
4. Hold `DOWN` into the bottom blue door to Acid Snakes Tunnel.

## Integrity
- progression_writes=0, capacity_writes=0, deaths=0
- energy assist writes=0; ammo restore writes only (missiles)
- practice promote ≠ continuous / STATUS evidence
