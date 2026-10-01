# Zelda I spine residual — Level 9 lookahead (2026-10-01, rr-yzb4)

## Current scope

The production change keeps the 56 px `PolicyGuard` hazard gate on the
Patra join only. It uses the shared `ROLLOUT_EVADE_TRIGGER_RADIUS`, whose
existing 24-frame rationale is in [rollout.py](../rollout.py). The unit
wiring check covers the selected radius.

The full Clean experiment `rr_yzb4_gate56` had the gate enabled on both the
Patra join and Ganon. Treat its recording as experimental join+Ganon evidence,
not as a full power-on run of the final join-only production code. It reached
credits in 313,860 frames, with final RAM matching `n9_credits` (TF `0xFF`,
16 containers, 5 bombs, 20R, health 244). Its tape replay matched every frame
with zero loads and zero RAM writes; `$0800` process noise is outside the RAM
sync window.

## Paired join measurement

The experimental run and `n9_credits` share the Patra join start frame
(298,976), stage length (10,058 frames), hearts (1.96 in, 4.96 out), damage
(1.0h in rooms 0x67 and 0x77), and leave CRC. Four button frames differ during
one guard detour; CRCs converge and match at the stage leave and afterward.
The gate reduced checks from 6,687 to 6,163 and rolled frames from 161,913 to
149,965. Stage wall time improved from 440.24s to 415.42s (0.38x to 0.40x).
This is a modest measured improvement, but the join still falls below 1x.

The Ganon arm from the experiment is reverted. It increased the fight from
1,244 to 2,172 frames and from 25.17s to 33.08s while raising `x_realtime`
from 0.82x to 1.09x. The final Patra remains below real time (0.89x in the
experiment).

## Next

`rr-yzb4` remains in progress. Find a faster join policy and compare it at
matched clean join pins across multiple RNG offsets. Require unchanged leave
state and health plus lower wall time; then rerun the full credits gate and
replay its tape before claiming completion. The experiment did not promote
STATUS.

Historical route and gate evidence: [STATUS.md](STATUS.md),
[RUN_METRICS.md](RUN_METRICS.md), and [plan.md](plan.md).
