# TMNT IV

Scriptable TMNT IV: Turtles in Time (SNES). One continuous Hard session from
power-on through the staff/cast credits, played Clean under ROM lookahead.

## Language

**Clean**:
The target contract. Zero RAM writes (no HP, iframe, stage, or lives
pokes), zero state loads, zero life losses, no A-special. Natural pizza is
the only heal. ROM lookahead is allowed and disclosed.
_Avoid_: calling an assisted or resumed run Clean

**Continuous run**:
One emulator session from power-on to the credits. Lookahead restores do not
break it; any other `set_state` does.
_Avoid_: concatenating stage pins as a clear

**Lookahead**:
Playing candidate futures on the emulator itself and restoring. The ROM is
exact (restore + replay is bit-exact), so a rollout *is* the future.
_Avoid_: modelling enemy motion by hand

**Guard** (`GuardedPolicy`):
The production wrapper. Commits only frames whose next `horizon` frames the
ROM has shown hit-free.
_Avoid_: a second policy beside it

**Verified plan**:
The guard's queue of future frames, each with the RAM CRC it expects to
start from, extended a chunk at a time past the live frame.

**Rescue**:
On a predicted hit, branching every macro from a snapshot node inside the
verified plan and splicing in the best hit-free branch. **Lead** is how many
frames before the hit the branch starts.

**Forced**:
A rescue with no hit-free branch; the guard takes the least damage. Every
forced line is a defect to work, not noise.

**Stall budget**:
When a fight has made no progress for 1800 frames, the guard accepts a
predicted hit that leaves HP at or above half the bar (`accepted`). Some
progress (the Technodrome tank throw) needs a trade.

**Unstick**:
While stalled, a 120-frame search over every macro for a branch that
damages an enemy. Breaks base-policy deadlocks without a per-tactic patch.

**Macro**:
A literal button prefix (walk, jump, jump-kick, poke) the guard tries before
handing control back to the base policy.

**Base policy** (`Stage1Policy`):
The behavior tree + `tactics/` that walks, fights, seeks pizza, and handles
bosses. Its job is progress. The name is historical; it covers stages 1-9.

**Tactic**:
A stage- or boss-specific `next(state)` inside the base policy that returns a
frame or falls through.

**Tape**:
The committed button vectors of one trial plus RAM CRC checkpoints
(`*.tape.npz`). **Replay** plays a tape with no policy and no lookahead; the
published video is a replay.

**Natural pin**:
A save state written at a stage entry of a real run (`--save-points`). The
dev entry for that stage. Not evidence on its own.
_Avoid_: hand-crafted HP pins as Clean evidence

**Assist**:
The old contract: emergency HP (HP ≤ 16 → 80) and the Super Shredder form-2
iframe hold. Previous M8 baseline, retained for comparison; never Clean.

**Pizza**:
Ground box char `0x30`. Picking it up with Y is play.

**Stage byte**:
RAM `0x0082`. Byte 0 is Big Apple (human Stage 1); human stage = byte + 1.
