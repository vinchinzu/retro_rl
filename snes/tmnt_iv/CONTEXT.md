# TMNT IV

Scriptable TMNT IV: Turtles in Time (SNES). One continuous Hard session from
power-on through staff/cast credits. First credits is assisted Bronze; Clean
(pizza-only) is the parallel track. Maturity stays M8.

## Language

**Continuous hard run**:
One emulator session from power-on through Hard staff/cast credits. No state
loads, no stage/lives writes, no A-special.
_Avoid_: checkpoint concat as the clear, mid-run save loads, labeling assisted
credits as Clean

**Clean**:
Zero emergency HP writes and zero form-2 iframe holds. Natural pizza is the
only heal. Parallel to the assisted contract, not a new maturity gate.
_Avoid_: emergency HP as Clean, iframe hold as Clean, A-special, global pizza
seek

**Assist**:
The production contract: emergency HP (HP ≤ 16 → 80) and Super Shredder form-2
iframe hold at 1. Counted in the manifest. Default continuous CLI stays
assisted until Clean is proven per stage.
_Avoid_: full-bar HP spam, item unlocks, stage writes, calling Assist a pizza
pickup

**Pizza**:
Ground box char `0x30`. Collecting it with Y is play, not an Assist. Full seek
is Big Apple only; Alleycat/Sewer grab underfoot or between waves.
_Avoid_: emergency HP as pizza, global seek on Skull & Crossbones, empty-screen
RIGHT+Y

**Stage byte**:
RAM stage at `0x0082`. Byte 0 is Big Apple (human Stage 1). Human stage numbers
in docs are byte+1.
_Avoid_: calling byte 1 “Stage 1”, treating Mode-7 Neon as a side-scroller Y

**Stage1Policy**:
The one production tick for every stage. The name is historical.
_Avoid_: a second policy per stage, a Leo-only policy as the continuous path

**Tactic**:
A stage- or boss-specific next(state) that may return a frame or fall through.
Production tick order is pizza → pack → spikes → Baxter → Technodrome → cave →
Slash → form-2 → combat tree → stall escape. HazardAvoid is not in that order.
_Avoid_: fight_action as a second undocumented dispatcher, KEEP traces as
production Slash

**CombatProfile**:
Per-frame poke-band, flank, and jump overlay for the shared fight. Grind knobs
overlay a subset; Alleycat/Sewer bands are not grindable.
_Avoid_: treating GrindKnobs as the live alley/sewer standoff

**Slash**:
Prehistoric boss char `0x50`. Production is SlashTactics, spin dodge adx 52.
Lab patterns are research adapters, not a second policy.
_Avoid_: rewriting production from a KEEP trace, porting probe spin=40 into
the continuous run

**KEEP**:
A grind trial that beat its baseline on a short probe. It is not production
until a stage suite and a continuous dry-run both hold.
_Avoid_: merging KEEP into policy.py automatically, checkpoint-only KEEP

**Path-RNG suite**:
Clean proof: fight-ready checkpoint plus at least one power-on or
continuous-faithful entry, heal=none, zero life losses.
_Avoid_: a single fight-ready pin as Clean, mid-wave Clear_w* as the gate
