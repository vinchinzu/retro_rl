# Plan — TMNT IV: Turtles in Time

Verified facts: [STATUS.md](STATUS.md). Play lessons:
[CLEAN_PLAYBOOK.md](CLEAN_PLAYBOOK.md). Assist:
[ASSIST_CONTRACT.md](ASSIST_CONTRACT.md). Tracker: `bd ready -l tmnt_iv`.

**Doc consolidation (2026-08-18):** deleted the second ticket board
(`docs/tasks/` — QUEUE / TRIAGE / BACKLOG / PROCESS / CLEAN_LADDER /
~50 T4-CLEAN and T4-ASSIST cards and residuals) plus `TASK_TEMPLATE.md`
and `CLEAN_TRACK.md`. Kept STATUS, plan, ASSIST_CONTRACT, CLEAN_PLAYBOOK,
ram_map, BASELINE_METRICS, and Slash lab notes. In-flight Clean work
lives here and in beads — do not recreate a QUEUE.

## Control style

Side-scrolling beat-'em-up (move, jump, attack). Attack = **Y**; avoid
special **A** (HP cost).

## Useful RAM

Documented in `docs/ram_map.md`. Highlights:

- Player `0x0400 + {X:0x08, Y:0x0C, HP:0x4A}`
- Enemies `0x08D0 + i*0x70` (same relative layout; skip April `char 0xC4`)
- Menu `0x0032`, event `0x0070`, stage `0x0082`, lives `0x1AA0`
- Pizza `char 0x30` (Clean heal); hazards `0x32`/`0x36` (Stage 1)

## Development approach

1. `uv run python scripts/setup_rom.py`
2. `SDL_VIDEODRIVER=dummy uv run python scripts/boot_probe.py`
3. Clear one segment at a time from save states
   (`python -m tmnt_iv.scripts.run_segment --stage N`).
4. **Clean proof per stage** (heal=none, multi-entry) before removing
   assists — add a `CleanProbeSpec` in `run/clean_suite.py`, not a copied loop.
   Tests stay ROM-free and protect finish / time / damage, not file layout.
5. Continuous validation:
   `uv run python -m tmnt_iv.scripts.record_full_hard_run`.

## Milestones (segment clear — done)

Full stage chain power-on → hard credits under low-assist is **done**
(see `STATUS.md` / `BASELINE_METRICS.md`). Historical segment work is
complete; remaining work is **Clean** (zero assists).

## Clean-hard architecture and tooling review (2026-08-30)

### Conclusion

The combat policy is no longer the main architecture problem: the production
dispatcher is small and stage tactics already live under `tactics/`. The
blocking seam is the **run/proof layer**. `run/clean_suite.py`,
`scripts/probe_boss_metrics.py`, `run/segment.py`, and the continuous recorder
each own a different emulator loop, outcome vocabulary, metrics path, and
failure behavior. That makes checkpoint results hard to compare with the real
power-on run.

Current evidence says the route is not close enough to survive by merely
turning assists off:

- Clean coverage stops at stage bytes **0–2**; only Big Apple is green.
- Alleycat is **2/4**: `Stage2` dies after a 24-damage hit and three later
  12-damage hits; the Stage-1-clear entry times out with
  `combat_stall_escape=9,660` and `edge_press=8,745`.
- `LiveHardStage3` reaches Rat King at **18 HP** after 62 wave damage,
  including two 16-damage spike hits; it dies during the boss.
- Later continuous assisted damage remains far above one health bar:
  Technodrome 971–1,022, Prehistoric 861–1,091, Skull 642+, Wounded Knee
  520+, Neon 449+, and Starbase 1,009 in the latest scratch run.
- The published-baseline pointer is not trustworthy: `BASELINE_METRICS.md`
  names `tmnt_iv_full_hard_dry_run.json` as **206,718f / 4,667 damage**, but
  that file currently contains **210,082f / 5,801**. The `rr_iprz5` scratch
  report is **212,829f / 5,529**. Evidence must be immutable before tuning.

Clean below means the full local contract: zero emergency-HP writes, zero
form-2 iframe writes, no A-special, zero life losses, one power-on Hard
session through completed staff/cast credits.

### The ten fixes, in dependency order

#### 1. Deepen the run module around one trial interface

Replace the four per-frame loop implementations with one deep module whose
interface is approximately `run_trial(entry, objective, contract, limits) ->
TrialResult`. Entries describe power-on or a state; objectives describe stage
advance, boss/fade completion, or credits; the contract describes Clean or
assisted play. Boot, observation, policy reset at natural stage entry, outcome
detection, metrics, and failure capture stay inside the implementation.

Do not add pass-through wrappers or expose tactic internals through this
interface. Keep the emulator as the existing adapter until a real second
adapter exists. Migrate Clean suite, boss probe, segment, and continuous
recorder callers, then delete their duplicate loop logic.

**Exit:** the same `TrialResult` schema and outcome rules drive every probe and
the power-on recorder; a result from a checkpoint and its continuous entry is
directly comparable.

#### 2. Make the Clean contract fail closed

Put all emulator mutations and controller actions behind an audited run
contract. Count actual RAM writes by address, state loads after launch, and
forbidden buttons; do not report `state_loads_zero`, `stage_writes_zero`,
`lives_writes_zero`, or A-special uses as hard-coded constants. Natural pizza
must be inferred from HP gain plus a real pickup transition, not merely labeled
`assist: pizza_only` because the probe did not call `apply_emergency_hp`.

**Exit:** any HP, iframe, stage, lives, or unknown RAM write makes a Clean trial
fail with the offending address/frame; any post-launch state load or A press
does the same. The full-run manifest is derived from the audit log.

#### 3. Fix result semantics before collecting more evidence

The existing Clean report has correctness holes: waiting entries hard-code
`start_hp=80`, the lives string uses the last observed value rather than a
captured start value, game-over detection differs by stage, checkpoint
exceptions are converted to rows while an extra-entry exception aborts the
suite, and Clean rows omit explicit assist-write counters. Normalize these in
the shared result module.

Capture entry and exit snapshots after gameplay becomes live; recognize KO,
player-dead, title/game-over, life decrement, forbidden action, timeout,
freeze, boss-down-with-fade, stage advance, and credits as distinct outcomes.
A Clean trial must never run the continue handler after a KO and later appear
successful.

**Exit:** every result includes truthful start/end HP and lives, numeric life
losses, all intervention counters, entry stage/event/character/difficulty,
target-specific completion evidence, and a stable failure classification.

#### 4. Add immutable evidence and save-state provenance

Stop overwriting named baselines. Write run artifacts under an immutable stem
containing date, commit, contract, and a short digest; promote a baseline only
through an explicit command that updates a small index. Make STATUS and
BASELINE_METRICS point to the indexed digest, not a mutable filename.

Build a state catalog for the very large `custom_integrations/TMNTIV-Snes/`
set. Each gate state needs: file hash, source run digest/frame, character,
difficulty, stage/event, HP/lives, assists already used in the prefix, and
whether it was naturally captured or RAM-crafted. Capture a progressive
Raphael Clean ladder from real predecessor clears; quarantine crafted and
healed pins from Clean gates.

**Exit:** the 206,718f / 4,667 baseline is either restored with its matching
digest or explicitly retired; a Clean suite refuses an unproven, wrong-character,
wrong-difficulty, or development-healed state.

#### 5. Build a failure flight recorder and progress watchdog

Generalize `FreezeWatch` beyond enemyless constant-X stalls. Keep a bounded
frame ring containing action/reason, player pose/animation/iframes, camera and
progress, event, hazards/pickups, and every enemy's kind/pose/animation/HP.
On damage, pizza, boss phase, KO, timeout, or progress stall, emit the relevant
window plus screenshot and recoverable state.

Progress should be stage-specific: camera/progress for streets, enemy or boss
HP for locked fights, event/form for bosses, and Mode-7 lane progress for
Neon. This catches the Alleycat bridge loop with living enemies instead of
waiting 12,000 enemyless frames.

**Exit:** every failed trial identifies the last real progress, first repeated
state/action cycle, and the exact pre-hit or pre-stall geometry in one artifact.

#### 6. Close both Alleycat failures, not just the checkpoint death

Use the flight recorder to split the work into two bugs:

1. At progress ~21,611–22,044, stop the left `0x5E` clump from landing the
   24/12/12/12 sequence. Trace enemy animation/facing and Raphael hurtbox to
   add a bounded right-shoulder exit that is neither walk-through, long hold,
   `LEFT+Y`, nor the rejected global jump-hop.
2. At the Stage-1-clear entry, isolate why `combat_stall_escape` and
   `edge_press` alternate for ~18k frames with no useful progress. Make the
   stage-specific recovery own that loop; do not weaken the global stall
   behavior for stages that already finish.

**Exit:** `Stage2`, late, Boss2, and Stage-1-clear entry all advance with zero
assists/life losses, twice, and no single recovery reason dominates a stagnant
window.

#### 7. Make Sewer survival predictive

The current spike rule reacts within an X radius but still takes two 16-damage
hits, and ordinary wave trades leave Rat King entry at 18 HP. Extend RAM
observation/trace with spike column phase, relative velocity, player airborne
state, and landing lane. Choose the jump from time-to-contact and safe landing,
not distance alone. Then use the same hit windows to remove the remaining
6/4-damage wave contacts without reintroducing dumpster thrash.

Keep Rat King and post-kill fade as separate objectives so a boss kill cannot
hide a fade death.

**Exit:** LiveHard and Stage-2-clear entries reach Rat King at **≥70 HP** and
advance through the fade with zero assists/life losses; the full Sewer
multi-entry suite passes twice.

#### 8. Extend the Clean ladder through Technodrome, Slash, Skull, Wounded Knee, and Neon

Add declarative trial rows for stage bytes 3–7 using the shared interface;
remove the `CLEAN_SPECS == {0,1,2}` ceiling. Every stage gets a Raphael
fight-ready entry, a predecessor/natural entry, explicit boss/fade completion,
and a damage-by-wave/hit budget. Work strictly in route order so the natural
entry state comes from the newly Clean predecessor.

Prioritize by survival rather than assisted speed: Technodrome duo/tank
contact and throws, Slash spin-52 play, Skull with no global pizza seek,
Wounded Knee stacked `0xB0`, then Neon Mode-7 depth. A checkpoint KEEP cannot
advance the ladder until its predecessor entry also passes.

**Exit:** stage bytes 3–7 each have a twice-green multi-entry Clean suite and
can be chained from Big Apple through Neon with zero assists/life losses.

#### 9. Prove Raphael Starbase and form-2 without protection or health writes

Capture the missing continuous-faithful Raphael form-2 pin; the existing
`Boss9_phase2` evidence is Leo, uses emergency heals, and dies after the boss
at heal=none. Instrument form-1/form-2 identity, projectile kind/position,
Shredder animation, player iframe source, demutation, and leftover-flame/fade
frames. Preserve the x=126 dumpster and the proven x=207/right-rail exits while
removing the 1,009-damage Starbase path.

Treat three gates separately: Starbase waves, form 1, and form 2 plus post-kill
fade. The iframe hold is removed only after the form-2 action policy wins on
Raphael's real continuous entry health.

**Exit:** all three gates pass from checkpoint and natural entry twice with
zero HP/iframe writes and zero life losses; stage 9 reaches completed credits
when entered from the Clean Starbase predecessor.

#### 10. Add progressive power-on validation and publish only immutable proof

Add a development mode such as `--clean-through STAGE` that runs from power-on
with both assists forbidden through the selected stage, records per-stage
contract counters, and may enable disclosed assists only after that gate. This
validates natural entry continuously without spending a full hour on every
early-stage edit. It must use scratch artifact stems and can never be labeled
Clean.

After all stage gates pass, run the real `--clean --dry-run` unchanged from
power-on. Require two complete dry-runs to expose path/timing variation, then
one recorded run with the same immutable manifest/digest. Promote STATUS only
from that recorded artifact.

**Exit:** power-on Hard → completed staff/cast credits, zero RAM assists,
zero A-special, zero state loads after launch, zero life losses, immutable JSON
and video evidence; close `rr-t4cl` only then.

### Delivery slices

| Slice | Items | Result |
|-------|-------|--------|
| A — trustworthy harness | 1–5 | **Done** — `run_trial` is the loop; Clean audit fails closed; results capture live HP/lives; scratch stems + catalog helper; stall recorder |
| B — early route | 6–7 | Alleycat scratch **2/4** (LATE+Boss2). Stage2 / stage1_clear still KO on 0x5E 24s. Sewer LiveHard reaches Rat King at **10 HP** then KO (3 residual 16s) |
| C — middle route | 8 | Technodrome → Neon chains Clean |
| D — finale | 9 | Starbase + both Shredder forms + fade are Clean |
| E — publication | 10 | Repeated power-on dry-runs and one immutable recorded Clean clear |

## Path to Bronze / Clean (whole game)

Emergency HP (≤16→80) and form-2 iframe hold are the only production
assists. Clean means both at **0**, pizza + play only.

Stage order (infra first, then stages, then ★ full run):

| Pri | Work | Exit criteria |
|-----|------|---------------|
| 0 | Clean infra (`*_clean` stems, `--clean` CLI, zero-assist asserts) | **Done** — assisted defaults intact |
| 1 | Alleycat Clean (BOSS+LATE done; REACH → CKPT → BRIDGE → suite) | Multi-entry 0 e-heals, 0 lives lost |
| 2 | LiveHard Sewer; residual 0x1C spikes; Rat King | Multi-entry 0 e-heals, 0 lives lost |
| 3 | Technodrome duo + tank | Suite green |
| 4 | Slash spin **52**; no blind spin-40 | Suite green |
| 5 | Skull; **never** global pizza seek | Suite green |
| 6 | Wounded Knee Raph cadence | Suite green |
| 7 | Neon near-band Mode-7 | Suite green |
| 8 | Form-2 dodge **without** iframe write | iframe frames → 0 |
| 9 | Power-on hard dry-run, both assists off | Bronze / Clean publish |

Parallel assisted polish (does not block Clean): Technodrome 1,022 /
Prehistoric 861 / Starbase 749 / Wounded Knee 579, then a planner
dry-run before BASELINE promote. Sitting map: `RAPH_SPEED_HANDOFF.md`
+ wiki notes `SPEEDRUN_STRATEGIES.md` (`rr-iprz`). Raph tools are
jump-kick and dash+Y, not standing Y and not A-special.

### Non-negotiable process (every stage)

1. Read `CLEAN_PLAYBOOK.md` anti-patterns before changing policy.
2. `heal=none` checkpoint probe first.
3. Second entry: natural / continuous-faithful / power-on — not only
   fight-ready state.
4. If a “clever” dodge helps checkpoint but fails suite → **keep offline**.
5. Full dry-run only after stage suite is green (or for assist-count
   tracking mid-rollout).
6. Prefer `RaphFullHard*` for continuous; menu selects Raphael.

### Parked (do not re-open blindly)

| Item | Why parked |
|------|------------|
| Stage 1 `HazardAvoid` in production tick | Jump-through killed Clean; offline wins suite |
| Slash `spin_dodge_adx=40` | Probe win, continuous +807 total dmg |
| Global pizza seek | Skull soft-lock |
| Empty-screen walk `RIGHT+Y` | Stutter; no Clean benefit |

## Local grind (optional)

Farm short probes to Ollama via `scripts/run_local_grind_agent.py`.
Whitelist knobs only (`grind_knobs.py`); KEEP does not auto-edit
`policy.py`. Prefer stage-local knobs; never merge KEEP without suite +
dry-run rules above.

## Stage combat notes (stable)

Faster than Final Fight; align-then-poke. Y axis normal screen coords.
Stage 2 dumpster: frozen `player_x` → DOWN + JUMP+RIGHT. Far-park Foot:
widen right margin. Sewer (byte **2** only): Y-clamp spikes, hold RIGHT.
Rat King: long poke, not jump-slash. Technodrome duo: left-flank + stall
suppress. Prehistoric: dino B+Y; Slash grounded hybrid. Neon: fight
`y≥140` only. Starbase: hover B+Y; form-2 demutation still needs play
solution for Clean iframe removal.
