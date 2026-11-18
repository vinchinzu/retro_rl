# ALTTP plan

Execution order assessed 2026-10-24. Verified maturity stays in
[STATUS.md](STATUS.md); C2 (Eastern prize from the real predecessor) is still
open. This is the single execution sequence. [PORT_PLAN.md](PORT_PLAN.md)
retains AP task details and checkboxes; [tasks/residual.md](tasks/residual.md)
retains the one live route handoff. The full-clear ORDER is a logic census,
not an executable campaign: only rows 1–4 are true, AP-105 is open, and
AP-197 credits acceptance is unimplemented.

## Assessment

- Sole working tree: `/home/v/01_projects/11_games/retro`, branch `main`,
  HEAD `6013e537`. Eleven old Harvest worktrees are preserved in
  `/home/v/01_projects/11_games/archive/retro-worktrees-20261024`, including
  unmerged edits and indexes. Do not apply older Harvest code over main.
- Grok has no active retro workflow at this audit. Its live process belongs
  to `openage/native`. Last ALttP completion: `alttp-b9-back2`.
- Latest development progress (2026-10-24 sitting): replayable B9 lower-channel
  crossing from `Play_eog9_7` to `(4888,5744)`, layer 1, HP 28 throughout.
  Forward replay: 396 frames, desync/loads/writes 0. Endpoint
  `Dev_B9Channel20261024`; evidence and limitations in `tasks/residual.md`.
  Room `0xA9` remains unentered. Pins 7–23 remain intact.
- Efficiency is poor: 26 completed B9 workflows, 98 reported agent launches,
  7.46 summed workflow-hours for this room. Whole-session usage reports
  1,307 model calls and 107,154,310 cumulative tokens, including 91,489,920
  cached input tokens. These are session totals, not B9-only or unique tokens.
  Workflow completion is not room completion.
- Before this sitting, residual, ORDER and PLAY contained about 171 KB of
  repeated frontier prose. The current sitting archived the raw evidence
  beside its recordings and replaced those repetitions with one residual.
  Do not add another handoff narrative.
- The renamed checkout broke worktree pointers and 49 local venv launcher,
  activation and import paths. These are repaired; archived copies preserve
  the old files. Core tests now pass after teaching the docs directory check
  that the beads directory was deliberately retired.
- Two ALttP offline tests remain red: the soldier census follows the old
  absolute scan directory from `scan/index.json`; the fake stairs-exit
  environment reports outdoors without matching the mapped destination.
  Keep those failures visible until their contracts are repaired.
- `bd ready` fails because there is no local database; the historical board
  is archived as documented in the repo. Do not initialize a replacement or
  claim that a bead was updated. AP-105 remains the existing route task.

Fresh checks on this checkout: core 552 passed; docs 8 passed; ALttP offline
395 passed, 2 failed, 27 deselected. Doctor, editor listing/usage, play
listing/help/rejection, adventure planning and matrix regeneration passed.
Real-ROM forward replays of both the 36,078-frame Sanctuary tape and the
38,277-frame Kakariko extended tape passed goal acceptance with zero desync,
loads or RAM writes and `clean_chain=true`. The failed fresh-controller tape
also replayed exactly, but correctly failed goal acceptance. Thus accepted
prefix playback works; fresh controller regeneration still needs repair.

## Serial work

1. **Protect and name the accepted prefix.** Preserve accepted tapes/reports
   before controller work. Use explicit tape paths and record their hashes;
   never infer a tape from the report basename. The successful Kakariko
   recordings are `recordings/clean_kakariko_20261006/extended.tape.npz` and
   `prefix_controller.tape.npz` (38,277 frames). `controller.tape.npz` in that
   directory is a failed 14,069-frame fresh-controller attempt, consistent
   with the known `rr-ihr8` rescue regression; it is not the accepted tape.
   Sanctuary C1 is `recordings/clean_escape_20261002/controller.tape.npz`
   (36,078 frames). Keep STATUS unchanged until its owning acceptance task.
2. **Close the health gaps.** Repair the moved-path scan lookup without
   inventing sprite evidence; repair the fake exit's destination contract
   without weakening real acceptance. Require the existing ALttP tests and
   the core/docs gate to pass. Audit the archived Harvest deltas by intent
   in a separate sitting before declaring all multi-game code integrated.
3. **Finish B9 from the reproducible lower crossing.** The development tactic
   `alttp.dungeon.nav:reach_cannonball_hall_clear` now crosses from
   `Play_eog9_7` in 396 frames with a synchronized, no-damage forward replay.
   Next prove the upper-floor exit into `0xA9`, then replay the whole room
   from that predecessor. Use the endpoint and constraints in the living
   residual. The exact predecessor-controller forecast count was lost in
   a replay-harness crash; it is unknown. No AP-105 closure or continuous
   promotion follows from this partial-room result.
4. **Compose the continuous Eastern prefix.** Diagnose the existing
   `eog9_spine_20261023` acceptance failure (`inventory_preserved=false`,
   outdoor blank keys 255 versus dungeon keys 0) against the actual counter
   semantics. Do not simply waive preservation. Replay from blank SRAM with
   the verified prefix plus the earned B9 policy. Require goal acceptance,
   every-frame synchronization and disclosed loads/writes/assists.
5. **Continue the one campaign spine.** Earn Eastern's required items,
   Armos/prize and C2 from the real predecessor before promoting that gate.
   Then follow the dependency order in PORT_PLAN through the remaining
   pendants, Master Sword, Agahnim, crystals, Ganon and AP-197 credits.
   Isolated Whatif dungeon pins and census reachability cannot advance it.

## Grok run contract

Use one route executor per bounded attempt. Add a reviewer only when an
actual ambiguity needs review; do not repeat act/record/skeptic cycles for
every eight pixels. Put the movement loop and RAM assertions in the
controller, and batch collision-safe steps within one attempt. Start with
a 15-minute sitting cap and at most two attempts at the same hypothesis.
If neither adds a reproducible policy or proves a new transition, stop and
report the blocker rather than launching another variant.

Each attempt should leave one compact result: source/pin and hashes,
committed game frames, endpoint room/xy/layer/HP, first prediction miss,
new transition or reproducible policy, replay verdict, model-call/token
usage and the next blocker. Compare structured integer fields, not wording.
Archive detailed act logs with game recordings and retain only the current
frontier, invariants and withdrawals in the living residual. This sitting consolidated the repeated ORDER/PLAY frontier into references
to that residual, preserving raw evidence before trimming prose.

Progress is a verified transition, a replayable segment, or a closed maturity
gate. A new pin is useful development evidence; count it separately.
Do not resume unattended full-game spending from this assessment.

`.grok/skills/alttp-session/SKILL.md` is the process for that cap. The
`alttp-*.rhai` scripts were removed from `.grok/workflows/` and now sit in
`snes/alttp/scratch/retired-workflows/`. They are not registered. A finished
script does not start the next one. Spending stays off until the user names
the sitting.

Audit artifacts: `recordings/health_20261024/` for ROM replay reports and
`.cursor/skills/verify-retro-rl/artifacts/20261024-consolidation/` at repo root
for CLI/test logs and the workflow/usage summary. The original user handoff
stays at `/tmp/handoff-alttp-b9-what-is-done.md`.
