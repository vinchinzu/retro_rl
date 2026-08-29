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
