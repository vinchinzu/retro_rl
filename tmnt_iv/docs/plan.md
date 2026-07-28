# Plan — TMNT IV: Turtles in Time

Ladder #3 (tier 1). See
`snes_oneshot/docs/GAME_SELECTION_NOTES.md` for program context.

**Clean lessons (do not relearn):** [`CLEAN_PLAYBOOK.md`](CLEAN_PLAYBOOK.md).

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
   (`scripts/run_stage*_segment.py`).
4. **Clean proof per stage** (heal=none, multi-entry) before removing
   assists — copy `scripts/probe_stage1_clean.py` pattern.
5. Continuous validation:
   `uv run python -m tmnt_iv.scripts.record_full_hard_run`.

## Milestones (segment clear — done)

Full stage chain power-on → hard credits under low-assist is **done**
(see `STATUS.md` / `BASELINE_METRICS.md`). Historical segment work is
complete; remaining work is **Clean** (zero assists).

## Path to Bronze / Clean (whole game)

Emergency HP (≤16→80) and form-2 iframe hold are the only production
assists. Clean means both at **0**, pizza + play only.

**Stage 1 Clean is finished.** Do not re-tune Stage 1 hazard jump-dodge
or reopen global pizza seek. Roll forward:

| Pri | Stage | Action | Exit criteria |
|-----|-------|--------|---------------|
| 1 | Alleycat (1) | Early/mid wave survival after 1 pizza; Metalhead already Clean; `probe_stage2_clean --suite` | Multi-entry 0 e-heals, 0 lives lost |
| 2 | Sewer (2) | LiveHard entry; residual 0x1C spikes; Rat King finish; `probe_stage3_clean --suite` | Multi-entry 0 e-heals, 0 lives lost |
| 3 | Technodrome (3) | Duo flank + tank; damage bucket #1 continuous | Probe + continuous entry |
| 4 | Prehistoric (4) | Slash spin **52** production; no blind spin-40 port | Probe + continuous entry |
| 5 | Skull & Crossbones (5) | Duo; **never** global pizza seek | Probe + continuous entry |
| 6 | Wounded Knee (6) | Raph cadence; stack jump-slash | Probe + continuous entry |
| 7 | Neon (7) | Near-band Mode-7; Krang | Probe + continuous entry |
| 8 | Starbase / form-2 (8–9) | Hover jump-slash; form-2 dodge **without** iframe write | form-2 iframe frames → 0 |
| 9 | Full continuous | One power-on hard dry-run, heal=none, iframe=0 | Bronze / Clean publish |

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
