# TAS BSNES Oracle — Super Metroid (long path)

**Status:** Phase 1 partial (2026-08-08) — intro-aware; authoring libsnes SEGV in PPU; v115 survives intro but desyncs  
**Priority:** replace snes9x open-loop thrash with **native-core truth extraction**  
**Playbook sibling:** [`TAS_ADAPT.md`](TAS_ADAPT.md) (snes9x harness path; still valid for product re-anchor)  
**Oracle env:** [`tas/ref/ORACLE_ENV.md`](../tas/ref/ORACLE_ENV.md)  
**Beads epic:** `rr-0lz6` (`bd show rr-0lz6`)

| Phase | Bead | Title |
|------:|------|-------|
| 0 | `rr-al8r` | Freeze thrash policy (CLI warn + docs) |
| 1 | `rr-wbsr` | Verify BizHawk plays `sniq_100p.bk2` (BSNES Compatibility) |
| 2 | `rr-l8zj` | Lua dump → `recordings/tas_oracle/sniq_100_bsnes` (blocked by Phase 1) |
| 3 | `rr-6sbs` | lsnes any% → `tas_oracle/sniq_any_lsnes` |
| 4 | `rr-gwd0` | Harness consumers prefer oracle boards (blocked by Phase 2) |
| 5 | `rr-rzxk` | Controlled short splices from oracle windows |
| 6 | `rr-81gp` | Stretch: same-core harness spike (optional) |

## Decision

We take the **best long path**, not more open-loop grid search under stable-retro.

| Rejected (easy / thrash) | Chosen (long / correct) |
|--------------------------|-------------------------|
| Power-on full BK2/LSMV under snes9x and “annotate harder” | Play movies on the **authoring core** (BizHawk BSNES / lsnes) |
| Wider `movie_start` search for Climb under snes9x | Dump **room/item/pose truth** from a syncing playback |
| Pretend thrash hops are tech encyclopedia | Boards only from **synced native runs** |
| Hope parser/LogKey fixes full sync | Inputs already correct — **core timing** is the gap |

**Product pure-first is unchanged.** Continuous tip stays snes9x product pure.  
TAS oracle feeds **route knowledge, door tech geometry, item order, and optional short splice candidates** — never STATUS from movie indices alone, never auto-wire continuous tip.

## Why open-loop harness fails (facts)

| Movie | Authoring core | Harness core | Result under harness |
|-------|----------------|--------------|----------------------|
| `tas/ref/sniq_100p.bk2` | BizHawk **BSNES Compatibility** (`Header.txt` Core=BSNES; SyncSettings Profile=Compatibility; Author=TASConverter) | stable-retro **snes9x_libretro only** | 222 789f; first_control ~11 183; **Ceres-only thrash**; items `0x0000` |
| `tas/ref/sniq_any_3653M.lsmv` | **lsnes** | snes9x | Same desync class after early Ceres |
| Landing splice @15000 | any% movie body | product Landing pin + snes9x | **Parlor once**, then Landing↔Parlor thrash; **no Climb** |

stable-retro hardwires SNES → snes9x (`cores/snes9x_libretro.so`). There is no switch-to-bsnes flag.  
Button parse (`tas/bk2.py`, `tas/lsmv.py`) and raw SNES-12 step (no L+R sanitize) are **not** the bottleneck.

## Goal

Build a **native-core oracle pipeline** that produces harness-shaped artifacts so agents never re-learn Super Metroid geometry from desynced thrash:

```text
┌─────────────────────┐     Lua / export      ┌──────────────────────────┐
│ BizHawk BSNES       │ ───────────────────► │ recordings/tas_oracle/   │
│  (+ lsnes any%)     │  pins, timeline,     │  sniq_100_bsnes/         │
│  syncing full movie │  windows, shots      │  sniq_any_lsnes/         │
└─────────────────────┘                      └────────────┬─────────────┘
                                                          │
                     extract_hops / stages / TRACK_100     ▼
                                              ┌──────────────────────────┐
                                              │ Truth board (usable=real)│
                                              │ room order, item order,  │
                                              │ door pins, tech tags     │
                                              └────────────┬─────────────┘
                                                          │
                     pure-first product (snes9x)           ▼
                                              ┌──────────────────────────┐
                                              │ Continuous tip           │
                                              │ optional short splice    │
                                              │ after re-anchor only     │
                                              └──────────────────────────┘
```

**Out of scope for v1 STATUS:** bit-exact continuous tip under snes9x of full 100%.  
**Stretch (separate epic):** same-core harness (bsnes in stable-retro) — only if program wants open-loop TAS tip; not required for oracle value.

## Hard rules (carry from TAS_ADAPT)

1. Never sanitize L+R on TAS bodies.  
2. Never STATUS-claim from movie frame indices alone.  
3. Assist off during TAS / oracle dumps.  
4. Product pure-first before graph edge / continuous promote.  
5. Oracle dumps are **reference**; product owns multi-room continuity.  
6. Do not re-open “full movie power-on under snes9x will work if we try harder.”

## Phases

### Phase 0 — Freeze thrash policy (docs + tooling defaults)

- [x] Document desync class (`TAS_ADAPT.md`, `sniq_100_full/desync_map.json`).  
- [x] Gate `extract_hops` post-desync thrash (`usable` cut; Zebes-first).  
- [x] Landing→Parlor materialize only short body (`tas/bodies/landing_to_parlor.json`).  
- [ ] Mark snes9x full-movie runs as **research-only / non-oracle** in CLI help and `TAS_ADAPT` pointer to this doc.  
- [ ] Deprioritize beads that assume thrash boards are skill sources (`rr-d7mq` retarget to oracle tech after Phase 2).

### Phase 1 — BizHawk environment + verify 100% BK2 sync

**Done when:** full `sniq_100p.bk2` completes under BizHawk BSNES Compatibility with items/rooms progressing off Ceres (ending or agreed late milestone).

Tasks:

1. [x] Install/confirm BizHawk on Linux (`~/.local/bin/bizhawk` → **2.11** Mono).  
2. [x] ROM SHA1 `DA957F0D63D14CB441D215462904C4FA8519C613` (verified).  
3. [x] Core: movie **BSNES** / `libsnes.wbx` Compatibility (authoring).  
4. [~] Play from power-on — **intro is ~3–5 min (~8–12k frames to first elev)**. Authoring **libsnes SIGSEGV** in `PPU::render_line` during intro. BSNESv115+ retarget **survives intro** (first elev ~8500, Ridley ~37k) but **desyncs** (not oracle). Tooling: `tas/oracle/{run_verify_100.sh,long_count.lua}`.  
5. [x] `tas/ref/ORACLE_ENV.md` (intro table + core split + coredump stack).

**Failure modes:** wrong ROM hash, wrong profile, missing firmware — fix env, do not fall back to snes9x.  
**Active blocker:** `rr-3vfm` — libsnes PPU SEGV on Mono Linux; need Wine/Windows or fixed libsnes for authoring-core GREEN.

### Phase 2 — Lua / headless dump pipeline (100%)

**Done when:** one command (or short runbook) produces a directory consumable by `extract_hops` / future `tas.oracle_load` without emulator in CI.

Artifact root:

```text
recordings/tas_oracle/sniq_100_bsnes/
  meta.json           # core, movie sha, rom sha, bizhawk version, frames
  pins.json           # same event kinds as harness annotate where possible
  room_timeline.csv
  series.jsonl        # optional strided WRAM
  windows/            # optional per hop: start_frame, end_frame, buttons
  shots/              # room_enter / item_gain screenshots
  summary.json
```

Minimum pin fields (align with `probe_pin` / `TraceEvent` shape):

| Field | Source (BizHawk domains) |
|-------|---------------------------|
| `frame` | `emu.framecount()` |
| `kind` | room_enter / room_leave / item_gain / beam_gain / capacity_gain / control / death / ending |
| `room_id` | WRAM room pointer / id (match harness `ram.py`) |
| `pose`, `x`, `y`, subs | Samus pose / position |
| `collected_items`, `collected_beams` | equipment masks |
| `energy`, missiles, supers, PBs | HUD RAM |
| `speed_counter`, `shinespark_timer` | if cheap |
| `pin` | nested dict compatible with existing JSON |

Implementation options (pick one, document):

1. **BizHawk Lua** while movie plays (simplest; interactive or `--lua=` if available).  
2. **External headless** if/when EmuHawk CLI supports movie+lua reliably on Linux.  
3. Python driver only if it shells to BizHawk — do not reimplement BSNES in Python.

Repo layout suggestion:

```text
snes/super_metroid/tas/oracle/
  README.md
  bizhawk_dump_sm.lua      # in-emulator script
  run_bizhawk_100.sh       # launch + paths
  schema.md                # pin/event contract
  import_oracle.py         # normalize → pins.json / extract_run adapter
```

### Phase 3 — lsnes any% oracle (parallel track)

**Done when:** `sniq_any_3653M.lsmv` has the same artifact shape under `recordings/tas_oracle/sniq_any_lsnes/`.

1. Install lsnes (or document BizHawk import of any% if a verified conversion exists — prefer **native lsnes** for bit-exact).  
2. Same dump contract as Phase 2 (adapt memory domain names).  
3. Compare early Ceres pins to harness first_control (~11 182) for residual boot research only.

### Phase 4 — Harness consumers (truth boards)

**Done when:** offline tools prefer oracle dirs over thrash `sniq_100_full`.

1. `extract_hops` / `extract_run`: accept `tas_oracle/*` with `source=bsnes_oracle` / `lsnes_oracle`.  
2. All hops from oracle default `usable=true` unless death/desync markers in oracle itself.  
3. `stages.py` movie_start/body hints filled from **oracle windows**, still tagged `plan_only` / `materialized_unproven` until product re-anchor.  
4. `docs/routes/TRACK_100.md` / CSV: optional column “oracle frame / room” for checklist (docs only; not STATUS continuous).  
5. Map viewer / path overlays: prefer oracle Crateria+ path density over thrash (see `map_viewer/paths.py` resync names).  
6. Retarget pure-probe skill ports (`rr-d7mq`) to **oracle tech tags** at product pins — pure-first still required.

### Phase 5 — Controlled harness splice (optional, after truth)

Only after Phase 4:

1. For high-value hops (doors, shines, mockball): take oracle **button window** + product **control state**.  
2. Search pad/start **small** (± few frames), not full-movie power-on.  
3. Prove dual/GREEN under pure process before continuous fold.  
4. Keep Landing→Parlor as the reference hybrid success case.

### Phase 6 — Stretch: same-core harness (program decision)

**Not required for oracle value.** Only if leadership wants open-loop full TAS under Gym API:

1. Feasibility spike: libretro **bsnes** (or BizHawk-headless RPC) behind a separate env backend.  
2. Savestate format, action space, determinism, CI cost.  
3. Re-validate entire continuous SM suite (every pure seed may desync vs snes9x).  
4. New integration ID / tip track — never silently swap core under existing STATUS greens.

Until Phase 6 exists, **do not** promise “full 100% reproduction in stable-retro.”

## Success metrics

| Metric | Target |
|--------|--------|
| Oracle 100% unique rooms | Includes Zebes + progression rooms (not 6 Ceres only) |
| Oracle item_gains | Non-empty; morph/bombs/… match route |
| `extract_hops` on oracle | `usable_hops` ≈ real hops; thrash cut N/A |
| snes9x full BK2 power-on | Documented dead-end; CLI warns |
| Continuous tip | Still product pure; oracle does not auto-STATUS |
| Landing→Parlor | Still only proven hybrid movie splice under snes9x |

## Explicit non-goals

- STATUS-promoting TAS frame counts.  
- Replacing product morph spine with movie bodies.  
- Infinite Climb `movie_start` grids under snes9x.  
- Treating `sniq_100_full` thrash as skill training data.  
- Porting bsnes into stable-retro as a “quick install.”

## Commands (target; implement in phases)

```bash
# Phase 1 — verify (manual / scripted)
# Open BizHawk: ROM + Core BSNES Compatibility + play tas/ref/sniq_100p.bk2

# Phase 2 — dump (after scripts land)
./snes/super_metroid/tas/oracle/run_bizhawk_100.sh \
  --out snes/super_metroid/recordings/tas_oracle/sniq_100_bsnes

# Phase 4 — consume offline
uv run python -m super_metroid.tas.extract_hops \
  snes/super_metroid/recordings/tas_oracle/sniq_100_bsnes
```

## References

- BK2 format: https://tasvideos.org/Bizhawk/BK2Format  
- Sniq 100% userfile (vendored `tas/ref/sniq_100p.bk2`)  
- Sniq any% TASVideos #3653M (`tas/ref/sniq_any_3653M.lsmv`)  
- Harness thrash evidence: `recordings/tas_import/sniq_100_full/desync_map.json`  
- Climb search negative: `recordings/tas_import/resync_zebes_climb/align_search.json`  
- Existing hybrid success: `tas/bodies/landing_to_parlor.json`, `resync_zebes_rooms/`

## Session prompt

See [`docs/tasks/NEXT_SESSION_TAS_ORACLE.md`](tasks/NEXT_SESSION_TAS_ORACLE.md).
