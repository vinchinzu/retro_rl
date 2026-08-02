# Agent Instructions — Super Metroid

Super Metroid scripted full-clear project. Shared process:
[`snes_oneshot/docs/FULL_RUN_PROCESS.md`](../snes_oneshot/docs/FULL_RUN_PROCESS.md).

## Evaluation contract

- Target: one continuous power-on-to-ending run.
- Allowed assists (primary path): unlimited health (in-game energy) and
  unlimited ammo, exactly as defined in
  [`docs/ASSIST_CONTRACT.md`](docs/ASSIST_CONTRACT.md).
- Resource assists remove attrition only. They must not grant uncollected ammo
  types, capacity, equipment, items, movement abilities, door state, map
  state, boss/event flags, rooms, or completion.
- Record every assist write in the full-run manifest.
- Completion requires the natural endgame escape and ending/credits evidence;
  defeating the final boss alone is not a clear.
- **Parallel Clean track:** no energy + no ammo writes (Bronze/Clean). Target
  tip Bomb Torizo. Contract: [`docs/CLEAN_TRACK.md`](docs/CLEAN_TRACK.md).
  Defaults stay assisted; clean artifacts use `*_clean` stems only — never
  overwrite assisted `recordings/start_to_*.json`.

## Layout

| Path | Role |
|------|------|
| `ram.py`, `assist.py`, `policy.py`, `progression.py`, `paths.py`, `room_timer.py` | Core package surface |
| `routes/continuous.py` | Power-on chain (… → Varia → Business return → Frog Save) |
| `routes/segment.py` | Segment / HopExecutor / ContinuousSession contracts |
| `routes/runtime.py` | Shared session, report harness, integrity |
| `routes/kpdr/` | Pure movement/combat controllers (no env ownership) |
| `rooms/` | Full-room graph, problem catalog, practice loop |
| `legacy/` | Frozen vision BC / model registry (do not import into continuous) |
| `dev/` | Door-warp / boss probes (not continuous evidence) |
| `scripts/record/` `verify/` `probe/` `export/` `room/` | CLI entry points |
| `docs/` | `STATUS.md`, `plan.md`, `ARCHITECTURE.md`, `ASSIST_CONTRACT.md` |
| `docs/routes/` | Accepted / working route boards |
| `docs/research/` | Path board, room catalog, legacy notes |
| `policies/room_clears/` | Curated room policies (tracked) |
| `maps/` | Generated graphs (gitignored except README) |
| `recordings/` | Baseline videos/manifests (gitignored) |
| `debug/` | Probe screenshots (gitignored) |
| `custom_integrations/SuperMetroid-Snes/` | Emulator integration + **anchor** states |
| `…/scratch/` | Ephemeral probe save-states |

## Immediate goal

**Primary: play KPDR continuous spine** (controller/policy). Door-warps
are topology diagnostics only — not route evidence.

1. **KPDR board:** [`docs/routes/ROUTE_KPDR.md`](docs/routes/ROUTE_KPDR.md)
   — authoritative continuous order (K→P→D→R). Spore Supers (no mockball);
   Alpha PB after Ice; **not** ship-first / early Pink PB.
2. **Path board:** [`docs/research/PATH_ROOM_BOARD.md`](docs/research/PATH_ROOM_BOARD.md)
   — 107 rooms / 199 hops; hop table is topology, not human KPDR order.
3. **Verified continuous tip:** power-on → Frog Savestation
   (`scripts/record/continuous.py --to frog`; two matching integrity-green
   runs, **114,923f**, 0 loads / progression / capacity writes / deaths).
   Prefixes: Hi-Jump **87,696f**, Varia **104,382f**, Business **113,723f**.
4. **★ Next play:** first Bubble via **Cathedral climb** pure stack from
   `scratch/post_business_continuous.state` (`SM-K4-CATH-01`…). Frog Save is a
   continuous K4.0 milestone; Frog Speedway is **post-Speed only** (Boost
   Blocks). Then Bubble → Speed → Wave → Ice → Alpha PB; no door-warp
   evidence or progression writes.
5. **Architecture + structure plan:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
   (layers, Segment contracts, tip recipe, **known structural debt**) and
   [`docs/plan.md`](docs/plan.md) (M6–M8 + Structure & API todos). Planner-serial
   arch cards: [`docs/tasks/QUEUE.md`](docs/tasks/QUEUE.md) (`SM-ARCH-*`).
6. **Dev topology (green):** `kpdr.py route-to-hijump` — 24 hops Big Pink →
   Hi-Jump room; anchors `dev_kpdr_*` / `dev_hijump_*`.
7. **Parked:** pure Pink PB maze; ship-first skip (not KPDR); vision BC in
   `legacy/` until gold.
8. Boss fights only after natural entry exists on the continuous chain.
   Pipeline: [`docs/BOSS_PIPELINE.md`](docs/BOSS_PIPELINE.md) — Phantoon next
   after Alpha PB / ship access.
9. **Parallel Clean track:** privilege reduction (no health/ammo assists) on
   the early continuous prefix → **Bomb Torizo**. Infra first
   (`SM-CLEAN-ARTIFACTS` / `CLI` / `INTEGRITY`), then morph, then bombs tip.
   Board: [`docs/CLEAN_TRACK.md`](docs/CLEAN_TRACK.md) · milestones Clean
   section in [`docs/routes/MILESTONES.md`](docs/routes/MILESTONES.md).

**Top-level milestone board** (status marks for every tip / practice rollup):
[`docs/routes/MILESTONES.md`](docs/routes/MILESTONES.md) ·
[`docs/routes/MILESTONES.csv`](docs/routes/MILESTONES.csv).

**Full backlog** (~288 tickets to M8 credits):
[`docs/routes/BACKLOG.csv`](docs/routes/BACKLOG.csv) ·
[`docs/routes/BACKLOG.md`](docs/routes/BACKLOG.md).

Tracker (chartable CSV/JSON/MD):
[`docs/routes/KPDR_TRACKER.csv`](docs/routes/KPDR_TRACKER.csv) · export
`scripts/export/kpdr_tracker.py`.

Status: [`docs/STATUS.md`](docs/STATUS.md). Plan: [`docs/plan.md`](docs/plan.md).
KPDR board: [`docs/routes/ROUTE_KPDR.md`](docs/routes/ROUTE_KPDR.md).
Boss pipeline: [`docs/BOSS_PIPELINE.md`](docs/BOSS_PIPELINE.md).
Docs index: [`docs/README.md`](docs/README.md).

### Cheap executor (OpenCode)

Farm **atomic** implementation to a cheap executor; keep integrity / STATUS /
natural-entry judgment on a strong planner (Grok) or human.

- Process: [`docs/tasks/PROCESS.md`](docs/tasks/PROCESS.md) — pure-first,
  stabilize waves, residual schema, metrics, dual-track
- Template: [`docs/TASK_TEMPLATE.md`](docs/TASK_TEMPLATE.md)
- Queue: [`docs/tasks/QUEUE.md`](docs/tasks/QUEUE.md)
- Wave dispatch: [`docs/tasks/WAVE-11.md`](docs/tasks/WAVE-11.md)
- Triage: [`docs/tasks/TRIAGE.md`](docs/tasks/TRIAGE.md)
- Source states: [`docs/SOURCE_STATES.md`](docs/SOURCE_STATES.md)
- Cards: [`docs/tasks/`](docs/tasks/) (`SM-*.md`)
- Dispatch: `./super_metroid/scripts/dispatch_opencode.sh SM-K4-03`
  (ownership conflict check on parallel hot modules)
- Session logs: `docs/tasks/logs/` (**gitignored** — do not force-add)
- Model IDs / provider routing: `scripts/dispatch_opencode.sh` (env-overridable)
  and local `opencode.json` (copy from `opencode.example.json`; gitignored).
  Auth stays outside the repo.

```bash
# From repo root (Flash auto-picked for docs/report/rollup cards)
./super_metroid/scripts/dispatch_opencode.sh SM-K4-03
./super_metroid/scripts/dispatch_opencode.sh SM-K4-03 SM-K4-04 SM-K4-05  # parallel if disjoint files
./super_metroid/scripts/dispatch_opencode.sh --flash SM-ROLLUP-STATUS
./super_metroid/scripts/dispatch_opencode.sh --foreground SM-K4-06
# Luna + max thinking (default for dispatch):
./super_metroid/scripts/dispatch_opencode.sh --luna --variant max SM-ROOM-SEG-01

# Dual-track room farm: 8-wide × N rounds (continuous tip parked)
./super_metroid/scripts/farm_room_waves.sh --rounds 10 --parallel 8
./super_metroid/scripts/farm_room_waves.sh --rounds 20 --parallel 8 --deadline-hours 2
```

Do **not** hand the executor open-ended “next continuous tip” work. Cards must
list exact files, recipe step, acceptance commands, and (for pure probes) the
**exact source state path + expected room id** (prefer `SOURCE_STATES.md`).

**Role guide:** Flash = tracker/docs/dwell report + STATUS **proposals**
(`SM-ROLLUP-STATUS`); Luna = tests + controller scaffold + primitives +
bounded geometry with a named source state. Planner only for STATUS apply,
continuous compose/re-record, and natural-entry design.

**Hard gates:** pure-green from continuous-like source before continuous;
stabilize wave after implement knobs; one-knob geometry; residual → next card
ID + one change; never parallel-edit `business_climb` / HJ return / spore /
`varia_return` geometry / continuous / STATUS.

**Post-Varia reverse source states (scratch)** — full index in
[`docs/SOURCE_STATES.md`](docs/SOURCE_STATES.md):

| State | Room | Use for |
|-------|------|---------|
| `scratch/post_varia_collected.state` | 0xA6E2 Varia | pure `varia-to-kraid` |
| `scratch/post_varia_to_kraid_pure.state` | 0xA59F Kraid | pure `kraid-to-eye-return` |
| `scratch/continuous_like_business_climb_entry.state` | Business floor | pure `business-to-warehouse` |

## Commands

### Continuous baselines

One CLI for every tip — play lives in `routes/continuous.py`; register tips in
`routes/catalog.py` (do **not** add `start_to_*.py` scripts).

**Post-Supers tip extension (room-by-room):** pure controller in `routes/kpdr/`
→ graph edge in `progression.py` → splits/`ContinuousTip` in `catalog.py` →
append `RouteHop`s and a thin `run_post_supers_tip(...)` wrapper in
`continuous.py`. Do not copy another full `run_start_to_*` body.

```bash
# Verified continuous tip: power-on → Frog Savestation (KPDR K4.0)
uv run python super_metroid/scripts/record/continuous.py --no-video
uv run python super_metroid/scripts/record/continuous.py --to frog
# Showcase video: shared VideoRecorder (audio + button footer + quality).
# Play always power-on; --video-start only trims the MP4 (default: zebes).
uv run python super_metroid/scripts/record/continuous.py --to frog \
  --video-start zebes --hq
uv run python super_metroid/scripts/record/continuous.py --to frog \
  --video-start after_credits --video-start-frame 900 \
  --fps 60 --scale 2 --crf 17 --preset medium
uv run python super_metroid/scripts/record/continuous.py --to varia
# Save a source only if that tip run itself passes all integrity checks.
uv run python super_metroid/scripts/record/continuous.py --to varia --no-video \
  --state-output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_continuous.state
uv run python super_metroid/scripts/record/continuous.py --to business --no-video \
  --state-output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_continuous.state
uv run python super_metroid/scripts/record/continuous.py --to frog --no-video \
  --state-output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_continuous.state
uv run python super_metroid/scripts/record/continuous.py --to kraid
uv run python super_metroid/scripts/record/continuous.py --to hijump --no-video
# Prefix milestones (shorter checks)
uv run python super_metroid/scripts/record/continuous.py --to warehouse --no-video
uv run python super_metroid/scripts/record/continuous.py --to below_spazer --no-video
uv run python super_metroid/scripts/record/continuous.py --to bat --no-video
uv run python super_metroid/scripts/record/continuous.py --to red_tower --no-video
uv run python super_metroid/scripts/record/continuous.py --to supers --no-video
uv run python super_metroid/scripts/record/continuous.py --to spore --no-video
uv run python super_metroid/scripts/record/continuous.py --to bombs --no-video
uv run python super_metroid/scripts/record/continuous.py --to morph --no-video
uv run python super_metroid/scripts/record/continuous.py --list
```

Video stack: ``retro_harness.video.VideoRecorder`` (shared). Metroid start
gates / cutoffs: ``super_metroid.video.continuous_video_config``.

### Path board + post-Super controller

```bash
uv run python super_metroid/scripts/export/path_room_board.py

# From natural_post_spore_spawn: Supers → Big Pink main, etc.
uv run python super_metroid/scripts/probe/post_spore_pb.py --to main
uv run python super_metroid/scripts/probe/post_spore_pb.py --to crest
uv run python super_metroid/scripts/probe/post_spore_pb.py --to tunnel-floor

# KPDR K1: Big Pink main (controller)
uv run python super_metroid/scripts/probe/post_spore_pb.py --to main

# Historical Kraid-before-Hi-Jump topology (dev door-warps; 24 hops)
uv run python super_metroid/scripts/probe/kpdr.py route-to-hijump --grant-hijump
uv run python super_metroid/scripts/probe/kpdr.py varia-to-hijump
uv run python super_metroid/scripts/probe/kpdr.py list

# Pure room controllers (no warp/write inside each segment)
uv run python super_metroid/scripts/probe/kpdr.py pure ghz-to-noob \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_kpdr_ghz.state
uv run python super_metroid/scripts/probe/kpdr.py pure noob-to-red \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_kpdr_noob.state
uv run python super_metroid/scripts/probe/kpdr.py pure warehouse-hijump-kraid \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/red_to_warehouse_controller.state
# First post-Varia door (controller_dev; natural post-collect source)
uv run python super_metroid/scripts/probe/kpdr.py pure varia-to-kraid \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_collected.state

# Chartable progress tracker
uv run python super_metroid/scripts/export/kpdr_tracker.py

# Offline high-dwell ranks from a continuous report (no emu re-run)
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_varia.json --top 15
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_varia.json --reasons --top 20

# Source catalog + pure RED pin (nav-mode RAM; no full-bank copy per frame)
uv run python super_metroid/scripts/probe/kpdr.py suggest-source \
  --room 0xA6E2 --segment varia-to-kraid
uv run python super_metroid/scripts/probe/kpdr.py pure varia-to-kraid \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_collected.state \
  --pin-json super_metroid/debug/varia_to_kraid_pin.json

# Scaffold pure tip hop (dry-run checklist; --write emits controller + card)
uv run python super_metroid/scripts/scaffold_tip.py \
  --segment business_to_frog_save --from-room 0xA7DE --to-room 0xB167 \
  --module k4_norfair --card-id SM-K4-BUBBLE-01 --dry-run
```

Controllers: `routes/kpdr/` (Super collect → Kraid; `post_spore_controller`
is a thin re-export). KPDR plan: `docs/routes/ROUTE_KPDR.md`. Tracker:
`docs/routes/KPDR_TRACKER.csv`.

### Topology skeleton (door-warp diagnostics only)

```bash
uv run python super_metroid/scripts/probe/route.py list
uv run python super_metroid/scripts/probe/route.py full
uv run python super_metroid/scripts/probe/route.py full-hybrid
uv run python super_metroid/scripts/probe/route.py phantoon-to-ridley
uv run python super_metroid/scripts/probe/route.py ridley-to-mb
uv run python super_metroid/scripts/probe/route.py late-full
```

Reports are labeled `developmentOnly: true`. Hop tables:
`maps/full_route_hops.json`, `maps/late_game_route_hops.json`. Runner:
`dev/route_dev.py`.

### Boss / late entry (development only)

Full-knowledge boss strategy (RAM hitboxes; vision BC parked until gold):
[`docs/BOSS_PIPELINE.md`](docs/BOSS_PIPELINE.md) ·
[`docs/research/STRUCTURED_BOSS_RL.md`](docs/research/STRUCTURED_BOSS_RL.md).

```bash
# Bomb Torizo structured strategy / natural capture / feature-vector RL
# (not continuous evidence; keep hash-pinned pit_to_post_torizo on acceptance)
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py --state BossTorizo
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py prove-natural
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py eval --episodes 1
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py train --timesteps 4096
# Kraid Super-spray + Varia closeout from room entry (no RL; not continuous evidence)
uv run python super_metroid/scripts/probe/kraid_combat.py strategy --state entry
uv run python super_metroid/scripts/probe/kraid_combat.py varia --state entry \
  --report super_metroid/debug/kraid_varia_run.json
uv run python super_metroid/scripts/probe/kraid_combat.py strategy --state dev_kpdr_kraid_entry
uv run python -c "from super_metroid.dev.kraid_dev import run_kraid_fight; print(run_kraid_fight())"
uv run python super_metroid/scripts/probe/phantoon.py collect-pb
uv run python super_metroid/scripts/probe/phantoon.py capture-entry
uv run python super_metroid/scripts/probe/phantoon.py ship-route
uv run python super_metroid/scripts/probe/mother_brain.py capture-mb
uv run python super_metroid/scripts/probe/mother_brain.py spray-mb --frames 3600
uv run python super_metroid/scripts/probe/mother_brain.py run-escape --frames 12000
```

### Room practice (easiest-first doorway segments)

Entry fixtures are **doorway-natural**: door-warp through the catalog entry
door, settle **just inside** (not mid-room). That keeps segments on real door
boundaries so RNG can be re-rolled later by re-entering the same door.

```bash
uv run python super_metroid/scripts/export/room_problems.py
# Ranked board: 262 problems easiest→hardest + % complete
uv run python super_metroid/scripts/export/room_work_queue.py
uv run python super_metroid/scripts/room/run_problem.py queue --limit 20
# Doorway entry states (default boot: natural_post_spore_spawn)
uv run python super_metroid/scripts/room/run_problem.py bootstrap --queue 1 --max 10
uv run python super_metroid/scripts/room/run_problem.py scaffold PROBLEM_ID
uv run python super_metroid/scripts/room/run_problem.py teleport PROBLEM_ID
# Green run + sha-gated promote (policy → verified_development_state)
uv run python super_metroid/scripts/room/run_problem.py run PROBLEM_ID --promote
uv run python super_metroid/scripts/room/run_problem.py ready --run
```

Board: `docs/routes/ROOM_WORK_QUEUE.md` · CSV/JSON under `docs/routes/` +
`maps/room_work_queue.json`. Units: 262 room problems (practice); 583 directed
edges are topology only; KPDR spine remains `docs/routes/KPDR_TRACKER.csv`.
Avoid late full-loadout boots for bootstrap — they can freeze input after warp.
Entry door pointers live on the catalog graph (`PhysicalEndpoint.door_ptr` /
`entry.doorPtr`); shared `EntryContract` covers bootstrap, scaffold, and run.

### Room timing (stock ROM / emulator frames)

```bash
uv run python super_metroid/scripts/probe/room_timer.py self-check
uv run python super_metroid/scripts/probe/room_timer.py offline -i samples.json
# docs: docs/ROOM_TIMER.md  ·  core: room_timer.py
```

## Dev traps

- `dev/common.py` owns `boot_from_state`, `door_warp`, `place_samus`,
  `apply_dev_survivability`, `enemy_hps`, `select_weapon`, `save_dev_state`.
  Reuse them; do not re-implement warp settle.
- Door-warp settle must wait for **game state 8** (not merely `ordinary`
  phase) — multi-screen loads sit in state 11 for 50–100+ frames.
- High WRAM (events/boss bits at `$7E:D820+`) must use `read_bank7e_wram` /
  `write_wram_u8` — raw `env.get_ram()[0xD820]` is open-bus garbage.
- Save **named anchors** under `custom_integrations/SuperMetroid-Snes/`;
  dump probe noise into `scratch/`.
- Prefer room/door/inventory progress vectors over coordinate-only watchdogs.
- Keep the last successful full-run baseline; candidates use separate reports.
- Do not route an infinite bomb jump for the current KPDR suffix. The Hi-Jump
  return uses the intended left-shaft ledges plus ordinary bombs in the top
  morph tunnel; Charge needs a conventional return.
