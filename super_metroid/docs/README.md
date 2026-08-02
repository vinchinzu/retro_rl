# Super Metroid docs

**Goal:** continuous power-on → ending/credits with unlimited energy + ammo
only ([ASSIST_CONTRACT.md](ASSIST_CONTRACT.md)). Current tip: **Frog
Savestation** (K4.0). Next: Frog Save → Speedway pure.

## Top-level status board

| Mark | Meaning |
|------|---------|
| ✅ | Continuous integrity green |
| ▶ | Next pure hop ready |
| 🟨 | Partial / in progress |
| ⬜ | Open |
| ⏸ | Parked (not KPDR) |

Full table: **[routes/MILESTONES.md](routes/MILESTONES.md)** · CSV
[routes/MILESTONES.csv](routes/MILESTONES.csv).

| | Milestone | Status | Frames / score |
|--:|-----------|--------|----------------|
| ✅ | Power-on → Morph … Supers (K0) | continuous | prefix |
| ✅ | → Red Tower (K1) | continuous | 80,445 |
| ✅ | → Hi-Jump / Kraid entry (K2) | continuous | 87,696 / 97,170 |
| ✅ | → Varia (K3) | continuous | **101,954** best |
| ✅ | → Business return | continuous | 113,723 ×2 |
| ✅ | → **Frog Savestation (K4.0)** | continuous | **114,923** ×2 |
| ▶ | Frog Save → Speedway pure | pure_open | — |
| ⬜ | → Speed / Wave / Ice (K4) | open | — |
| ⬜ | → Alpha PB (K5) | open | — |
| ⬜ | → Phantoon / Gravity (K6) | open | — |
| ⬜ | → Botwoon / Draygon / SJ (K7) | open | — |
| ⬜ | → Ridley (K8) | open | — |
| ⬜ | → MB + Escape + **Credits (M8)** | open | — |
| 🟨 | Room practice easy+standard | partial | 62/108 ready |
| 🟨 | Path rooms continuous | partial | 39/107 |

## Doc map

| File / dir | Role |
|------------|------|
| **[STATUS.md](STATUS.md)** | Verified facts + maturity gate (M5: → Frog Save) |
| **[plan.md](plan.md)** | Strategy, M6–M8 roadmap, structure/API plan |
| **[routes/MILESTONES.md](routes/MILESTONES.md)** | **Top-level milestone status** (every tip / rollup) |
| **[routes/BACKLOG.csv](routes/BACKLOG.csv)** | **~290 atomic tickets** to full clear |
| [routes/BACKLOG.md](routes/BACKLOG.md) | Backlog epic summary |
| [routes/KPDR_TRACKER.csv](routes/KPDR_TRACKER.csv) | Per-segment KPDR spine (chartable) |
| [routes/ROUTE_KPDR.md](routes/ROUTE_KPDR.md) | Authoritative continuous any% KPDR plan |
| [routes/ROOM_WORK_QUEUE.md](routes/ROOM_WORK_QUEUE.md) | Dual-track practice board (262 problems) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layers, Segment contracts, tip recipe, debt |
| [BOSS_PIPELINE.md](BOSS_PIPELINE.md) | Boss catalog → strategy → continuous order |
| [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md) | Allowed resource assists |
| [SOURCE_STATES.md](SOURCE_STATES.md) | Continuous-like pure entry states |
| [TASK_TEMPLATE.md](TASK_TEMPLATE.md) | OpenCode card format |
| [tasks/QUEUE.md](tasks/QUEUE.md) | Live wave board + ★ tip |
| [tasks/PROCESS.md](tasks/PROCESS.md) | Pure-first / stabilize / residual rules |
| [tasks/archive/](tasks/archive/) | Completed cards, residuals, farm one-shots |
| [archive/](archive/) | Superseded route notes |
| [research/](research/) | Path board, room catalog, boss RL, legacy |
| [ram_map.md](ram_map.md) | WRAM addresses |
| [ROOM_TIMER.md](ROOM_TIMER.md) | Stock room timing |

## Backlog depth

| Epic | Tickets (approx) | Focus |
|------|-----------------:|-------|
| K4 | 50 | Speedway → Bubble → Speed → Wave → Ice |
| K5 | 10 | Alpha Power Bombs |
| K6 | 31 | Moat → WS → Phantoon → Gravity |
| K7 | 41 | Maridia → Botwoon → Draygon → SJ |
| K8 | 25 | Lower Norfair → Ridley |
| K9 | 41 | G4 → Tourian → MB → Escape → Credits |
| PRACTICE | 34 | Dual-track room waves |
| ARCH / DOCS / BOSS-INFRA | ~25 | Structure + process |
| **Total** | **~288** | Target 200–300 working depth |

## Commands (quick)

```bash
# Current continuous tip
uv run python super_metroid/scripts/record/continuous.py --to frog --no-video

# ★ Next pure hop
uv run python super_metroid/scripts/probe/kpdr.py pure frog-save-to-speedway \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_continuous.state

# Tracker / board exports
uv run python super_metroid/scripts/export/kpdr_tracker.py
uv run python super_metroid/scripts/export/path_room_board.py
uv run python super_metroid/scripts/export/room_work_queue.py
```

Full command surface: package [`AGENTS.md`](../AGENTS.md).
