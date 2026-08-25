# Super Metroid docs

**Goal:** continuous power-on → ending/credits with unlimited energy + ammo
only ([ASSIST_CONTRACT.md](ASSIST_CONTRACT.md)). Living tip: **Phantoon**
(`--to phantoon`, **195,336f** ×2). ★ Next: Gravity (`rr-kw8t`). Language:
[CONTEXT.md](../CONTEXT.md). Tip history: [STATUS.md](STATUS.md).

**Parallel Clean track:** Morph Clean green; ★ next bombs/Torizo Clean
([CLEAN_TRACK.md](CLEAN_TRACK.md)).

**Parallel Early Spazer / 100%:** red-room walljump Spazer detour → tip →
continuous fold; 100% board scaffold
([routes/TRACK_100.md](routes/TRACK_100.md)).

**Practice ROM + repertoire** (preset menus/saves): [PRACTICE_ROM.md](PRACTICE_ROM.md).

## Top-level status board

| Mark | Meaning |
|------|---------|
| ✅ | Continuous integrity green |
| ▶ | Next pure hop ready |
| 🟨 | Partial / in progress |
| ⬜ | Open |
| ⏸ | Parked (not KPDR / post-Speed only) |

Full table: **[routes/MILESTONES.md](routes/MILESTONES.md)**.

| | Milestone | Status | Frames / score |
|--:|-----------|--------|----------------|
| ✅ | Power-on → Morph … Supers (K0) | continuous | prefix |
| ✅ | → Red Tower (K1) | continuous | 80,445 |
| ✅ | → Hi-Jump / Kraid entry (K2) | continuous | 87,696 / 97,170 |
| ✅ | → Varia (K3) | continuous | **101,954** best |
| ✅ | → Business return | continuous | 113,723 ×2 |
| ✅ | → Frog Savestation (K4.0) | continuous side tip | 114,923 ×2 |
| ✅ | Cathedral + Bubble → Bat Cave (K4.4) | continuous (previous tip) | **122,304** ×2 |
| ✅ | → **Speed Booster (K4.5)** | continuous **primary** | **130,388** ×2 |
| ▶ | **Speed return → Bubble** | pure_open | ★ next pure (`rr-g4i`) |
| ⏸ | Frog Save → Speedway pure | parked | post-Speed only |
| ⬜ | → Wave / Ice (K4) | open | after Speed return |
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
| **[STATUS.md](STATUS.md)** | Verified facts + maturity gate (M5: → Ice) |
| **[plan.md](plan.md)** | Future work + structure plan |
| **[routes/MILESTONES.md](routes/MILESTONES.md)** | Prefix / tip names |
| [routes/KPDR_TRACKER.csv](routes/KPDR_TRACKER.csv) | Per-segment KPDR spine (code-owned) |
| [routes/ROUTE_KPDR.md](routes/ROUTE_KPDR.md) | Authoritative continuous any% KPDR plan |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layers, Segment contracts, tip recipe |
| [BOSS_PIPELINE.md](BOSS_PIPELINE.md) | Boss catalog → strategy → continuous order |
| [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md) | Allowed resource assists (primary path) |
| **[CLEAN_TRACK.md](CLEAN_TRACK.md)** | Parallel Bronze/Clean tips (no energy/ammo) |
| **[routes/TRACK_100.md](routes/TRACK_100.md)** | Early Spazer / 100% notes |
| [SOURCE_STATES.md](SOURCE_STATES.md) | Continuous-like pure entry states |
| [ram_map.md](ram_map.md) | WRAM addresses |
| [ROOM_TIMER.md](ROOM_TIMER.md) | Stock room timing |
| `bd ready -l super_metroid` | Ready / in-flight work |

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
| CLEAN | 11 | Parallel no-assist early tips → Torizo |
| **Total** | **~308** | Target 200–310 working depth |

## Commands (quick)

```bash
# Current continuous tip (default = ice)
uv run python snes/super_metroid/scripts/record/continuous.py --to phantoon --no-video

# Side tip (Frog Save — not primary)
uv run python snes/super_metroid/scripts/record/continuous.py --to frog --no-video

# ★ Next pure: Bat → Speed Hall (scaffold / register segment when implementing)
# Source: post_bat_cave_continuous or post_bubble_to_bat_pure (room 0xB07A)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure <bat-to-speed-hall-segment> \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bat_cave_continuous.state

# Tracker / board exports
uv run python snes/super_metroid/scripts/export/kpdr_tracker.py
uv run python snes/super_metroid/scripts/export/path_room_board.py
uv run python snes/super_metroid/scripts/export/room_work_queue.py
```

Full command surface: package [`AGENTS.md`](../AGENTS.md).
