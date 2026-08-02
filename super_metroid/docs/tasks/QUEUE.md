# Super Metroid task queue

Planner (Grok / human) owns design, natural-entry judgment, STATUS, and
integrity. Executors take **one card per session**.

| Doc | Role |
|-----|------|
| **[MILESTONES.md](../routes/MILESTONES.md)** | Top-level status board (every product tip / practice rollup) |
| **[BACKLOG.csv](../routes/BACKLOG.csv)** | Full ~290-ticket decomposition to assisted full clear |
| [BACKLOG.md](../routes/BACKLOG.md) | Epic summary + ready list |
| **[TRIAGE.md](TRIAGE.md)** | Backlog review, critical path, parallel tracks |
| **[WAVE-11.md](WAVE-11.md)** | Multi-agent dispatch board (ready cards) |
| [KPDR_TRACKER.csv](../routes/KPDR_TRACKER.csv) | Per-segment KPDR spine status |
| [PROCESS.md](PROCESS.md) | Pure-first, stabilize, residual schema |
| [TASK_TEMPLATE.md](../TASK_TEMPLATE.md) | Card format |
| [SOURCE_STATES.md](../SOURCE_STATES.md) | Pure entry states |
| [plan.md](../plan.md) | Strategy + structure debt |
| [archive/](archive/) | Completed cards, residuals, farm one-shots |

Dispatch from repo root:

```bash
# Spine serial
./super_metroid/scripts/dispatch_opencode.sh --luna --variant max SM-K4-SPEEDWAY-PURE
./super_metroid/scripts/dispatch_opencode.sh --flash SM-K4-SPEEDWAY-SRC

# Wave-11 parallel (disjoint own-files; width ≤ 8)
./super_metroid/scripts/dispatch_opencode.sh --flash \
  SM-PATH-ROOM-W01a SM-PATH-ROOM-W01b SM-PATH-ROOM-W01c SM-PATH-ROOM-W01d \
  SM-BOSS-PRIM-LANE SM-BOSS-NATURAL-ENTRY-CLI SM-ARCH-RED-DIAG
./super_metroid/scripts/dispatch_opencode.sh --luna SM-ARCH-HOPS-MODULE

# Dual-track bulk practice
./super_metroid/scripts/farm_room_waves.sh --rounds 10 --parallel 8
```

## Process gates (non-negotiable)

1. **Pure-first** from continuous-like source before continuous re-record.
2. **Stabilize wave** after spine knobs land (pure + continuous before more knobs).
3. **One knob / residual** → next card ID + one change.
4. **Serialize** hot modules: climb/return geometry, `continuous.py`,
   `progression.py`, `catalog.py`, `STATUS.md`.
5. **Dual track:** `ROOM_WORK_QUEUE` practice ≠ KPDR continuous integrity.
6. **Force-pass ban:** scaffolds never claim pure/continuous green.

## ★ Live tip (2026-08-01)

| Gate | Status | Evidence / action |
|------|--------|-------------------|
| Power-on → Frog Save (K4.0) | **continuous** | 114,923f ×2 integrity green |
| Checkpoint | `scratch/post_frog_continuous.state` | room `0xB167` |
| Frog Save → Speedway pure | residual **GREEN** | pure controller green; **not** first Bubble path |
| Speedway → Farm pure | residual **RED** | Boost Blocks need Speed (`SM-K4.2-PURE-residual.md`) |
| **K4 repath (chosen)** | Cathedral first Bubble | Business → `0xA7B3` → `0xA788` → `0xAFA3` → Bubble |
| **Next serial** | **`SM-K4-CATH-01` pure** | [`SM-K4-CATH-01`](SM-K4-CATH-01.md) from `post_business_continuous` |
| After Cathedral → Bubble | Bat Cave → Speed Hall → Speed → Wave / Ice | then K5 Alpha PB |

```bash
# Pure successor already captured (re-verify only if flaky)
uv run python super_metroid/scripts/probe/kpdr.py pure frog-save-to-speedway \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_continuous.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state \
  --pin-json super_metroid/debug/frog_save_to_speedway_pure_pin.json
```

Cards: Frog Save continuous + Speedway pure GREEN (parked as post-Speed) ·
next **[`SM-K4-CATH-01`](SM-K4-CATH-01.md)** Cathedral climb.  
Wave-11 closeout: [`WAVE-11.md`](WAVE-11.md). Triage: [`TRIAGE.md`](TRIAGE.md).

**Planner only after pure green:** graph edge → compose tip → continuous
stabilize → STATUS. Never in a farm batch.

## Epic board (product path → M8)

```text
✅ K0 Supers → ✅ K1 Red → ✅ K2 Kraid entry → ✅ K3 Varia/Business
✅ K4.0 Frog Save
▶  K4 Speed/Wave/Ice     ← YOU ARE HERE (Cathedral repath → CATH-01 pure)
⬜  K5 Alpha PB
⬜  K6 Moat → Phantoon → Gravity
⬜  K7 Maridia → Botwoon → Draygon → Space Jump
⬜  K8 Lower Norfair → Ridley
⬜  K9 G4 → Tourian → MB → Escape → Credits  (M8 full clear)
```

Detail + frames: [MILESTONES.md](../routes/MILESTONES.md).  
Ticket depth by epic: [BACKLOG.md](../routes/BACKLOG.md).  
Critical path + parallel lanes: [TRIAGE.md](TRIAGE.md).

### Ticket recipe (each hop)

| Step | Kind | Owner |
|------|------|-------|
| 1 | `pure` controller | Luna |
| 2 | `graph` edge | Luna / Flash |
| 3 | `compose` tip hops | Luna scaffold → planner |
| 4 | `stabilize` pure + continuous | planner-serial |
| 5 | `status` STATUS + tracker | Flash proposal → planner apply |

Boss hops insert `boss` catalog/phase/closeout cards before continuous tip.

## Wave-11 (closed 2026-08-01) — all ready cards GREEN

| Track | Cards | Residuals |
|-------|-------|-----------|
| **B spine** | `SM-K4-SPEEDWAY-SRC` | catalog `post_frog_save_to_speedway_pure` |
| **C practice** | `SM-PATH-ROOM-W01a`…`W01d` | practice-promoted (dual-track) |
| **BOSS-INFRA** | `SM-BOSS-PRIM-LANE`, `SM-BOSS-NATURAL-ENTRY-CLI`, `SM-BOSS-UNIT-MATRIX` | lane window + capture CLI + matrix tests |
| **ARCH** | `SM-ARCH-HOPS-MODULE`, `SM-ARCH-RED-DIAG` | `routes/kpdr/hops.py` + pure RED frame dumps |

**Spine repath applied (2026-08-01):** option **1** — first Bubble = Cathedral
climb. Graph: `business_to_cathedral_entrance` … `rising_tide_to_bubble`;
`speedway_to_farm` / `farm_to_bubble` require `speed_booster`; reverse
`frog_save_to_business` for tip at Frog. **Next pure:** [`SM-K4-CATH-01`](SM-K4-CATH-01.md).

Follow-ons (parallel OK): `SM-BOSS-PRIM-SPRAY`, `SM-ARCH-TIP-SPEC`, path-room farm.  
Detail: [`WAVE-11.md`](WAVE-11.md).

## Parallel tracks

| Track | What | Integrity |
|-------|------|-----------|
| **B spine** | KPDR continuous tips | Assist contract; natural entry |
| **C practice** | `ROOM_WORK_QUEUE` / farm waves | Dual-track only; not continuous |
| **ARCH** | Structure debt | Planner-serial; no tip claims |
| **BOSS-INFRA** | Catalog / primitives / capture CLI | Dev only until natural entry |

Practice board after Wave-10: easy+standard ready **62/108 (57.4%)** —
see `ROOM_WORK_QUEUE`. Generator: `scripts/generate_room_segment_cards.py`.

## Architecture debt (planner-serial)

| ID | Goal | Status |
|----|------|--------|
| SM-ARCH-TIP-SPEC | Hop tables out of `continuous.py` | partial |
| SM-ARCH-GRAPH-API | Typed path-summary model | partial |
| SM-ARCH-PARSE-SCOPE | Session-scoped parse counters | partial |
| SM-ARCH-HOPS-MODULE | `routes/kpdr/hops.py` | **ready** (Wave-11 card) |
| SM-ARCH-RED-DIAG | Pure RED clip + PLM/door snapshot | **ready** (Wave-11 card) |
| SM-ARCH-PROFILE | WRAM-copy profile on long tips | open |

Full list in `BACKLOG.csv` epic `ARCH`. Detail: [ARCHITECTURE.md](../ARCHITECTURE.md).

## Closed waves (pointer only)

Historical wave write-ups, residuals, and farm cards live under
[`archive/`](archive/). Do not treat archive greens as current tip evidence.

| Wave | Result |
|------|--------|
| Wave-10 dual-track farm | 9 honest GREEN promotes; continuous tip re-verify green |
| K3 return / Business / Frog | continuous 113,723f / 114,923f |
| Wave-6 stabilize + process seed | closed 2026-07-31 |

## Hygiene

- **Do not** commit `docs/tasks/logs/` (gitignored).
- Archive residuals after the successor card exists; keep only living cards in
  `docs/tasks/SM-*.md`.
- After tip promotion: update `MILESTONES.csv`, `KPDR_TRACKER.csv`, `STATUS.md`,
  regenerate path board / tracker export.
- Backlog is the capacity buffer (~290 tickets). Living markdown cards are only
  for **ready / in-flight** executor work — scaffold from `TASK_TEMPLATE.md`
  when promoting a backlog row.
