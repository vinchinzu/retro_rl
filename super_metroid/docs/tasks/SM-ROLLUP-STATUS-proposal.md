# SM-ROLLUP-STATUS Proposal — NOT APPROVED

> **Planner must approve** every diff below. This file is a proposal only.
> Executor Flash has read QUEUE, JSON reports, and all source documents.
> No changes to `STATUS.md`, `KPDR_TRACKER.csv`, `PATH_ROOM_BOARD.md`, or
> `scripts/export/path_room_board.py` have been applied by this card.

---

## 1. Facts from QUEUE + JSON reports

### Verified continuous baselines (reports on-disk)

| Tip | Frames | Date | Integrity | Outcome |
|-----|-------:|------|-----------|---------|
| `--to varia` | **101,954** | 2026-07-30 | 0 loads / 0 prog writes | `varia_collected` |
| `--to kraid` (Wave-5 green) | **96,924** | 2026-07-31 17:11 UTC | 0 loads / 0 prog writes | `kraid_entry` |
| `--to hijump` | **87,696** | 2026-07-30 | 0 loads / 0 prog writes | `hijump_collected` |
| `--to warehouse` | **83,512** | 2026-07-31 | 0 loads / 0 prog writes | `warehouse_entry` |
| `--to below_spazer` | **82,300** | 2026-07-30 | 0 loads / 0 prog writes | `below_spazer_entry` |
| `--to bat` | **81,652** | 2026-07-30 | 0 loads / 0 prog writes | `bat_room_entry` |
| `--to red_tower` | **80,445** | 2026-07-30 | 0 loads / 0 prog writes | `red_tower_entry` |
| `--to supers` | **73,251** | 2026-07-30 | 0 loads / 0 prog writes | `spore_supers_collected` |

### Wave-5 dwell delta (kraid 96,924f vs STATUS prior 97,170f = −215f)

| Dwell | Prior baseline | Wave-5 green | Delta | Status |
|-------|--------------:|:------------:|:-----:|--------|
| Total | 97,170f | 96,924f | **−215f** | Not stable; single run |
| `business_to_warehouse` | 2,257f | 2,006f | **−251f** | Not promoted (multi-run pending) |
| `hj_shaft_to_business` | ~1,871f | 1,835f | **−36f** | Band noise |

Source: `recordings/start_to_kraid.json` splits analysis; splits have no `id`
field but frame sequence matches known hop order.

### Pure controller status (from QUEUE + progression.py)

| Edge | Verification | Pure status | Notes |
|------|-------------|-------------|-------|
| `varia_to_kraid` (0xA6E2→0xA59F) | `controller_dev` (progression.py:1304) | GREEN per QUEUE | Promoted; has source state |
| `kraid_to_eye_return` (0xA59F→0xA56B) | `unverified` (progression.py:1314) | **RED** | Door_transition=0 across 06B/06C/06D |
| `eye_to_baby_return` | `unverified` | **BLOCKED** | No chained eye source |
| `baby_to_kihunter_return` | `unverified` | BLOCKED | Scaffold only |
| `kihunter_to_zeela_return` | `unverified` | BLOCKED | Scaffold only |
| `zeela_to_warehouse_return` | `unverified` | BLOCKED | Scaffold only |

### Progression graph state (reverse edges)

- `varia_to_kraid`: `controller_dev` ✓
- `kraid_to_eye_return` through `zeela_to_warehouse_return`: all `unverified`
- All forward K2/K3 edges: `continuous`

### SOURCE_STATES catalog

Sources known:
  - `scratch/post_varia_collected.state` → 0xA6E2 (varia collect)
  - `scratch/post_varia_to_kraid_pure.state` → 0xA59F (post-varia return)
  - `scratch/continuous_like_business_climb_entry.state` → Business floor

Gaps:
  - `eye-to-baby` needs source (blocked by pure door RED)
  - HJ shaft mid-climb isolation partial (SM-HJ-SRC)

---

## 2. Proposed diffs (planner must approve, executor may not apply)

### A. STATUS.md — minimal honesty update

**Change 1:** Update `Last verification` date from `2026-07-30` → `2026-07-31`
(supported by `recordings/start_to_kraid.json` generated_at
`2026-07-31T17:11:37+00:00`).

**Change 2:** Add a note under the Kraid baseline table (section before "Business
climb continuous fixes" block):

> ```
> **Wave-5 note (2026-07-31):** `--to kraid` re-recorded at **96,924f** (−251f
> business, −36f hj_shaft; total −215f). Not promoted to STATUS — multi-run
> stability pending. Dwell deltas from single-run data; see QUEUE.md Wave-5
> rollup.
> ```

**Do NOT change:** frame counts in STATUS.md tables (97,170f stays until planner
approves multi-run stable; 96,924f is a single run).

**Do NOT touch:** verification date on the varia section (no new varia report
after Jul 30).

### B. KPDR_TRACKER.csv — minor date/note updates, no status changes

**Change 1:** K2.18 (row 34) — notes column add suffix:
```
Current: "97170f continuous tip; natural eye-door"
Proposed: "latest 96924f (Wave-5 green); prior 97170f in STATUS — multi-run stability pending"
```

**Change 2:** K3.2 (row 37) — status is already `controller_dev` ✓; no change.
Justification from QUEUE: `varia_to_kraid` pure-green, promoted to
`controller_dev` in progression.py.

**Change 3:** K3.3 (row 38) — add to notes:
```
Current: "return edge kraid_to_eye_return (0xA59F→0xA56B); scaffold exists pure-unverified"
Proposed: (no status change) append ", still RED @ door_transition=0 (06B/06C/06D)"
```

**No other status changes:** K3.4–K3.8 remain `open` (scaffold-only; no pure
source).

### C. PATH_ROOM_BOARD.py + regenerated PATH_ROOM_BOARD.md

**These require planner approval because path_room_board.py is the source of
truth for the board — if planner approves edits, executor runs the export
script.**

#### C1. ROOM_STATUS updates (in `scripts/export/path_room_board.py`)

Facts from QUEUE + STATUS.md:
- K2 is continuous through Kraid entry (K2.18) and Varia (K3.1)
- The K2.7–K2.18 rooms (Business Center, HJ Shaft, HJ Room, Warehouse Zeela,
  Kihunter, Baby Kraid, Eye Door) are all on the continuous `--to kraid` /
  `--to varia` chain, not just `controller_dev`

| Room | Current | Proposed | Source |
|------|---------|----------|--------|
| `0xA7DE` Business Center | `controller_dev` | **`continuous`** | On `start_to_kraid` chain (K2.13) |
| `0xAA41` Hi-Jump Shaft | `controller_dev` | **`continuous`** | On `start_to_kraid` chain (K2.11–12) |
| `0xA9E5` Hi-Jump Room | `controller_dev` | **`continuous`** | On `start_to_kraid` chain (K2.10) |
| `0xA471` Warehouse Zeela | `controller_dev` | **`continuous`** | K2.14 on `start_to_kraid` |
| `0xA4DA` Warehouse Kihunter | `controller_dev` | **`continuous`** | K2.15 on `start_to_kraid` |
| `0xA521` Baby Kraid Room | `controller_dev` | **`continuous`** | K2.16 on `start_to_kraid` |
| `0xA56B` Kraid's Eye Door | `controller_dev` | **`continuous`** | K2.17 on `start_to_kraid` |
| `0xA59F` Kraid's Room | `boss_deferred` | **`continuous`** | K2.18 + K3.0 on continuous chain; fight is a segment within `continuous` |
| `0xA6E2` Varia Suit Room | `open` | **`continuous`** | K3.1 on `start_to_varia` chain |

#### C2. HOP_STATUS updates

| Hop | Current | Proposed | Source |
|-----|---------|----------|--------|
| `0xA6A1→0xA7DE` | `controller_dev` | **`continuous`** | `warehouse_to_business` on `start_to_kraid` |
| `0xA7DE→0xAA41` | `controller_dev` | **`continuous`** | `business_to_hj_shaft` |
| `0xAA41→0xA9E5` | `controller_dev` | **`continuous`** | `hj_shaft_to_hj_room` |
| `0xA9E5→0xAA41` | `controller_dev` | **`continuous`** | `hj_room_to_shaft` |
| `0xAA41→0xA7DE` | `controller_dev` | **`continuous`** | `hj_shaft_to_business` |
| `0xA7DE→0xA6A1` | `controller_dev` | **`continuous`** | `business_to_warehouse` |
| `0xA6A1→0xA471` | `controller_dev` | **`continuous`** | `warehouse_to_zeela` |
| `0xA471→0xA4DA` | `controller_dev` | **`continuous`** | `zeela_to_kihunter` |
| `0xA4DA→0xA521` | `controller_dev` | **`continuous`** | `kihunter_to_baby_kraid` |
| `0xA521→0xA56B` | `controller_dev` | **`continuous`** | `baby_kraid_to_eye` |
| `0xA56B→0xA59F` | `controller_dev` | **`continuous`** | `eye_to_kraid` |
| `0xA59F→0xA6E2` | (missing) | **`continuous`** | `kraid_to_varia` — K3.1 continuous |
| `0xA6E2→0xA59F` | (missing) | **`controller_dev`** | `varia_to_kraid` — K3.2 pure-green |
| `0xA59F→0xA56B` | (missing) | **`open`** | `kraid_to_eye_return` — K3.3, scaffold only |
| `0xA56B→0xA521` | (missing) | **`open`** | `eye_to_baby_return` — K3.4, scaffold only |
| `0xA521→0xA4DA` | (missing) | **`open`** | `baby_to_kihunter_return` — K3.5 |
| `0xA4DA→0xA471` | (missing) | **`open`** | `kihunter_to_zeela_return` — K3.6 |
| `0xA471→0xA6A1` | (missing) | **`open`** | `zeela_to_warehouse_return` — K3.7 |

#### C3. furthestContinuous update

```python
# Current:
"furthestContinuous": {
    "roomIdHex": "0xA6A1",
    "name": ...,
    "evidence": "recordings/start_to_warehouse.json (83512f)",
},
# Proposed:
"furthestContinuous": {
    "roomIdHex": "0xA6E2",
    "name": "Varia Suit Room",
    "evidence": "recordings/start_to_varia.json (101954f)",
},
```

#### C4. furthestControllerDev update

K3.2 (varia_to_kraid reverse) is the new furthest `controller_dev` hop:

```python
"furthestControllerDev": {
    "roomIdHex": "0xA59F",
    "name": "Kraid's Room",
    "position": ...,  # capture from pure probe or keep
    "note": "Post-Varia pure return (varia_to_kraid, controller_dev); K3.2",
    "probe": "kpdr.py pure varia-to-kraid --source scratch/post_varia_collected.state",
},
```

#### C5. W2 wave status update

Move "continuous Warehouse → Hi-Jump → Kraid" from open → done in the
`_waves()` function:

```python
# Current open list:
"open": ["continuous Warehouse → Hi-Jump → Kraid"],
# Proposed:
"done": [
    ...existing items...,
    "continuous Warehouse → Hi-Jump → Kraid (start_to_kraid/varia verified)",
],
"open": [],   # W2 fully done
```

Then either close W2 or leave `in_progress` with a note that W3 is the next
focus. Since W3 is still marked `open`, the simplest change is just emptying
the W2 open list.

#### C6. After applying C1–C5: regenerate board

```bash
uv run python super_metroid/scripts/export/path_room_board.py
```

---

## 3. Open pure blockers with next card IDs

| Blocked edge | Room | Status | Root cause | Next card |
|-------------|------|--------|------------|-----------|
| `kraid_to_eye_return` (K3.3) | 0xA59F→0xA56B | **RED** | door_transition=0 after 06B/06C/06D | SM-K4-06E (one-knob pure door — blocked until Wave-6 stabilize) |
| `eye_to_baby_return` (K3.4) | 0xA56B→0xA521 | BLOCKED | No chained eye exit source | SM-SRC-EYE (after K3.3 pure green → capture) |
| HJ shaft mid-climb | 0xAA41 | PARTIAL | ensure_morph RED | SM-HJ-SRC follow-up or continuous dump |

**Key blocker:** `kraid_to_eye_return` pure-green still requires a new
primitive / approach. QUEUE Wave-6 says "no free spin; next one-knob only
after stabilize."

---

## 4. QUEUE metrics (Wave 6 — partial)

| Metric | Value |
|--------|------:|
| Pure-green rate (geometry cards, Wave 5) | 1/5 (P2B green; 06B/C/D/E red) |
| Continuous regression rate (Wave 5) | 0/1 (green @ 96,924f) |
| Top dwell (from 96,924f report) | business_to_warehouse 2,006f; hj_shaft_to_business 1,835f |
| Session count | Wave 5: 11 sessions; Wave 6: 3 running (STAB-KRAID, PRIM-01, this) |
| Wave-5 stack conflicts | business_climb 01C + P2 (race accepted); 04A reverted |

---

## Residual — SM-ROLLUP-STATUS

### Result
GREEN (proposal filed; no files changed)

### Files changed
- `docs/tasks/SM-ROLLUP-STATUS-proposal.md` — created with full proposed diffs

### Verify paste
```text
$ test -f super_metroid/docs/STATUS.md && echo "STATUS.md: OK"
STATUS.md: OK
$ test -f super_metroid/docs/tasks/QUEUE.md && echo "QUEUE.md: OK"
QUEUE.md: OK
$ test -f recordings/start_to_kraid.json
41044 recordings/start_to_kraid.json
```

### Acceptance
- [x] Proposal file exists with STATUS / tracker / board sections
- [x] Each proposed change cites QUEUE or JSON field
- [x] Explicit "planner must approve" banner at top
- [x] Residual uses PROCESS residual schema (next card + one change)

### Residual risks
- STATUS frame numbers (97,170f) are stale — 96,924f report is newer but single
  run. Status quo is honest: STATUS says 97,170f and the file is Jul 30; the
  new report (96,924f, Jul 31) simply has not been approved.
- PATH_ROOM_BOARD.py ROOM_STATUS is significantly outdated — 9 rooms (Business
  Center through Varia Suit) listed as `controller_dev` or `open` when they
  are actually on the continuous chain. Planner should prioritize board sync.
- PATH_ROOM_BOARD.py HOP_STATUS missing 6 hops (K3.0–K3.7). Adding them with
  correct verification tags would make the board match reality.
- Multi-run stability not yet established; do not promote frame savings.

### Next action (required)
- **Next card ID:** SM-STAB-VARIA (after SM-STAB-KRAID green) or PLANNER-GATE
  for reviewing/approving these proposals
- **One change:** Planner reviews proposal, either applies diffs to
  path_room_board.py + regenerates board (then STATUS + tracker), or marks
  sections as "deferred until stabilize"
- **Source state:** N/A (docs-only card)

### Non-claims
- Did not STATUS-promote or edit STATUS.md
- Did not edit KPDR_TRACKER.csv or regenerate tracker
- Did not edit path_room_board.py or regenerate boards
- Did not forge progression/capacity/door/event/boss RAM
- Not continuous evidence

### Probe pin
N/A (docs-only rollup card; no emulator interaction)