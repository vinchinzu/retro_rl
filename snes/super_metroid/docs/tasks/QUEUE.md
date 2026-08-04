# Queue

Planner owns STATUS, natural entry, and continuous promote. Executors take
**one card per session**. Living markdown here is **ready / in-flight only**.

## Live tip

| Gate | Status | Notes |
|------|--------|-------|
| Power-on → Bat Cave (K4.4) | **continuous green** | **122,304f** ×2 → `0xB07A` |
| Checkpoint | `scratch/post_bat_cave_continuous.state` | also pure `post_bubble_to_bat_pure` |
| ★ Next serial pure | **Bat → Speed Hall** | from Bat pure/continuous successors |
| After that | Speed Room → Wave / Ice pure → compose tips | then K5 Alpha PB |
| Parked | Frog → Speedway pure GREEN | post-Speed shortcut only |

```text
✅ … → Cathedral → Bubble → Bat continuous
▶  Bat → Speed Hall pure   ← YOU ARE HERE
⬜  Speed / Wave / Ice continuous tips
⬜  K5 Alpha PB → K6 Moat/Phantoon → … → M8 credits
```

## Ready / in-flight

| Pri | Work | Card / track | Notes |
|-----|------|--------------|-------|
| **P0** | Bat → Speed Hall pure | *scaffold from template* | Source: `post_bubble_to_bat_pure` / `post_bat_cave_continuous` |
| P0 | Graph + compose Speed tip | planner after pure green | Then stabilize + STATUS |
| **P2** | Clean bombs/Torizo continuous | [`SM-CLEAN-BOMBS.md`](SM-CLEAN-BOMBS.md) | After morph green; `*_clean` only |
| P2 | Early Spazer + 100% board | BACKLOG / `SPAZER_EARLY` in routes | Parallel; no spine block |
| — | Room practice farm | planner opt-in only | Metrics board only; not product next-work |
| — | ARCH / boss primitives | BACKLOG epic rows | Planner-serial on hot modules |

No living residual stack. Closed Bubble/Cathedral/Wave-11/room-seg debris
deleted 2026-08-04 — see MILESTONES / BACKLOG / STATUS for history.

## Process pointers

| Doc | Role |
|-----|------|
| [`PROCESS.md`](PROCESS.md) | Roles, pure-first, residuals, hot modules, dual-track |
| [`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md) | Hard in-room geometry playbook |
| [`BUBBLE_TECHNIQUES.md`](BUBBLE_TECHNIQUES.md) | Bubble technique ref (tip done; maintenance) |
| [`TASK_TEMPLATE.md`](../TASK_TEMPLATE.md) | Card scaffold |
| [`SOURCE_STATES.md`](../SOURCE_STATES.md) | Pure entry states |
| [`STATUS.md`](../STATUS.md) | Verified tip claims |
| [`routes/BACKLOG.csv`](../routes/BACKLOG.csv) | Full ticket buffer |
| [`routes/MILESTONES.md`](../routes/MILESTONES.md) | Product board |
| [`CLEAN_TRACK.md`](../CLEAN_TRACK.md) | Clean parallel contract |

## Dispatch commands

```bash
# From monorepo root — scaffold Bat→Speed pure card first if missing
./snes/super_metroid/scripts/dispatch_opencode.sh --luna --variant max SM-CLEAN-BOMBS
./snes/super_metroid/scripts/farm_room_waves.sh --rounds 10 --parallel 8
```
