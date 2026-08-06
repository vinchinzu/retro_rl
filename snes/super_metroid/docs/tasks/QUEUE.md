# Queue

**Source of truth for ready work:** monorepo beads —

```bash
bd ready -l super_metroid
bd show <id>
```

This file is a **human snapshot** only. Update it when the tip changes; do not
invent tickets here that are not in `bd`. Planner owns STATUS, natural entry,
and continuous promote. Executors take **one bead per session**.

## Live tip

| Gate | Status | Notes |
|------|--------|-------|
| Power-on → Bat Cave (K4.4) | **continuous green** | **122,304f** ×2 → `0xB07A` (default tip) |
| Pure Bat → Speed Hall | **pure green** | ~810–814f → `0xACF0` |
| Pure Hall → Speed collect | **pure green** | 1695f dual, items `0x3105` |
| Spine tip `--to speed` | **wired** (unit/graph) | not continuous-green yet |
| Business continuous + Spazer | **GREEN** | **117,875f**, beams `0x1004` |
| CATH-01→CATH-02 live + Spazer | **GREEN** | pure 1202f + live dual; continuous clears CATH→Bubble |
| Continuous bat_cave / speed | **RED** | Spazer: Bubble→Bat Super door (`rr-cwu`) |
| ★ Next serial | **Stabilize Bubble→Bat under Spazer** | bead `rr-cwu` → then `rr-d20` |
| Parallel P0 | Spazer warehouse dual + STATUS | `rr-jx9` |
| Parked | Frog → Speedway pure GREEN | post-Speed shortcut only |

```text
✅ … → Business return continuous (Spazer 117,875f)
✅ Bat → Speed Hall pure + Hall → Speed pure + tip wired
✅ CATH entrance Super door under Spazer continuous (rr-n2v)
▶  Stabilize Bubble→Bat Super door (Spazer continuous)  ← rr-cwu
⬜  Continuous --to speed dual (rr-d20)
⬜  STATUS promote speed (rr-cd0)
⬜  Speed return → Wave / Ice pure stack
```

## Process pointers

| Doc | Role |
|-----|------|
| `bd ready -l super_metroid` | Ready / blocked work |
| [`PROCESS.md`](PROCESS.md) | Roles, pure-first, residuals, hot modules, dual-track |
| [`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md) | Hard in-room geometry playbook |
| [`TASK_TEMPLATE.md`](../TASK_TEMPLATE.md) | Card scaffold (geometry residuals) |
| [`SOURCE_STATES.md`](../SOURCE_STATES.md) | Pure entry states |
| [`STATUS.md`](../STATUS.md) | Verified tip claims |
| [`routes/BACKLOG.csv`](../routes/BACKLOG.csv) | Full ticket buffer (history / long tail) |
| [`routes/MILESTONES.md`](../routes/MILESTONES.md) | Product tip board |
| Root [`AGENTS.md`](../../../../AGENTS.md) | Monorepo beads rules |
