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
| Power-on → Speed Booster (K4.5) | **continuous green** | **130,388f** ×2 → `0xAD1B` beams `0x1004` items `0x3105` (default tip) |
| Bat Cave (K4.4) | previous tip history | **122,304f** ×2 non-Spazer dual (superseded as default) |
| Pure Bat → Speed Hall | **pure green** | continuous-integrated on Speed tip |
| Pure Hall → Speed collect | **pure green** | continuous-integrated on Speed tip |
| Spine tip `--to speed` | **STATUS-promoted** | dual exact match (`rr-d20` / `rr-cd0`) |
| Spazer warehouse dual | **STATUS-promoted** | **89,416 + 90,904f** beams `0x1004` (prefix) |
| ★ Next serial | Stabilize wave after Speed | bead `rr-07b` |
| ★ Next pure | Speed return → Bubble | bead `rr-g4i` |
| Parked | Frog → Speedway pure GREEN | post-Speed shortcut only |

```text
✅ Spazer warehouse dual + STATUS (rr-jx9 / rr-4wg)
✅ Continuous --to speed dual + STATUS (rr-d20 / rr-cd0) — 130388f exact
✅ Bat → Speed Hall pure + Hall → Speed pure continuous-integrated
▶  Stabilize wave after Speed continuous  ← rr-07b
▶  Pure Speed return → Bubble  ← rr-g4i
⬜  Wave / Ice pure stack + continuous
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
