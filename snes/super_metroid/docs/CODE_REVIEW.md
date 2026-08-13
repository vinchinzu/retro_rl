# Code review: `snes/super_metroid/`

**Date:** 2026-08-12 (condensation: shims / unused helpers)  
**Scope:** package health / maintainability (strict code-quality bar)  
**Verdict:** **pass**

Tests: **397 passed, 1 skipped**.

---

## Approval bar

| Criterion | Status |
|-----------|--------|
| One continuous tip runner | **pass** — `TipSpec` + `play_tip` / `run_tip` |
| Early hop-composed | **pass** — real spines; no `custom_play` |
| One tip identity surface | **pass** — CLI fields on TipSpec; ContinuousTip/NamedRoute derived |
| No dead dual-path escape hatches | **pass** — `custom_play` / `custom_run` deleted |
| One hop runner only | **pass** — `tips.play_hops` only; HopExecutor gone |
| Typed tip play evidence | **pass** — `TipPlayResult` |
| Spine under 1k | **pass** — facade ~162; hops/types/segments split |
| room_graph under 1k | **pass** — **509** + topology 337 + pathfind 140 |
| Report schema / boss catalog | **pass** |
| Historical alias freeze | **pass** — RouteHop / EarlyTipSpec / PostSupersTipSpec gone |

---

## Condensation wave (2026-08-12)

Deleted import-only shims and unused helpers. Product controllers / combat /
autopilot / practice paths untouched.

| Deleted | Why |
|---------|-----|
| `models.py`, `visual_models.py` | Top-level re-exports of `legacy/` |
| `human_tape_replay.py`, `human_tape_trim.py` | Re-exports of `human_tape` |
| `routes/spore_spawn_controller.py` | Re-export of `kpdr.spore_spawn` |
| `routes/post_supers_aliases.py` | Generated `play_<tip>` / `run_<tip>` on `continuous` — unused outside tests |
| `routes/kpdr/k4_wave.py` | Re-export of `kpdr.wave` |
| `routes/kpdr/guide_paths.py` | Re-export of `kpdr.guides` |

Also removed unused helpers (`list_sources`, `events_from_item_frames`,
`module_source_evidence`, `ceres_ridley_catalog_entry`, `air_enemy_count`,
`export_all_slices`, TAS one-line converters, unused escape-room stubs).

---

## Residual wave (landed via sub-agents)

Five parallel workstreams completed the prior review residual list:

### 1. Tips runner (`tips.py`)

- Deleted `custom_play`, `custom_run`, `_invoke_custom_play`, `is_spine_driven`, `is_hop_composed`
- `run_to_tip` is a thin alias of `run_tip`
- Typed `TipPlayResult` (`last` / `boss` / `super_collect`) end-to-end
- `register_tips` dead first loop removed; rebuilds catalog views after merge
- `_invoke_after` single signature `(session, splits, result)`

### 2. Tip CLI identity (`catalog.py` + TipSpec fields)

- CLI fields on TipSpec / TipSegment: `display_name`, `description`, `aliases`, `supports_*`
- `_CONTINUOUS_TIP_META` and hand-written `ROUTE_*` blocks **deleted**
- `ContinuousTip` + `NamedRoute` rebuilt by `rebuild_from_tip_specs()` on registration
- `CONTINUOUS_TIP_ORDER` tracks live `TIP_SPECS` order

### 3. Shadow hop runner (`segment.py`)

- Deleted `HopExecutor`, `HopResult`, `ContinuousSession`
- Kept practice adapters: `Segment`, `ControllerSegment`, `PolicySegmentAdapter`, `segment_from_kpdr`
- ARCHITECTURE: continuous hops only via `tips.play_hops`

### 4. Historical aliases (`hops.py` / early)

- Deleted `RouteHop`, `PostSupersTipSpec`, `EarlyTipSpec`, `POST_SUPERS_TIP_*` TipSpec aliases
- One public hop-table name per tip (no underscore twins)
- Super+ canonical: `SUPER_TIP_SPECS` / `SUPER_TIP_BY_ID`
- Kept thin Super+ `play_<tip>` aliases for segment registry; early `play_*_hops` for probes

### 5. room_graph split

| File | LOC | Role |
|------|----:|------|
| `rooms/topology.py` | 337 | Physical graph load / components |
| `rooms/pathfind.py` | 140 | Grid + capability path |
| `rooms/room_graph.py` | 509 | Problem gen + public facade |

Public imports via `room_graph` unchanged.

---

## Extend a continuous tip (current)

1. Pure controller in `routes/kpdr/` (+ `KPDR_SEGMENTS` if needed)
2. Graph edges in `progression/stages/` as needed
3. `SpineHop` on the spine (+ DoorEdge meta when product door)
4. **New tip:** `TipSegment` with parent, report strings, **and CLI fields**  
   (early: put CLI fields on the TipSpec in `early_continuous.py`)
5. TipSpec generated / registered → ContinuousTip + NamedRoute refresh automatically
6. `run_to("<tip>")` — no new runner, no parallel catalog prose

---

## Optional later (non-blocking)

1. Further thin NamedRoute if harness consumers never need identity milestones
2. Stronger typing on `TipPlayResult.boss` / `super_collect` (protocol or unions)

---

## Historical waves

| # | Finding | Status |
|---|---------|--------|
| 1–13 | Prior TipSpec / spine / skills / report / boss table | **done** |
| 14 | Triple tip identity (CLI) | **done** — TipSpec-derived ContinuousTip/NamedRoute |
| 15 | Dead custom_play/run | **done** |
| 16 | HopExecutor shadow runner | **done** |
| 17 | Typed tip play evidence | **done** |
| 18 | room_graph near 1k | **done** — split |
| 19 | Alias surface freeze | **done** |
