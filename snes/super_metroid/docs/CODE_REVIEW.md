# Code review: `snes/super_metroid/`

**Date:** 2026-08-04 (wave: unified TipSpec + residual agents)  
**Scope:** package health / maintainability (strict code-quality bar)  
**Verdict:** **pass**

Tests: **396 passed, 1 skipped**.

---

## Approval bar

| Criterion | Status |
|-----------|--------|
| One continuous tip interface | **pass** — `routes/tips.py` `TipSpec` only |
| RouteHop projection gone | **pass** — `RouteHop = SpineHop` |
| Single hop runner | **pass** — `tips.play_hops` (early + Super+) |
| Early play hop-composed | **pass** — real `hops=` + parent chain; `custom_play=None` |
| Early run finishers | **pass** — `assist_mode` + `final_conditions_fn` on TipSpec → `run_tip` |
| `spine.py` under 1k | **pass** — facade 162 LOC; hops/types/segments split |
| Kraid return open-loop density | **pass** — product hop 335 LOC; skills extracted |
| Report schema kind branches | **pass** — unified `to_dict` key set |
| ContinuousTip ↔ TipSpec order | **pass** — `CONTINUOUS_TIP_ORDER` + align test |
| Boss catalog table | **pass** — `_BOSS_TABLE` + thin wrappers |

---

## Residual agent wave (landed)

Five parallel agents finished the review residual list:

### 1. Early tips on TipSpec (play path)

- Early tips register real spines (`MORPH_SPINE` … `SUPERS_SPINE`) + `parent_tip_id`.
- Multi-split bookkeeping is SpineHop `after` hooks (not three hop loops).
- Public `play_*` → `play_tip`; `custom_play=None`.
- Early finish plugins: `assist_mode`, `final_conditions_fn`, `source_policy_fn`
  (no `custom_run` dual path; public `run_*` thin-wrap `run_tip`).

### 2. Spine split

```text
routes/kpdr/spine_types.py   — SpineHop, TipSegment
routes/kpdr/spine_hops.py    — POST_SUPERS_SPINE (~596)
routes/kpdr/tip_segments.py  — POST_SUPERS_TIP_SEGMENTS
routes/kpdr/spine.py         — facade (~162) + helpers
```

Public import path unchanged: `from super_metroid.routes.kpdr.spine import ...`.

### 3. Kraid return skills

| Module | Role |
|--------|------|
| `skills/door_exit.py` | Lip stage, beam open, period exit push |
| `skills/morph_bomb.py` | Align / bomb-hole climb / roll |
| `skills/kraid_return.py` | Named Kihunter + Zeela phases |
| `from_kraid.py` | **810 → 335** product hops |

### 4. Report + catalog

- `ContinuousRunReport.to_dict`: one key set for all kinds (null/empty optional fields).
- `CONTINUOUS_TIP_ORDER` + `_CONTINUOUS_TIP_META` build `CONTINUOUS_TIPS`.
- Test: catalog order matches `TIP_SPECS` / `TIP_BY_ID`.

### 5. Boss catalog table

- `_BOSS_TABLE` single source; `*_catalog()` one-line wrappers; public names preserved.

---

## Still optional (non-blocking)

1. ~~Fold early `custom_run` into generic `run_tip`~~ **done** (`assist_mode` + finish plugins).
2. ~~Drop continuous historical hop re-exports~~ **done** (hop tables only on `kpdr/hops`).
3. ~~Eye mid-room / Zeela wall phase splits~~ **done** (`eye_mid_room_approach`,
   `zeela_shotblock_clear` / `wall_replant` / `wall_spin_climb`).
4. Merge ContinuousTip CLI meta into TipSpec entirely (optional; order already locked).
5. Drop Super+ `play_<tip>` / `run_<tip>` aliases on continuous when scripts stop using them.

---

## Extend a continuous tip

1. Pure controller in `routes/kpdr/` (+ `KPDR_SEGMENTS` if needed).
2. Graph edges in `progression/stages/`.
3. `SpineHop` (+ `TipSegment`) in spine modules — first Super+ parents to `supers`.
4. CLI meta row in `catalog.py` (`CONTINUOUS_TIP_ORDER` + `_CONTINUOUS_TIP_META`).
5. `run_to("<tip>")` — no new runner.

---

## Historical waves

| # | Finding | Status |
|---|---------|--------|
| 1 | Tip multi-registry | **done** → TipSpec |
| 2 | progression `stages/` | **done** |
| 3 | Segment ≠ continuous spine | **done** |
| 4 | Bubble skills | **done** |
| 5 | K4 skills | **done** |
| 6 | Hop renames | **done** |
| 7 | Dual graph | intentional |
| 8 | RouteHop delete | **done** |
| 9 | Early vs Super+ types | **done** |
| 10 | Spine under 1k | **done** |
| 11 | Kraid return skills | **done** |
| 12 | Report schema + catalog order | **done** |
| 13 | Boss catalog table | **done** |
