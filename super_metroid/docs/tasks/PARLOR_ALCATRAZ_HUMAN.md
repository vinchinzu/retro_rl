# Parlor Alcatraz (post–Bomb Torizo)

## Open-loop double WJ (authoritative)

**Do not** treat human task JSON as path truth — use only for rough WJ pulse
feel. Product climb is derived open-loop from geometry.

| Item | Value |
|------|-------|
| Probe | `scripts/probe/parlor_chimney_double.py` |
| Best recipe | `--recipe ext_40` (default) |
| Chim min_y | **≈243** (beats human peak ~256) |
| Double class | **YES** — chained rises y459→363 (+96) then y363→266 (+97) |
| Lip class y≤210 | **NO** (still open) |
| Video | `recordings/parlor_chimney_double_best.mp4` |
| Ref path (KPDR) | `recordings/alcatraz_ref_t2846_10s.mp4` (yt t=2846, 10s) |

### Recipe (ext_40)

1. **Base:** door (968,651) → hop ladder → left-wall floor **(805,545)**
2. **Mid ledge:** spin RIGHT+A 40 / LEFT+A 28 ×3 → **~(828,459)**
3. **Chimney:** into left wall, then pairs `(40,30),(36,28),(32,26),(28,24)` of
   RIGHT+A spin / LEFT+A latch

```bash
uv run python super_metroid/scripts/probe/parlor_chimney_double.py
uv run python super_metroid/scripts/probe/parlor_chimney_double.py --recipe baseline  # chim≈252
uv run python super_metroid/scripts/probe/parlor_chimney_double.py --recipe midrise_260
```

### Next

- [ ] Third bounce toward shaft lip y≤210 (mid-rise cut after y~270)
- [ ] Midair morph-out at lip (Alcatraz escape)
- [ ] Fold into product parlor controller only after lip greens ×2

---

## Human demos (archive — not path authority)

## Sources (human guided_human)

| Task | Frames | Path | Role |
|------|--------|------|------|
| `tasks/parlor_left_human.json` | 3244 | long practice / many tries | multi p132 clusters |
| `tasks/parlor_left_human2.json` | 630 | clean short demo | approach + one high contact |
| Start pin | — | `scratch/post_torizo_parlor_continuous.state` | (968,651) Flyway door |
| End states | — | `tasks/parlor_left_human*_end.state` + scratch copies | |

Route guide: `parlor-left` / `GUIDE_PARLOR_ALCATRAZ` in `routes/kpdr/guide_paths.py`.

## Geometry (room `0x92FD`)

Not the product Terminator platform hop. This is **Alcatraz shaft** left of Flyway door:

| Pin | Approx xy | Notes |
|-----|-----------|--------|
| flyway-door | (968, 651) | continuous post-BT settle |
| mid platform | ~(895, 539) | first land after left hop |
| mid ledge | ~(840–860, 459) | setup ledge for shaft |
| left wall | x≈805 | left face of chimney |
| right wall / high | ~(855–875, 256–360) | spin-up (p131) + latch (p132) |

Human min_y ≈ **256–258** (one high contact class). Neither demo cleared shaft-lip (~210) cleanly.

## Human2 (clean) one-contact recipe

Frames relative to task start:

1. **f29–57** LEFT+B run from door  
2. **f67–88** RIGHT+A hop → mid platform y~540  
3. Settle ~y539  
4. **f180–204** LEFT+B+A toward shaft  
5. Mid-ledge work ~y459  
6. **f411–446** LEFT+B+A → left wall ~(805, 354)  
7. **f452–491** RIGHT+A **spin (pose 131)** up right wall → peak ~(858, **256**)  
8. **f497–530** pose **132** latch LEFT(+B)+A ~34f — **slides down** (y257→311); no clean bounce  

**Human report:** can land **one** jump / high contact; back-and-forth possible; wants **string 2** WJ.

## Human1 p132 clusters (real up-WJ when y falls)

| Cluster | len | y0 → min_y | Notes |
|---------|-----|------------|--------|
| f1856–1890 | 35 | 259 → 259 | latch at peak, no gain |
| f2124–2141 | 18 | 350 → 307 | partial |
| f2441–2478 | 38 | 356 → **258** | **best up-WJ** LEFT+A on right wall |
| f2606–2637 | 32 | 355 → 307 | partial |
| f2780–2809 | 30 | 354 → **264** | second good up-WJ |

No two p132 clusters within 100f — **double not landed in demos**. Closest pattern: spin-up RIGHT+A → LEFT+A latch/WJ1 → (need) flip to left wall RIGHT+A WJ2.

## Extracted open-loop knobs (for probe)

From human2 approach + human1 best WJ1:

```
LEFT run / hop to mid ledge y459
LEFT+B+A ~29–36f into left wall x805
RIGHT+A spin (p131) ~35–40f up right wall
LEFT+A (p132 window) ~28–36f  → WJ1
# experimental WJ2:
RIGHT face / RIGHT+A ~20–30f into left wall bounce
```

Shared skill surface: `controller_common.WallJumpTiming` / `walljump_once` / `consecutive_walljumps`.

## Probe results (2026-08-03, same day as demos)

| Probe | min_y | p132 frames | Note |
|-------|------:|------------:|------|
| open-loop approach + WJ1 | 508 | 0 | missed mid-ledge (stayed y~635) |
| open-loop + WJ2 | 508 | 0 | same |
| human2 replay f0–496 + inject WJ1+WJ2 | **256** | 33 | matches human peak; inject latched but **no height gain** / no 2nd bounce |

**String-2 status:** not yet. Human peak is spin-up (p131) to y≈256; p132 LEFT+A at peak slides down unless timed mid-rise like human1 f2441 (y356→258 while latched).

Video artifacts: `recordings/parlor_alcatraz_wj{1,2}.mp4`, `recordings/parlor_alcatraz_human2_plus_wj2.mp4`.

## Next session parse checklist

- [ ] Re-read both task JSON traces; mark successful WJ1 frames only (pose 132 with **decreasing** y)  
- [ ] Measure inter-WJ gap (frames + Δx) if a human lands 2  
- [ ] Retune `GUIDE_PARLOR_ALCATRAZ` waypoints from human min path (**partially done** — mid-plat/ledge/high-contact)  
- [ ] Product: replace wrong Terminator “chimney” open-loop only after pure probe greens 2-WJ height class  
- [ ] Task replay helper: human2 through f450 (left wall y355) then open-loop spin + WJ1 mid-rise + WJ2  
- [ ] Optional: bit-exact human2 replay CLI for video compare
## Commands

```bash
# Human re-record
uv run python super_metroid/scripts/record/guided_human.py \
  --from parlor --route parlor-left --name parlor_left_human3

# Probe open-loop 1–2 WJ (debug video)
uv run python super_metroid/scripts/probe/parlor_alcatraz_wj_probe.py
```
