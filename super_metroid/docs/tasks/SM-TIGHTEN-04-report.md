# SM-TIGHTEN-04: Deep analysis — green_brinstar_main_shaft dwell (report only)

**Date:** 2026-07-31
**Source:** `recordings/start_to_varia.json` (outcome=`varia_collected`, total_frames=101,954)
**Model:** Flash (report-only; no controller/continuous/STATUS edits)

## 1. Commands

```bash
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_varia.json --top 15
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_varia.json --reasons --top 30
```

## 2. Split location and definition

- Split id `green_brinstar_main_shaft`, room `0x9AD9` (Green Brinstar Main Shaft).
- Defined in `routes/continuous.py:663-665` as `split_for_transition(..., 0x9938, 0x9AD9)` —
  i.e. the **entry frame** of `0x9AD9` after the Green Elevator room (`0x9938`).
- Dwell rank 10/15 by room dwell: **2,806 frames** (~46.8 s at 60 Hz) between
  entering `0x9AD9` and the next recorded split.
- Nearest sibling split: `terminator_energy_tank` (`0x990D`, 4,693f) ends at
  the E-Tank collection (frame 49,972); the shaft entry lands at frame 52,778
  (dwell ends at 55,584). The gap includes Green Pirates `0x99BD` (740f
  `green_pirates_descent` reason) and Lower Mushrooms `0x9969` — those are
  **not** their own splits.

## 3. Controller phase map (line/reason labels)

The shaft dwell is played by **`routes/spore_spawn_controller.py`**, not a
`routes/kpdr/*.py` room controller. The Green Brinstar main-shaft leg is the
second half of `play_parlor_to_main_shaft` (elevator exit) plus the first
section of `play_main_shaft_to_spore_spawn` (shaft descent to Dachora).

### 3a. Entry: Green Elevator settle → shaft landing
`routes/spore_spawn_controller.py:229-247` (inside `play_parlor_to_main_shaft`):

| Reason label | Line | Frames (fixed) | Verdict |
|--------------|-----|---------------:|---------|
| `green_elevator_entry_settle` | 229 | 240 | scripted settle |
| `green_elevator_center` loop | 231-244 | up to ~600 | positioning loop (elevator center) |
| `green_elevator_descend` | 245 | 10 | input |
| `green_elevator_descent_settle` | 246 | 1,000 | **scripted elevator ride** (fixed) |
| — `_require_room(0x9AD9)` | 247 | — | split boundary |

### 3b. Shaft descent (inside `play_main_shaft_to_spore_spawn`)

| Reason label | Line | Frames (fixed) | Verdict |
|--------------|-----|---------------:|---------|
| `main_shaft_entry_settle` | 254 | 1,000 | **idle settle — top tighten target** |
| `main_shaft_descent` | 255-261 | 4 × 60 = 240 | platform zigzag descent |
| `main_shaft_descent_settle` | 262 | 50 | short settle |
| `main_shaft_dachora_level` | 263-270 | 5 × 80 = 400 | descent to Dachora door level |
| `main_shaft_dachora_door_settle` | 271 | 30 | short settle |
| `select_missiles` + settle | 272-273 | 1 + 10 = 11 | weapon select |
| `open_dachora_red_door` loop | 274-277 | 15 × (2+15) = 255 | missile-door cadence (scripted) |
| `enter_dachora` + settle | 277-278 | 100 + 250 = 350 | door entry + post-door settle |
| — `_require_room(0x9CB3)` | 279 | — | exit boundary |

**Sum (fixed slices):** 1,000 + 240 + 50 + 400 + 30 + 11 + 255 + 350 =
**2,336f** accounted of 2,806f. The ~470f remainder is the door/transition
run for the elevator settle → landing pose + the Dachora door push frames not
in the fixed table.

## 4. Top waste analysis

| Band | Frames | % of split | Nature |
|------|-------:|-----------:|--------|
| `green_elevator_descent_settle` (1,000f) | 1,000 | 35.6% | **scripted elevator ride — not controllable** |
| `main_shaft_entry_settle` (1,000f) | 1,000 | 35.6% | **idle wait after elevator — controllable (reduce)** |
| `open_dachora_red_door` (255f) | 255 | 9.1% | missile-door cadence — scripted-ish |
| `enter_dachora` post-door settle (250f) | 250 | 8.9% | settle after door — semi-controllable |
| `main_shaft_dachora_level` (400f) | 400 | 14.3% | active descent — tighten only with care |

Top takeaways:

1. **~2,000f (~71%) of the split is wait time**: one scripted elevator ride
   (unavoidable) and one 1,000f idle settle that is deliberately generous.
2. **The controllable slack is `main_shaft_entry_settle` (1,000f)**. The
   controller waits a full second idle at the shaft top before starting the
   zigzag descent. A pose/x/y-guarded settle (like `wait_ordinary_room`) could
   cut this to ~100–200f while keeping the split green.
3. Remaining active movement (240+50+400+30+11+350 ≈ 1,081f) is mostly the
   unavoidable descent through platforms plus door work.

## 5. Elevator/scripted verdict

**Mostly elevator + scripted? Yes — but only ~45% is truly unavoidable.**
- `green_elevator_descent_settle` (1,000f) is a fixed elevator ride; the ride
  itself cannot be shortened, and settle-after-load is safety.
- The 1,000f `main_shaft_entry_settle` is **not** a scripted wait — it is a
  blanket idle the controller added for settle safety. That is the single
  tightenable slice.
- Combined scripted-ish bands (elevator ride + red-door cadence + entry
  settles): roughly 1,000 + 255 + 250 + 50 + 30 ≈ 1,585f (~56%). The rest
  (~1,221f) is controllable movement + idle.

## 6. Future patch recipes (speculative bands — no savings claimed)

### Patch A — guard the shaft-top settle (highest value, lowest risk)
- Replace `_hold(session, 1_000, reason="main_shaft_entry_settle")` at
  `spore_spawn_controller.py:254` with a pose/x/y-guarded settle
  (`wait_ordinary_room`-style) using the recorded landing boundary
  (x≈118–126, standing pose) and a timeout ≈ 300–400f.
- Speculative saving: ~600–800f on this split. Requires re-record.
- Entry natural: split `green_brinstar_main_shaft` begins exactly here, so a
  regression would appear in `--reasons` (`main_shaft_entry_settle` frames).

### Patch B — trim `enter_dachora` post-door settle
- `spore_spawn_controller.py:277-278`: `100 + 250 = 350f` for the Dachora door
  push + settle. Reduce the trailing 250f to ~150f with a `room_id == 0x9CB3`
  guard; the settle exists to absorb load time.
- Speculative saving: ~100f. Medium risk (multi-screen door), needs re-record.

### Patch C — tighten `main_shaft_dachora_level` cadence
- `spore_spawn_controller.py:263-270`: five 80f `(RIGHT/LEFT, B)` holds while
  descending. Could drop to ~60f per leg (~100f total) if x/y progress stays
  monotonic; monitor for bounce-back on the lower ledge.
- Speculative saving: ~100f. Medium risk (geometry-sensitive).

**Combined speculative upper bound: ~800–1,000f (~13–17 s) on a 2,806f split,
if Patch A lands.** All values are pre-re-record bands only.

## 7. Acceptance command for an implement card

```bash
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_varia.json --reasons --top 30 | grep -E \
  "main_shaft_entry_settle|main_shaft_descent|main_shaft_dachora"
# Then re-record and confirm green integrity:
uv run python super_metroid/scripts/record/continuous.py --to spore --no-video
```

An implement card must:
1. Name exact lines: `spore_spawn_controller.py:254` (Patch A), `277-278` (B),
   `263-270` (C).
2. Guard settles with pose/x/y, not blanket `_hold`.
3. Re-record `--to spore` (or the segment probe) and diff the split's
   `green_brinstar_main_shaft` dwell vs. 2,806f.
4. Not touch `continuous.py` / `STATUS.md` / graph verification.

## 8. Residual / notes

- **Primary controller path:** `routes/spore_spawn_controller.py`
  `play_main_shaft_to_spore_spawn` (descent) + `play_parlor_to_main_shaft`
  (elevator entry). Not a `routes/kpdr/*.py` file.
- **Implement worth it?** Yes — Patch A alone targets ~35% of the split as
  idle wait and is low-risk (guarded settle). Combined with B/C it is the
  best single-split tighten of the K4-05 candidates after `business_to_warehouse`.
- **Report-only diff:** yes — no controller / continuous / STATUS / graph edits.
- **No savings claimed** without re-record; all bands are speculative.
