# Super Metroid Speedrun Plan: ZebesStart → Bomb Torizo

## Current Status (Feb 18, 2026)

**Route**: 12 segments, Landing Site → Morph Ball → Bomb Torizo
**Working**: 4/12 segments (landing_site, climb_descent, elevator_return, pit_room_descent@60%)
**Failing**: 8/12 segments at 0% despite 200k–5.3M training steps each
**Root cause**: Pure PPO with directional rewards cannot solve complex navigation

| # | Segment | Dir | Pass Rate | Training | Verdict |
|---|---------|-----|-----------|----------|---------|
| 1 | landing_site | LEFT | 100% | 50k | DONE |
| 2 | parlor_descent | DOWN | 0% | 500k | BLOCKED — route bottleneck |
| 3 | climb_descent | DOWN | 100% | 50k | DONE |
| 4 | pit_room_descent | DOWN | 60% | 50k | NEEDS POLISH |
| 5 | elevator_descent | DOWN | 0% | 250k | FAILING |
| 6 | morph_ball_collect | COLLECT | 0% | 20k | UNDERTRAINED |
| 7 | morph_ball_return | UP | 0% | 550k | FAILING |
| 8 | elevator_return | UP | 100% | 250k | DONE |
| 9 | pit_room_return | UP | 0% | 5.3M | STUCK — PPO plateau |
| 10 | climb_return | UP | 0% | 5.3M | STUCK — PPO plateau |
| 11 | parlor_to_flyway | RIGHT | 0% | 350k | FAILING |
| 12 | flyway_to_torizo | RIGHT | 0% | 350k | FAILING |

**Key lesson**: Segments 9 & 10 prove that throwing more PPO steps at hard
navigation doesn't work. The new platformer_common integration unlocks the
**record → hill climb** workflow that solved DKC levels.

---

## Strategy: Record → Hill Climb → Chain

The DKC experience proved: **1 human playthrough + hill climbing >> millions of PPO steps**
for navigation-heavy levels. The newly integrated platformer_common CLI now supports
SM segments directly:

```bash
# Record a manual playthrough
uv run python -m super_metroid_rl -l climb_return play

# Verify the recording (check room transitions)
uv run python -m super_metroid_rl -l climb_return verify --actions recording.json --trace

# Hill climb to optimize
uv run python -m super_metroid_rl -l climb_return hillclimb --seed recording.json --iterations 2000

# Watch the result
uv run python -m super_metroid_rl -l climb_return watch --actions hillclimb_best_final.json
```

---

## Phase 1: Quick Wins (1–2 hours)

Unblock the easy segments that are failing due to undertaining or simple fixes.

### 1.1 Record + verify all 8 failing segments manually

For each failing segment, record ONE successful playthrough:

```
uv run python -m super_metroid_rl -l parlor_descent play
uv run python -m super_metroid_rl -l elevator_descent play
uv run python -m super_metroid_rl -l morph_ball_collect play
uv run python -m super_metroid_rl -l morph_ball_return play
uv run python -m super_metroid_rl -l pit_room_return play
uv run python -m super_metroid_rl -l climb_return play
uv run python -m super_metroid_rl -l parlor_to_flyway play
uv run python -m super_metroid_rl -l flyway_to_torizo play
```

Each recording takes 30–90 seconds. Verify each with `--trace` to confirm
the room transition fires correctly. This also validates all 12 level configs
against real gameplay.

### 1.2 Verify all configs with selftest

```bash
for seg in landing_site parlor_descent climb_descent pit_room_descent \
           elevator_descent morph_ball_collect morph_ball_return \
           elevator_return pit_room_return climb_return \
           parlor_to_flyway flyway_to_torizo; do
    echo "=== $seg ==="
    uv run python -m super_metroid_rl -l $seg selftest
done
```

This catches any room_id mismatches immediately (the biggest risk from
hardcoded hex values in the config).

---

## Phase 2: Hill Climb All Failing Segments (2–4 hours)

Run hill climbing on each recorded seed. Start with the easiest wins.

### Priority order (easiest first):

**Tier A — Should complete quickly (short rooms, simple navigation):**
- `morph_ball_collect` — just walk right and grab item
- `elevator_descent` — ride elevator down
- `flyway_to_torizo` — short corridor right
- `parlor_to_flyway` — horizontal right, door at end

**Tier B — Medium difficulty (vertical platforming):**
- `parlor_descent` — ROUTE BOTTLENECK, vertical with some ledges
- `morph_ball_return` — upward with newly acquired morph ball

**Tier C — Hardest (tall vertical shafts, going UP):**
- `pit_room_return` — upward through Pit Room
- `climb_return` — upward through the Climb (tallest shaft)

### For each segment:

```bash
# Hill climb with 2000 iterations
uv run python -m super_metroid_rl -l SEGMENT hillclimb \
    --seed RECORDING.json --iterations 2000

# Watch the optimized result
uv run python -m super_metroid_rl -l SEGMENT watch \
    --actions hillclimb_best_final.json
```

**Expected frame counts** (from best_times.json where available):
- landing_site: ~200 frames (3.3s)
- parlor_descent: ~700 frames (11.7s)
- climb_descent: ~700 frames (11.7s)
- pit_room_descent: ~600 frames (10.4s)
- elevator_descent: ~340 frames (5.7s)
- morph_ball_collect: ~420 frames (7.0s)
- morph_ball_return: ~1740 frames (29.0s)
- elevator_return: ~200 frames (3.3s)
- pit_room_return: ~780 frames (13.0s)
- climb_return: ~3350 frames (55.9s) — longest segment
- parlor_to_flyway: ~500 frames (8.3s)
- flyway_to_torizo: ~500 frames (8.3s)

---

## Phase 3: Chain Verification (1 hour)

Once all 12 segments have working action sequences, verify the full chain.

### 3.1 Build a segment chain runner

Create a script that loads each optimized action sequence in order and plays
them back-to-back on a single emulator instance (no state reloads between
segments — the hill-climbed sequences must naturally transition rooms).

```bash
uv run python -m super_metroid_rl chain \
    --actions seg01.json seg02.json ... seg12.json
```

This requires a small `cmd_chain` addition to the runner (play actions for
segment 1, detect room change, immediately start segment 2 actions, etc.).

### 3.2 Measure total route time

Target: **complete ZebesStart → Bomb Torizo room entry**
Sum of estimated best frames: ~9,930 frames ≈ 165 seconds (2m 45s)
TAS comparison: ~80–100 seconds for this section
Realistic target: **under 4 minutes** (240 seconds / 14,400 frames)

---

## Phase 4: Speed Optimization (ongoing)

Once the route completes reliably:

### 4.1 Per-segment hill climbing refinement

Run more hill climbing iterations on the slowest segments:
- `climb_return` (55.9s) — biggest time save potential
- `morph_ball_return` (29s) — second biggest
- `parlor_descent` (11.7s) — route bottleneck, optimize transitions

```bash
uv run python -m super_metroid_rl -l climb_return hillclimb \
    --seed best_so_far.json --iterations 5000
```

### 4.2 Segment boundary optimization

Adjacent segments may waste frames at room transitions. Record transitions
where Samus enters the door and the next segment picks up mid-animation.
Trim leading/trailing NOOPs from each segment.

### 4.3 PPO fine-tuning for stochastic robustness (optional)

If deterministic hill-climbed sequences are too brittle (fail when chained
due to tiny frame offsets), use the optimized sequences as PPO training
demos (behavioral cloning seed) to get a more robust policy.

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Room ID mismatch in configs | Selftest fails, can't run | Selftest catches immediately; fix hex values |
| Hill climber can't solve upward segments | Climb_return/pit_room_return stuck | Record longer seeds, increase iterations to 5000+ |
| Segments don't chain cleanly | Route breaks between rooms | Record segments from actual room entry (not [from X] states) |
| morph_ball_collect needs item-bit check | Completion detection wrong | May need custom completion signal (not level_id_change) |
| SM-specific death detection edge cases | False deaths or missed deaths | health_zero signal is simple; validate with selftest |

---

## File Reference

```
# CLI commands
uv run python -m super_metroid_rl list-levels          # show all 12 segments
uv run python -m super_metroid_rl -l SEGMENT selftest  # validate config
uv run python -m super_metroid_rl -l SEGMENT play      # record manual run
uv run python -m super_metroid_rl -l SEGMENT verify --actions X.json --trace
uv run python -m super_metroid_rl -l SEGMENT hillclimb --seed X.json --iterations 2000
uv run python -m super_metroid_rl -l SEGMENT watch --actions X.json
uv run python -m super_metroid_rl -l SEGMENT trace-map --trace X_trace.json --area crateria -o out.png

# Key files
platformer_common/levels/super_metroid.py        # 12 segment configs
platformer_common/runner.py                      # CLI + trace collection in watch/replay
super_metroid_rl/navigation/trace_renderer.py    # position trace → area map overlay
super_metroid_rl/maps/                           # area PNGs (crateria, brinstar, etc.)
super_metroid_rl/train_curriculum.py             # PPO training (complement to hill climb)
super_metroid_rl/custom_integrations/            # 70 save states
super_metroid_rl/models/                         # PPO checkpoints
super_metroid_rl/world_map.json                  # room ID reference
```

### Trace & Map Workflow

The `watch` command now saves a `_trace.json` alongside the actions file on exit,
containing per-frame position data (room_id, x, y, health, buttons), room visit
summaries, and center-of-gravity stats.

```bash
# 1. Watch a segment (ESC to quit → trace saved automatically)
uv run python -m super_metroid_rl -l sm_parlor_descent watch --actions best.json

# 2. Render trace on area map
uv run python -m super_metroid_rl -l sm_parlor_descent trace-map \
    --trace best_trace.json --area crateria -o parlor_traced.png
```

The trace JSON is also structured for AI analysis (per-frame coords, room transitions).
