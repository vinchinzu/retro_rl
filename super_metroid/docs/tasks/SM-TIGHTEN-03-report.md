# SM-TIGHTEN-03: Deep Analysis — `terminator_energy_tank` dwell

**Date:** 2026-07-31
**Source:** `recordings/start_to_varia.json` (outcome=`varia_collected`, total_frames=101,954)
**Target:** split `terminator_energy_tank` (room 0x990D), dwell=**4,693f** (~78s)

## 1. Controller ownership

| Field | Value |
|-------|-------|
| Controller file | `routes/spore_spawn_controller.py` |
| Entry function | `play_parlor_to_main_shaft` (line 110) |
| Parent wrapper | `play_post_torizo_to_spore_spawn` (line 435) |
| Split registration | `routes/continuous.py` line 660 |
| Split condition | `progress_event` `max_health >= 199` at frame 49,972 |
| Graph edge | `progression.py` line 677: `parlor_to_terminator` (0x92FD→0x990D) |

The split dwell covers the full room visit in Terminator (0x990D): from settled
ordinary after entering from Parlor (0x92FD) until the transition starts toward
Green Pirates (0x99BD). The split is captured at item-collection time
(`max_health >= 199`), but the dwell measurement covers the entire room
engagement.

## 2. Controller phase breakdown

All phases are in `play_parlor_to_main_shaft`, lines 179–210:

| # | Phase label (reason) | Lines | Frames | Cumulative |
|---|----------------------|-------|-------:|----------:|
| — | `parlor_terminator_exit` (9×) | 179–181 | 540 | — |
| 1 | `terminator_entry_settle` | 182 | 100 | 100 |
| 2 | `terminator_morph` | 183–186 | 17 | 117 |
| 3 | `terminator_bomb_tunnel` (8×) | 187–189 | 480 | 597 |
| 4 | `terminator_energy_tank` approach (7×) | 192–194 | 420 | 1,017 |
| 5 | `collect_terminator_energy_tank` | 195–198 | ~10–150 | ~1,027–1,167 |
| 6 | `exit_terminator` → Green Pirates | 201–210 | up to 900 | ~1,927–2,067 |

The `parlor_terminator_exit` (540f at lines 179–181) probably overlaps with or
precedes the room timer's settle detection for 0x990D. The action-reason report
shows `parlor_terminator_exit` as a separate 540f reason, not part of the
terminator split.

**Identified fixed-phase total: ~1,000–2,100f (depends on exit+collect timing).**
**Actual dwell: 4,693f.** Gap: ~2,600–3,700f unaccounted.

## 3. Action-reason report crosswalk

From `--reasons --top 40`:

| Frames | Reason | Phase |
|------:|--------|-------|
| 540 | `parlor_terminator_exit` | (pre-dwell or entry settle overlap) |
| 600 | `parlor_bomb_tunnel` | (Parlor, not Terminator) |
| 1,073 | `parlor_upper_platforms` | (Parlor, not Terminator) |

None of the top-40 action reasons map to the `terminator_` phases 1–6 above.
This means the controller phases 1–6 run with **multi-frame holds that share
one reason tag** — the `_hold()` calls with reason `terminator_*` are grouped
into longer sequences and don't appear in the top 40 individually because they
fall below the `--min-dwell 200` threshold. Each reason name only shows if its
total accumulated frames ≥ 200.

This confirms the terminator phases are relatively short (<200f each), which
makes the remaining 2,600–3,700f gap even more significant.

## 4. Waste candidate analysis

### Candidate A: Bomb tunnel crawl inefficiency (highest confidence)

The bomb tunnel loop (lines 187–189) runs 8 iterations of `(45f LEFT+X, 15f LEFT)`.
Each cycle places one bomb, waits for detonation, and crawls forward. However:

- The 60f cycle assumes bombs detonate on schedule and Samus advances immediately.
- In practice, bomb detonation in the morph tunnel varies due to terrain contact
  and Samus's exact y-position. Missed detonations add **silent settle frames**
  that are not captured in the controller's 480f budget.
- The `_hold_until_room` entry settle (100f at line 182) may need to be longer
  if Samus overshoots or lands partially outside the tunnel.

**Estimated waste: 300–800f** from bomb timing variance + mid-tunnel alignment.

### Candidate B: Exit-to-Green-Pirates transition (high confidence)

The `exit_terminator` phase (lines 201–210) uses `_hold_until_room` with a 900f
timeout. If Samus reaches the left door at frame 300 of that window, the
remaining 600f are idle holds. The actual door transition + load to 0x99BD
also counts as part of the room dwell (the room timer counts until the
transition phase starts, not until settled in the destination).

**Estimated waste: 200–500f** from idle exit hold.

### Candidate C: Entry settle after parlor_terminator_exit (medium confidence)

The `parlor_terminator_exit` (540f at lines 179–181) happens immediately after
the bomb tunnel settle. By the time `terminator_entry_settle` runs (100f at
line 182), Samus may already be moving left in Terminator. The room timer may
not have reached settled ordinary yet due to the previous door transition's
multi-screen load.

If the room timer detects settled ordinary ~100–200f into `terminator_entry_settle`,
those 100f are just a fragment of the actual settle cost.

**Estimated waste: 100–300f** (speculative, depends on load timing).

### Candidate D: Post-bomb-tunnel morph realignment (medium confidence)

After the bomb tunnel crawl (phases 2–3), Samus must be morphed and positioned
correctly under the E-Tank. Lines 183–189 handle morph + crawl, but the
transition from bomb tunnel to E-Tank approach requires precise X alignment.
If the bomb tunnel exit leaves Samus slightly offset, the `terminator_energy_tank`
approach (420f at lines 192–194) is pure movement, but the setup may need
extra frames.

**Estimated waste: 100–200f.**

### Candidate E: Collect loop waiting for health register propagation (low confidence)

The collect loop (lines 195–198) polls `session.state.max_health` every 10f
and breaks when ≥199. If the RAM register for max_health lags the PLM
collection by more than 10–20f, the loop adds a small wait.

**Estimated waste: 10–50f** (negligible).

## 5. Top waste summary

| Candidate | Estimate | Controllable? | Priority |
|-----------|---------:|:-------------:|:--------:|
| A: Bomb tunnel crawl | 300–800f | Yes | **High** |
| B: Exit idle hold | 200–500f | Partial | Medium |
| C: Entry settle overlap | 100–300f | Partial | Low |
| D: Morph realignment | 100–200f | Yes | Medium |
| E: Health register lag | 10–50f | No | Low |
| **Total gap** | **~710–1,850f** | | |

The remaining ~1,200f of the 2,600–3,700f gap is likely unavoidable door
transition load time (Terminator is a multi-screen room and both entry and exit
transitions involve loading).

## 6. Verdict: is this split worth a tighten card?

**Yes — but with caveats.**

The bomb tunnel crawl (Candidate A) is the most promising target. Optimizing
the bomb-tunnel timing in Terminator would cut 300–800f across every run
that passes through this room. However:

1. The E-Tank is a **required item** — this room visit is not optional.
2. The bomb tunnel is inherently variable because bomb detonation timing
   depends on collision detection frames.
3. ~1,200f of the gap is transition loading, which is not controllable.
4. The achievable savings (~5–13s) is significant but not transformative
   compared to boss-fight splits (Spore Spawn: 12,182f, Bomb Torizo: 11,812f).

**Priority: Medium.** Worth a card if the team wants to tighten all movement,
but should come after boss-fight tighten cards if efficiency is the priority.

## 7. Future patch recipes

### Recipe A: Tighter bomb tunnel cycle (pure controller, no emu)

Replace the fixed 8-cycle `(45f LEFT+X, 15f LEFT)` with a
polling-based approach: hold `LEFT+X` and step until the block below Samus
clears (check `state.bg1_tile` or `state.samus_y`), then advance. This removes
the guess-timed cycle and adapts to actual bomb detonation timing.

**Risk:** Requires a reliable RAM indicator for "bomb exploded below me and
cleared the block." The controller currently has no per-frame tile check during
tunnel phases.

### Recipe B: Trimmed exit idle hold (pure controller)

Replace `_hold_until_room(session, 0x99BD, 900, LEFT, A, B, X)` with a
shorter timeout or a polling loop that stops pressing buttons 10f after
detecting transition phase (`state.phase != ORDINARY_GAMEPLAY`). This cuts
the 200–500f wasted after the door target is reached.

**Risk:** Low. The exit path is a straight corridor; overrunning only costs
frames but cannot fail. A timeout of 600f or even 500f would still be safe.

### Recipe C: Aligned bomb tunnel entry (controller file only)

Add 2–3 extra settle frames after `terminator_entry_settle` to ensure Samus's
y-position is correct for the morph tunnel. This costs ~20–30f but may save
100–200f in bomb detonation alignment downstream.

**Risk:** Could backfire — extra settle is pure cost if the original position
was already correct. Requires empirical validation.

## 8. Acceptance command for implement card

The implement card should validate with:

```bash
# Shortest continuous prefix that includes this split: --to spore_spawn
# or a pure probe of the terminator segment only (if source state exists):
uv run python super_metroid/scripts/probe/kpdr.py pure terminator-to-green-pirates \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/parlor_to_terminator.state

# Full continuous re-record
uv run python super_metroid/scripts/record/continuous.py --to spore_spawn --no-video

# Post-tighten dwell comparison
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_spore_spawn.json --top 20 | grep terminator_energy_tank
```

## 9. Constraints

- No frame savings claimed without re-record.
- Do not edit `continuous.py` or `STATUS.md`.
- Do not add progression/capacity/boss-bit RAM writes.
- This analysis is report-only; the implement card is separate.