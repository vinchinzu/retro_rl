# Handoff — 2026-09-11 reactive combat tooling + measured Clean tip

No STATUS claim. `route_eligible=false` throughout. No pokes, no assists,
no `--infinite-life`, no state restores. Four commits on `main`.

## Read this first

`uv run python nes/zelda_i/scripts/clean_tip.py` is the ladder of record.
It prints the tip, the next open hop, and blockers grouped by root cause.
**`tip()` returns `None`** — the power-on run is red at the first row.

## What changed

| Commit | What |
|--------|------|
| `9ea8c84f` | `dungeon/{tracking,threat,postmortem}.py`, `spine/clean_tip.py` + CLI, L6 `0x78` wiring. Also carried prior sessions' uncommitted tree state. |
| `3f90a42c` | Three real bugs in the above, found in review. |
| `64ad1956` | M5 Clean measured RED; ladder corrected. |
| `60e50744` | Occupancy grid stops learning walls from live bodies. |

Suite: **1320 passed, 3 deselected**
(`QT_QPA_PLATFORM=offscreen uv run pytest nes/zelda_i/tests -q`).

Full reasoning and ROM evidence:
[`rr-npv-reactive-combat.md`](rr-npv-reactive-combat.md).

## The three tools, and when to reach for them

- `dungeon/tracking.py` — `ObjectTracker`: slots → tracks with velocity and
  hazard class. Motion-first classification, so an unrecognised type at
  shot speed is a shot whatever its HP byte says (live L6 `0x59` carries
  `hp=128`). Tracks reset on `(level, screen)` change.
- `dungeon/threat.py` — time-to-contact per candidate button; `ReactiveEvader`
  with commitment and a reverse-lock; `in_firing_line` / `off_line_step` for
  the pre-emptive case.
- `dungeon/postmortem.py` — `DamageLog`. Wired into **both**
  `GenericDungeonRoomController` and `HopController`, so
  `controller.report()["damage"]` gives hits-by-cause and a `death_cause`
  line on every room and dest hop.

**The measurement that retires a whole class of work:** Link walks 1 px/frame,
so a sidestep needs `MIN_DODGE_BODY`=16 / `MIN_DODGE_SHOT`=12 frames of
warning. `threat.dodgeable(impact)` is `False` below that. When it is False,
no position table can work at that pose — the answer is the sword, the
shield, or a stand cell that was never on the axis. Check it before tuning.

## Open, in priority order

1. **L1 `0x33` `clear33_key`** — blocks M5, the whole Clean campaign.
   Link stalls at `(88,165)`, deaths 0, only 1940 of 6000 frames are combat
   frames. Lead: `Room33ScoopController._scoop_if_low` (`level1/dungeon.py`)
   walks a naive 4-way delta toward the heart drop with no occupancy
   awareness, unlike `engine._scoop_heart`; a drop behind an obstacle walks
   into a wall forever and the clear condition is never re-checked. A human
   watching an earlier run said: "drop south to y=173 first, then RIGHT onto
   the key tile." **Unverified — dump the `$6530` tile map before believing
   any of it.** An agent was started on this and stopped before writing
   anything; nothing to recover.
2. **L1 `0x23` combat throughput** — after the `60e50744` pathing fix,
   misses fell 4627 → 4082 (−11.8%) and the stage still times out. The live
   number: Link kills **1 of 3 Goriyas in 6000 frames while taking zero
   damage** (`hits: 0` both before and after). Never hit, never killing, is
   a failure to *engage*, not to navigate — look at the `should_swing_at`
   hitbox gate and `engage_distance`, and instrument attacks-landed vs
   attacks-attempted. This is the best-posed open question in the tree.
3. **L6 `0x78`** — `rr-d6v`'s next lever is **`level6_east_key_0x7a`, not
   `0x78`**. The damage census shows 6 of 7 hits land in that *green* stage;
   `0x78` is where Link runs out, not where he loses the health. Do not
   re-run the v4–v9 east-waist chase.
4. L5 `0x77` (`body_undodgeable`), L8 `0x1E` (`shot_undodgeable`),
   L7 bait (`inventory_gap`) — see the ladder.

## Housekeeping

- `stash@{0}` "verify-m5-baseline-stash" is **stale and safe to drop**:
  `git diff stash@{0} 9ea8c84f -- nes/zelda_i` is zero-deletion, so it holds
  nothing unique. Left in place only because dropping renumbers `stash@{1}`
  ("watcher-temp harvest/beads"), which belongs to another session.
- `snes/alttp/refs/z3-json-data` has three deleted PNGs inside it, unrelated
  to this work. `git -C snes/alttp/refs/z3-json-data checkout .` if
  unintended.
- `scratch/` (39 one-off probe scripts) was committed at the user's request
  in `9ea8c84f`, against `CODING_STANDARDS.md`'s delete-on-sight rule.
  `git rm -r --cached scratch/ && echo 'scratch/' >> .gitignore` to undo.
- **Shared-tree hazard:** a `git stash -u` during this session swept three
  concurrent agents' untracked work. Use `git worktree add` for baseline
  comparisons. Never stash, `checkout -- .`, or `reset --hard` in this tree.

## Method note worth keeping

Read the per-stage damage census **before** tuning the room that went red.
Twice in one sitting the health was being spent in a stage that reports
green. A green/red stage table cannot show that; one census can.
