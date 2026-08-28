# Review handoff — generalist overnight (do not train)

Read-only review of the contractor training stack before any overnight PPO.
Write findings; do not start `overnight`, do not edit `STATUS.md`, do not
touch `DEFAULT_CONTINUOUS_TIP`. Practice ROM only. ADR:
[`docs/adr/0009-generalist-contractor.md`](../adr/0009-generalist-contractor.md).

**Done when** the review names a single verdict (`fix-first` / `run-overnight` /
`reject-plan`) plus every issue below as pass, fail, or n/a, with a file:line.

## Why this exists

The 200k Crateria mix (`models/generalist/ppo_crateria.zip`, 16 hops, 4-ep
eval) matched the walk-to-xy teacher at 1/4 Join. The old next step was freeze
`kpdr25/crateria/ship` → parlor on Landing. That hop is same-room floor-left
and the teacher already Joins it, so a green there would not prove a general
agent and would hide the two bugs that actually explain the table.

Rejected plan: ship-only PPO until 08:00.

Replacement (unstarted; a ~90s premature kickoff was killed): occupancy from
editor collision, door potential when the Join pin is in another room,
per-session eval N=8, two PPO seeds × 8 envs, same-room Crateria first, mix
only if Join beats the teacher.

## Scope (read these)

| File | Role |
|------|------|
| `snes/super_metroid/generalist/solid.py` | Editor 16px grid; door=9 walkable; `potential_xy` |
| `snes/super_metroid/generalist/env.py` | `observe(..., solid=, steer_*)`; `_distance` uses potential |
| `snes/super_metroid/generalist/obs.py` / `goals.py` | 226-dim unchanged; dx/dy can steer at a door |
| `snes/super_metroid/generalist/evaluate.py` | Per-session Join; occupancy-aware heuristic |
| `snes/super_metroid/generalist/train.py` | `--same-room`, `--n-envs`, SubprocVecEnv fork, BC |
| `snes/super_metroid/generalist/overnight.py` | Population, discard, same-room → mix |
| `snes/super_metroid/generalist/corpus.py` | `same_room` / `session_ids` |
| `snes/super_metroid/tests/test_generalist_*.py` | ROM-free + one rom-marked env smoke |
| `snes/super_metroid/models/generalist/overnight/PLAN.json` | Operator contract |
| `snes/super_metroid/models/generalist/train_crateria.json` | 200k mix numbers (baseline, not a tip) |

The `generalist/` package is untracked. Diff vs HEAD is the whole tree, not a
patch. Editor grids live **outside** the repo:

`../snes_editor/super_metroid_rl/super_metroid_editor/export/sm_nav/rooms/room_*.json`

(`SUPER_METROID_EDITOR_NAV` override). Bank `$7F` clipdata is **not** on this
practice core (WRAM block is `$7E` 128KiB; `$7F6402` was zeros).

## Evidence already taken (do not re-train to re-learn this)

- Crateria trainable: 16 hops, **4 same-room** (ship, parlor_downback, morph,
  bomb_torizo), **12 cross-room**.
- Ship pin (1153,1146) `0x91F8` → parlor (121,1179) **same room**. Parlor
  session then Goals parlor_downback in `0x92FD` — xy potential pulls **right**
  on Landing, away from the left door.
- Live `observe` before the fix: occupancy all zeros (`solid=` never passed).
- After the fix, ship reset: occupancy max 1.0, 64 nonzero cells, steer
  dx≈−2.02 (left). Landing editor grid 144×80, 16 door tiles.
- `uv run pytest` on `test_generalist_{solid,overnight,obs,goals,corpus}` →
  18 passed. `test_generalist_env` is `@pytest.mark.rom` (deselected by default).
- Premature overnight (~20:33 local) did spawn 2×8 SubprocVecEnv children
  (fork worked). Killed. 45m scheduler cancelled. **No `best.zip`, no
  `status.json`.** Worker logs under `models/generalist/overnight/` are noise.

## Review steps

1. **Occupancy is actually in the 226-vector during `env.step`.** Trace
   `GeneralistEnv._observe` → `load_room_solid` → `RoomSolid.is_solid` →
   `occupancy_grid`. Door clip 9 must stay walkable (else the policy is trained
   to treat the Join as a wall). OOB as solid. Completion: cite the call chain
   or a hole.
2. **Dense reward is not anti-door.** `_distance` + `potential_xy`: same-room
   uses Join xy; other-room uses nearest clip-9 in **this** room. Check the
   leftover `+ 256` when rooms differ — constant offset vs spike on room-enter.
   Completion: say whether the parlor hop (start on Landing door, Goal in
   `0x92FD`) is pulled through the door or still rightward.
3. **Goal tail vs Join pin.** `goal_vector` still stores absolute Join xy;
   only dx/dy steer. Confirm Join (`is_join` / `LeaveSpec`) is unchanged.
4. **Eval can reject a coin-flip.** `eval_per_session` must run every corpus
   row (not 4 mixed episodes). N=8 on 4 same-room hops is 32 episodes — still
   noisy at a 0.05 promote margin (~1–2 Joins). Say if the margin is usable.
5. **Curriculum is not ship-only.** Same-room set is four hops, not Landing.
   Morph / bomb_torizo may need morph or a fight; if the teacher Join is ~0
   there, “beat the teacher” is a weak gate. Call that out.
6. **Overnight keep/discard/promote.** `decide_keep` vs the phase switch in
   `run_overnight`. Worker-level `"promote"` is not what flips `same_room`.
   Discard must not delete `best.zip`. Mix resume from best same-room zip on
   worker 0 only — confirm. Occupancy-empty abort is **not** implemented in
   `overnight.py` (only in the killed manager prompt). That is a hole if the
   editor path is missing.
7. **Compute plan.** 2 jobs × 8 envs = 16 emulators on a 16-core / 32-thread
   box. Cycle 1 still does serial heuristic+BC on each parent before the vec
   (duplicate 32-ep eval). Subproc start_method=`fork`. Look for pickle,
   already-open emulator, torch thread oversubscribe (`torch.set_num_threads(1)`
   is in `train`, not in the overnight parent).
8. **Contract / hygiene.** OBS_DIM stays 226. Reward-contract digest **did**
   change (text). Old `ppo_crateria.zip` is a different reward. No STATUS.
   Editor JSON is an undeclared runtime dep.

## Known doubts (confirm or dismiss)

- Static editor collision, not live clipdata (shot/crumble/door PLMs stale).
- Nearest door is the nearest clip-9, not the door that leads to the Goal room.
- Heuristic only looks at the occupancy row through Samus, then jump-left/right.
- BC still clones that heuristic. If the teacher is still weak on morph/bomb,
  BC poisons the net.
- `heuristic_join <= 0` → always `keep` (never discard). A broken eval that
  reports 0 Join freezes a bad seed.
- No `--baselines-only` path; the cheap “does occupancy+teacher Join ship?”
  probe is not a first-class CLI.

## Commands (read-only)

```bash
uv run pytest snes/super_metroid/tests/test_generalist_solid.py \
  snes/super_metroid/tests/test_generalist_overnight.py \
  snes/super_metroid/tests/test_generalist_obs.py \
  snes/super_metroid/tests/test_generalist_goals.py \
  snes/super_metroid/tests/test_generalist_corpus.py -q

# optional, ROM: occupancy non-zero on ship reset
uv run pytest snes/super_metroid/tests/test_generalist_env.py -q -m rom
```

Do not run `overnight` or `train --timesteps` in this review.

## Verdict bar

- **fix-first** — occupancy, anti-door reward, Join semantics, or overnight
  discard would burn 11h. Name the patch.
- **run-overnight** — those four hold; remaining issues are cycle-time waste
  or noisy margins. Name the start command.
- **reject-plan** — occupancy+door-potential is still the wrong bet (e.g. the
  contractor cannot be a same-room walker, or editor grids are the wrong
  solid). Name the alternative, not “try ship-only.”
