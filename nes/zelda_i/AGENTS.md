# Agent Instructions — zelda_i

NES Legend of Zelda. The verified Clean power-on gate is the L8 Triforce
(`clean_poweron98`); L9 / credits is the active frontier (`docs/plan.md` ladder).
Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/PRE_L1.md`.
Session: `.grok/skills/zelda-session/SKILL.md`.
Tracker: `bd ready -l zelda_i -l spine`. Living residual: `docs/PRE_L1.md`.

## Tracks

Survival health refill is not Clean. `--through pre-l1` forces that assist off.
Gathering is the spine's default prefix (`spine/survival.py` `_run_gathered_prefix`):
pre-l1 assist-off, gather chain under its own refill (`--gather-engage-hearts`,
default 1 = last-heart; 0 is the next rung), Blue Ring and later Bait
purchases at 0x34, then L1 from the 0x37 door. The ring is paid from hidden rupee caves
(`SECRET_RUPEE_CAVES`); nothing writes the wallet, and it caps at 255. Ringless L1 and later saves are obsolete on the
main spine; `--resume` rejects them. Regenerate from power-on.
Planner owns `docs/STATUS.md`. The 18909f wooden M5 result is historical:
the 2026-09-24 natural-entry recheck fails 2/2 at L1 `clear33_key`.
Clean re-measure is `run_level1_complete` without `--infinite-life`.

## Save points

`run_survival_spine.py --save-points` writes `Spine_<stage>.state` at every
stage start and `Spine_fail` on a red run; `--resume <stage>` loads one and
plays on (disclosed as `resumed_from`). A resume proves one pose only: re-run power-on continuously before claiming a stage, and fix a continuous failure from its `Full_<stage>` point (load with no idle frame first). Custom suffixes (L3/L4 boss, L5
whistle/TF) have no save point; resume from the stage before. Each stage
reports `hearts` (in/out/damage/damage_by_room).

## Commands

```bash
bd ready -l zelda_i -l spine
uv run python nes/zelda_i/scripts/run_survival_spine.py --no-video --trials 1              # default: gather → L1 TF
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level9-credits --save-points Full --no-video --trials 1  # continuous power-on → credits (~1h); own prefix keeps Spine_*
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level7 --save-points R25 --resume level7_post_l6_overworld --no-video --trace /tmp/tape.json  # one level from a save point, per-frame tape
uv run python nes/zelda_i/scripts/run_metrics.py nes/zelda_i/recordings/<tag>.json   # docs/RUN_METRICS.md row
uv run python nes/zelda_i/scripts/pin_probe.py <state> --tiles --items --press DOWN:40   # pose, objects, $6530 + lattice, item flags; --fixture writes tests/fixtures
uv run python nes/zelda_i/scripts/pin_probe.py <state> --doors   # ROM door type per side (open/bomb/key/shutter) for this dungeon room and its neighbours
uv run python nes/zelda_i/scripts/stage_replay.py <state> zelda_i.level6.dungeon:ROOM_29_SPEC --assist --window A-B   # one controller/spec from a save point; --idle N = RNG offset
uv run python nes/zelda_i/scripts/run_survival_spine.py --clean --through level7 --save-points C9 --no-video --trials 1   # C9: Clean power-on through L7 (rr-rgum). The gate today is L6.
uv run python nes/zelda_i/scripts/run_survival_spine.py --engage-hearts 1 ...        # refill at the last heart only: refills = deaths prevented
uv run python nes/zelda_i/scripts/run_survival_spine.py --engage-hearts 1 --observed-damage-guard ...  # keep last-heart target; safety refill after larger observed hits
uv run python nes/zelda_i/scripts/run_survival_spine.py --through gather --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --no-gather --no-video --trials 1   # legacy wooden-sword prefix
uv run python nes/zelda_i/scripts/run_survival_spine.py --through pre-l1 --no-video --trials 1
uv run python nes/zelda_i/scratch/eval_gather_clean.py --state CL61_walk_2c --start walk_2c --stop walk_37 --offset 3   # Clean gather stages from a pin, one RNG offset; hits by cause
uv run python nes/zelda_i/scratch/offset_pins.py CL64_enter_level3 L3o enter_level3 8   # L3o<n>_enter_level3 for --resume per offset
uv run python nes/zelda_i/scripts/clean_tip.py
uv run python -m zelda_i.overworld.gather_segments pin     # power-on pre-l1 → PreL1BombLeave
uv run python -m zelda_i.overworld.gather_segments chain   # → 0x37, 6 HC + White Sword + Blue Ring; chain:<stage> resumes
uv run python nes/zelda_i/scripts/run_level1_complete.py --natural-entry --trials 2
uv run pytest nes/zelda_i/tests/test_pre_l1.py nes/zelda_i/tests/test_clean_tip.py -q
uv run pytest nes/zelda_i/tests -q
```

Leave proof is RAM plus `zelda_i.screen_glance`, with `--no-video`.
`--rollout` on the pre-l1 walk is opt-in. It does not change the default arm.

## Traps

- This prefix buys bombs at coast `0x6F` in `overworld/shop_p7.py`. The later arrow cave is inland `0x4A`. Do not join the shop through `0x68`, the `0x5C` maze, or candle `0x5E`.
- `0x79` east is y=165 only. The hop that leaves `0x7D` carries `SCREEN_7E_EAST_BAND`, 137 to 145. `0x7B` and `0x7C` stay any-row. Do not put any-row back on the `0x7D` hop. y=133 on `0x7E` does not scroll.
- Stop is `ADDR_BOMBS >= 1`. While the wallet is under 20 the coast hunt stays open so the walk can arrive over the price. A short arrival on `0x6F` still ends the walk. `overworld/topup.py` hunts north `0x5F`, then west `0x6E`. Still short, it hunts the nearest coast screen `RoomHistory` has dropped — not a transit screen, not inland — and comes back. It does not finish while short. The buy runs only once the wallet can pay. Do not lap west as the walk. `laps` stays 0.
- Bombs and keys are natural from power-on through L8. `BombRestockController` (`overworld/bomb_shop.py`) buys a 20R pack at 0x4A before L2 and at 0x44 on the L4 and L8 walks only when the count is short of the next dungeon's `want`; a reshuffled drop skips it. A red wall is a short `want` or a missing restock, never a new top-up. A placed last bomb reads `bombs=0` while it burns: guard on bombs before placing only.
- Do not poke Food, bombs, keys, rupees, the candle, or `$066F`. `--through pre-l1` still forces heart assist off; a short wallet at `bomb_topup` hunts the coast (`overworld/topup.py`). Quote a tape only against the code that produced it. Read `reason_by_screen` before changing a hop.
- After the ring, 0x62's 100R pays for Bait on a second 0x34 visit. Its payout is still counting when `exit_62` ends; the bait controller checks 60R at the shop. `ring_return` skips heart scoops because a detour into the already-open 0x56 cave strands the walk. The pre-L1 red potion was displaced by Bait; do not assume a carried potion at L1.
- Secret caves: scan slot 11 on a `BFS_<screen>` pin (0x63 rock, 0x64 tree), sweep stands with a what-if candle write, then add a `SECRET_RUPEE_CAVES` row. A candle flame DOWN from tree_y-27 opens; a cave's exit pose is per screen (0x62 lets Link out west of its bush column). Payouts past 255R are lost.
- 0x57 has a solid south tree row: reach 0x67's 30R rock from start 0x77 UP, via 0x58/0x68/0x78 on the post-L8 bomb-shop detour. The cave costs one bomb; the second 0x4A pack returns the bag to 7. Run 53 cleared L9 with 5 left.
- A `@dataclass` copies field defaults into `__init__`. Setting the default on the class later does not change instances.
- Walls come from `dungeon.tilemap.ow_walkable_nodes`, the ROM collision on the 8 px turn grid, not from a screenshot. The old `measured_walker` samples one pixel and misses Link's width. 0x79 y=165 dead-ends at x=192.
- Walls, doors, stairs and block pushes go through the ROM lattice helpers in `dungeon/hop_controller.py` (`LatticeDoorWalker`, `lattice_goto`, `block_push_step`, `stairs_step`); hand waypoint policies are fallbacks only.
- Every spine run prints a ledger (`spine/ledger.py`): frames per room visit, drop outcomes (picked/expired/left), missed hearts, flutter (1-2 px reversals), damage per room, dungeon room items left untaken (world flags `$06FF` L1-6 / `$077F` L7-9), and inventory rises (play vs written between frames). Read `flutter rooms` and `slowest visits` before tuning; record milestones in `docs/RUN_METRICS.md`.
- Link turns only on the ROM lattice (x%8==0, y%8==5). A press on the other axis slides him onto it first, so a greedy "bigger axis first" step, a 1 px sidestep, or an exact-pixel stop flips every frame. Walk with `room_step` / `mouth_step` / `LatticeDoorWalker` / `exit_door` / `stairs_step`; one helper owns a goal end to end (a lattice approach plus a separate pixel finish is a tug-of-war).
- On a deployed stepladder (0x5F) only its axis moves; wrap hand presses in `release_action`. The ladder sprite sits 3 px below Link's row. The dock raft (0x61) and stepladder are not combatants.
- A room's wave spawns a few frames after the scroll: wait for it before counting live enemies (`_fight_if_live`).
- Do not edit runtime modules while a spine run is in flight: level modules import lazily, so a run picks up half-edited code (power-on 6 died on an `ImportError` after `ram.py` changed mid-run).
- Water rooms: a dungeon clear whose exit is on Link's bank sets `reachable_only`; the ladder is owned by `LadderEscape` / goal-aware `ladder_release`; a patrol waypoint the tiles call water is moved to land (`_patrol_vertex`).
- Stop predicates and handoff checks must not require full hearts: only the Survival refill fills them, and a `--engage-hearts 1` run stalls on them (L6 0x1C heart, L7/L8 leaves). Check the item or container byte, not a pose held N frames.
- Replay a last-heart failure with `stage_replay.py --last-heart`, never `--assist`: the full refill keeps the beam firing and hides it.
- A second worktree is not isolated: the venv's `retro_rl_paths.pth` imports the main tree's `zelda_i`. Launch with `PYTHONPATH=$W:$W/snes:$W/nes` from the worktree `$W`, or main-tree edits land in the run.
- The sword shot appears 13 frames after the A press (blade states 1 then 2). A 9-frame A cadence makes it look like 4.
- An approach waypoint must be a lattice node, or the lattice approach is skipped for a hand press (L8 0x4C (120,109): 8000f). Never hand an unstick rung an idle that waits for `stuck` to fall: idling keeps it rising.
- `--clean` was stripped from `sys.argv` by a `level3/spine.py` import until 2026-09-24: every earlier `--clean` spine tape is Survival. A module must never edit `sys.argv`.
- `defend=True` on an `OverworldPathController` runs `ScreenHunter.defend` (strike/peel/shield/duck) ahead of the hop ladder and every hand phase. A subclass that latches a pose (a bomb cell) must drop the latch in `_on_defended`: a 2 px duck left 0x47's flame stand latched and the tree stayed shut.
- A clock drop sets `$066C` (`snap.clock`) until Link leaves the room. Goriyas freeze where they stand and L7 0x0D's Wallmaster ring stops spawning. When a hand clear times out on bodies that do not move, check the clock first.
- Every stage report has `hits` (DamageLog): cause, action, and a 24-frame trail per hit. Read it before tuning: the 2026-09-25 "close peel" hits were swings pinned 13 frames, not the peel.
- The hunter's peel is `common.body_escape` (every input flown on the lattice), its swing waits for `_swing_pays` (blade out on frames 4-11 of the 13-frame pin), its Zora duck is `shot_escape`. A candidate that leaves the hunt box is not a candidate: a peel off 0x63's edge scrolled to 0x53 and lost the hop.
- A bomb or burn cell off the turn lattice (0x48's x=188) is reached by a straight press along the shared row (`_nudge_dir`); `room_step` alone flips around it.
- A potion drink can leave B on the potion. Burn/bomb cells reselect their `b_item` before pressing; a B press on the potion slot drinks it.
- Darknut rooms opt into `CombatTuning.flank_shielded`: stand 16 px off a side the shield is not on, commit to the pick (`FLANK_COMMIT_FRAMES`); a per-frame re-pick flipped 2 px for 28000 frames.
- Boss loops that step the env themselves (L3 suffix `_tick` / `_drive_hop`, L4 Gleeok) sit outside `run_controller_stage`'s potion guard: call `dungeon.pause_select.drink_if_low` every frame. `--save-points` writes `<prefix>_level3_manhandla` for `scratch/mh_eval.py`.
- Offset pins (`scratch/offset_pins.py`) must sit after any settle that waits for a game event: idle frames before the L3 TF settle came out as one tape shifted by a frame.
- Read the door table (`pin_probe.py --doors`) before adding a dungeon clear. An `open` or `key` exit needs no fight, while a shutter or a push block needs every enemy dead (L6 0x38/0x09 blocks do not move otherwise). L6 fought 0x78, 0x28 and the 0x18 Gleeok for exits that were open or reachable by bomb.
- The Magical Sword detour (`overworld/magical_sword.py`: coast hearts 0x5F/0x2F after L4's 0x67 rock, the 0x21 grave on the L6 walk) and 0x13's rupee rock skip as a unit through `spine.hops.GatedLeg`. The grave's Old Man needs 12 containers, so a pin that skipped the coast hearts reaches L6 with the White Sword. Decide a route lever with a what-if pin first (`--set` RAM at load, a measurement only), and delete those pins afterwards.
- Score a combat change on the multi-offset eval, not on one tape (`stage_replay.py --idle`). A dungeon reroute that touches a room M5 uses (0x23, 0x33) must be re-run against M5's 18909f.

## Pointers

[docs/PRE_L1.md](docs/PRE_L1.md) · [docs/STATUS.md](docs/STATUS.md) ·
[docs/plan.md](docs/plan.md) · [docs/ASSIST_CONTRACT.md](docs/ASSIST_CONTRACT.md) ·
[docs/HYGIENE.md](docs/HYGIENE.md).

Files under `docs/tasks/` are lane notes the clean-tip ladder still cites.
They are not this sitting.
