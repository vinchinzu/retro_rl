# Agent Instructions — zelda_i

NES Legend of Zelda. Clean gate is M5: power-on to the Level 1 Triforce.
The next open row is the gathering prefix, `pre_l1`, not Level 2.
Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/PRE_L1.md`.
Session: `.grok/skills/zelda-session/SKILL.md`.
Tracker: `bd ready -l zelda_i -l spine`. Living residual: `docs/PRE_L1.md`.

## Tracks

Survival health refill is not Clean. `--through pre-l1` forces that assist off.
Gathering is the spine's default prefix (`spine/survival.py` `_run_gathered_prefix`):
pre-l1 assist-off, gather chain under its own refill (`--gather-engage-hearts`,
default 1 = last-heart; 0 is the next rung), Blue Ring purchase at 0x34,
then L1 from the 0x37 door. Survival tops rupees to 250 before the shop and
records that count write. Ringless L1 and later saves are obsolete on the
main spine; `--resume` rejects them. Regenerate from power-on.
Planner owns `docs/STATUS.md`. The 18909f wooden M5 oracle is retired (2026-09-22); do not protect it.
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
uv run python nes/zelda_i/scripts/stage_replay.py <state> zelda_i.level6.dungeon:ROOM_29_SPEC --assist --window A-B   # one controller/spec from a save point; --idle N = RNG offset
uv run python nes/zelda_i/scripts/run_survival_spine.py --engage-hearts 1 ...        # refill at the last heart only: refills = deaths prevented
uv run python nes/zelda_i/scripts/run_survival_spine.py --engage-hearts 1 --observed-damage-guard ...  # keep last-heart target; safety refill after larger observed hits
uv run python nes/zelda_i/scripts/run_survival_spine.py --through gather --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --no-gather --no-video --trials 1   # legacy wooden-sword prefix
uv run python nes/zelda_i/scripts/run_survival_spine.py --through pre-l1 --no-video --trials 1
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
- Do not poke Food, bombs, keys, the candle, or `$066F`. `--through pre-l1` still forces heart assist off. It does write `$066D` up to 20 before `bomb_topup` when the wallet is short. That write is not the rr-ttyu.3 buy. Quote a tape only against the code that produced it. Read `reason_by_screen` before changing a hop.
- A `@dataclass` copies field defaults into `__init__`. Setting the default on the class later does not change instances.
- Walls come from `dungeon.tilemap.ow_walkable_nodes`, the ROM collision on the 8 px turn grid, not from a screenshot. The old `measured_walker` samples one pixel and misses Link's width. 0x79 y=165 dead-ends at x=192.
- Walls, doors, stairs and block pushes go through the ROM lattice helpers in `dungeon/hop_controller.py` (`LatticeDoorWalker`, `lattice_goto`, `block_push_step`, `stairs_step`); hand waypoint policies are fallbacks only.
- Every spine run prints a ledger (`spine/ledger.py`): frames per room visit, drop outcomes (picked/expired/left), missed hearts, flutter (1-2 px reversals), damage per room, dungeon room items left untaken (world flags `$06FF` L1-6 / `$077F` L7-9), and inventory rises (play vs written between frames). Read `flutter rooms` and `slowest visits` before tuning; record milestones in `docs/RUN_METRICS.md`.
- Link turns only on the ROM lattice (x%8==0, y%8==5). A press on the other axis slides him onto it first, so a greedy "bigger axis first" step, a 1 px sidestep, or an exact-pixel stop flips every frame. Walk with `room_step` / `mouth_step` / `LatticeDoorWalker` / `exit_door` / `stairs_step`; one helper owns a goal end to end (a lattice approach plus a separate pixel finish is a tug-of-war).
- On a deployed stepladder (0x5F) only its axis moves; wrap hand presses in `release_action`. The ladder sprite sits 3 px below Link's row. The dock raft (0x61) and stepladder are not combatants.
- A room's wave spawns a few frames after the scroll: wait for it before counting live enemies (`_fight_if_live`).
- Do not edit runtime modules while a spine run is in flight: level modules import lazily, so a run picks up half-edited code (power-on 6 died on an `ImportError` after `ram.py` changed mid-run).
- Water rooms: a dungeon clear whose exit is on Link's bank sets `reachable_only`; the ladder is owned by `LadderEscape` / goal-aware `ladder_release`; a patrol waypoint the tiles call water is moved to land (`_patrol_vertex`).
- Score a combat change on the multi-offset eval, not on one tape (`stage_replay.py --idle`). A dungeon reroute that touches a room M5 uses (0x23, 0x33) must be re-run against M5's 18909f.

## Pointers

[docs/PRE_L1.md](docs/PRE_L1.md) · [docs/STATUS.md](docs/STATUS.md) ·
[docs/plan.md](docs/plan.md) · [docs/ASSIST_CONTRACT.md](docs/ASSIST_CONTRACT.md) ·
[docs/HYGIENE.md](docs/HYGIENE.md).

Files under `docs/tasks/` are lane notes the clean-tip ladder still cites.
They are not this sitting.
