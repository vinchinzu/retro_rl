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
default 1 = last-heart; 0 is the next rung), then L1 from the 0x37 door.
Planner owns `docs/STATUS.md`. Leave the 18909f M5 oracle alone.
Clean re-measure is `run_level1_complete` without `--infinite-life`.

## Commands

```bash
bd ready -l zelda_i -l spine
uv run python nes/zelda_i/scripts/run_survival_spine.py --no-video --trials 1              # default: gather → L1 TF
uv run python nes/zelda_i/scripts/run_survival_spine.py --through gather --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --no-gather --no-video --trials 1   # legacy wooden-sword prefix
uv run python nes/zelda_i/scripts/run_survival_spine.py --through pre-l1 --no-video --trials 1
uv run python nes/zelda_i/scripts/clean_tip.py
uv run python -m zelda_i.overworld.gather_segments pin     # power-on pre-l1 → PreL1BombLeave
uv run python -m zelda_i.overworld.gather_segments chain   # → 0x37, 6 HC + White Sword; chain:<stage> resumes
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
- Score a combat change on the multi-offset eval, not on one tape. A dungeon reroute that touches a room M5 uses (0x23, 0x33) must be re-run against M5's 18909f.

## Pointers

[docs/PRE_L1.md](docs/PRE_L1.md) · [docs/STATUS.md](docs/STATUS.md) ·
[docs/plan.md](docs/plan.md) · [docs/ASSIST_CONTRACT.md](docs/ASSIST_CONTRACT.md) ·
[docs/HYGIENE.md](docs/HYGIENE.md).

Files under `docs/tasks/` are lane notes the clean-tip ladder still cites.
They are not this sitting.
