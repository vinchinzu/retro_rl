# Agent Instructions — zelda_i

NES Legend of Zelda (graph nav; **M5** Clean power-on → Level 1 Triforce).
Shared: `retro_harness.adventure`, `retro_harness.nes`.
Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/HYGIENE.md`,
`docs/ASSIST_CONTRACT.md`, `docs/PRE_L1.md`. Session:
`.grok/skills/zelda-session/SKILL.md`.
Tracker: `bd ready -l zelda_i -l spine`.

## Dual track

**Survival** (`--infinite-life` / health refill) vs **Clean**. Assisted greens
are not Clean STATUS. Planner owns STATUS. Clean M5 =
`run_level1_complete` without `--infinite-life`. Do not overwrite.

## Immediate goal

**Survival power-on → credits is green** (2026-09-07): `--through
level9-credits` 1/1, 354346f, `set_state=0`, mode 19, TF `0xFF`, deaths 0.
Not Clean STATUS. M5 Clean is still L1 only.

**M5 Clean is GREEN as of 2026-09-14 at 18909f** — measured, not inherited.
`run_level1_complete --natural-entry --trials 2` both `ok=True`,
`triforce=0x01`, end 18909, ~26s. That run is 3 containers and the wooden
sword. **19416f is dead**: it was the pre-`6ca2a9a0` tree. `6ca2a9a0` landed
the `shortest_path` goal guard on unit tests alone and took M5 red (death in
0x23 at f1453, then a 9000f collect stall in 0x45) — re-measure the live
oracle after *any* change under the walker, never the suite alone. Next Clean prefix is Zelda Dungeon The Gathering **before** L1
([`docs/PRE_L1.md`](docs/PRE_L1.md)): bombs at 0x6F (bypass 0x79), two bomb
hearts, candle at 0x0C, White Sword around Lost Hills, burn heart, then 0x37.
`--through pre-l1` now clears every screen on the way (`overworld/hunt.py`),
1/1 green: 14 kills, 0 damage, walk 4010f — and **0 rupees**. The corridor
does not fund the 20R bombs; the forced 5-rupee at 10 kills never fires
(`streak_best` 4, `streak_resets` 6 on a zero-damage walk). Kills are in the
runner line and in `hunt.report()`.
Do not overwrite the 18909f claim; re-measure L1 after the prefix greens. `clear45_key` 1568f 0 hits (was
death 828f `{0x27_S}`, then a 9000f collect stall). Planner owns
STATUS; this is the ROM claim, not a STATUS rewrite.

Per-stage health ledger is the tool. Read it before tuning any room
(`in`/`out` hearts + `hits_by_cause` per stage). Current spend:
`clear52` 1 (`0x1b_E`), `clear23_key` 1 (`0x5c_W`), `clear44` 2
(`0x06_N`, `0x5c_E`). 0x33, 0x43, 0x45 are 0-hit. Link still arrives
at 0x45 on half a heart; the key hunt now finishes anyway. Hearts on
the floor in 0x23/0x44 are still unbanked.

Three fixed root causes, all the same shape: a position rule that
silences the reactive layer. The 0x23 low-health mask replaced the
planner's only wanted direction with an idle frame; `_off_wall_step`
and the 0x44 west-mouth table ran ahead of `threat.decide`. Reactive
now runs first in `_combat`. Rooms measure walls from `$6530`
(`occupancy_from_tilemap`). `OccupancyGrid.shortest_path` is
minimum-turn (Link snaps off-axis on every turn). `_scoop_heart`
yields on an unreachable drop instead of idling, with a 20-frame
cached BFS verdict. Collect skips a waypoint when manhattan to it
has not dropped in 48 frames (0x45 sat at (144,141) for 7666f in a
3px y-loop that never tripped in-place stuck).

Do not feed the occupancy grid into `ReactiveEvader._can_move`: every
variant that changed 0x23 evade buttons capped `clear23_key` at 6000f.
Ladder: `uv run python nes/zelda_i/scripts/clean_tip.py`.

Remaining spine: strip Survival pokes. `bd ready -l zelda_i -l spine`.
Living residual: [`docs/tasks/rr-8t4.4-residual.md`](docs/tasks/rr-8t4.4-residual.md)
(Food poke; shop hop wired. `rr-ps7.3` leftover is Clean 0x4A
`(155,141)` hp `0x31` 1/4 `hits_taken=0`. Gathering 4.5.1 leftover is
play `0x68` `(48,198)` hp `0x22` 3/3 hits 0 — west-column east is dead;
next is `0x59` SOUTH onto `0x69`).
Clean lanes: [`docs/tasks/rr-npv-clean-parallel.md`](docs/tasks/rr-npv-clean-parallel.md)
(`rr-npv.1`–`.7` fixture-live; not spine). Also open: `rr-wabn`, `rr-doua`,
`rr-sz8.8` (optional; leftover was 10 HC). Do not add pokes.

## Commands

```bash
bd ready -l zelda_i -l spine

uv run python nes/zelda_i/scripts/run_survival_spine.py --no-video
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-clear3a --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-stairs3a-warp --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-cellar08 --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-south1d --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-west2d --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-north2c --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-gohma --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-heart --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-north0c --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6 --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level1-bow --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level1-bow-cellar --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level1-bow-pickup --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level1-arrows --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level1-bombs --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through pre-l1 --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level2-entry --no-video --trials 1
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level7-bait-shop --no-video --trials 1

# Clean M5 (do not overwrite) — 2/2 TF 0x01 @ 18909f, see Immediate goal
uv run python zelda_i/scripts/run_level1_complete.py --natural-entry --trials 2

# Clean ladder: tip, next open hop, blockers grouped by root cause
uv run python nes/zelda_i/scripts/clean_tip.py
# Which levels run the shared engine mechanisms (measured off the live specs)
uv run python nes/zelda_i/scripts/clean_tip.py --adoption
# Every level-entrance pin's $066F. Exit 1 if any is incoherent.
uv run python nes/zelda_i/scripts/audit_pins.py

uv run pytest zelda_i/tests -q
```

`--no-video` on spine CLIs. Leave proof is RAM + `zelda_i.screen_glance`,
not an MP4. Segment CLIs (L2–L9, TAS, lab): `docs/plan.md`.

## Layout

| Path | Role |
|------|------|
| `ram.py`, `overworld/graph.py`, `overworld/nav.py` | Snapshots + OW graph / L1 path |
| `overworld/path.py` | Shared hop engine (L2–L8) |
| `walk/physics.py`, `walk/predict.py` | OccupancyWalker + RAM claims |
| `dungeon/engine.py` + `level*/dungeon.py` | Combat + **specs/stop predicates only** |
| `spine/hops.py` | `SpineHop` rows + `attach_hops` / `ready` |
| `dungeon/hop_controller.py` | Dest-hop timeout/death/scroll guard |
| `dungeon/token_path.py` | L4 maze hold-token walker |
| `level*/path.py`, `level*/spine.py` | Path controllers + dest spine tables |
| `level*/overworld.py` | Hop tables + thin `overworld.path` subclasses |
| `runner.py` | Script env/assist/report helpers |

Map a room from the cart-WRAM `$6530` tile map
(`dungeon/tilemap.py`), never from `$049E` `colliding_tile` sweeps —
`$049E` is the tile Link walks *into*, so it is direction-sensitive.

Size: [CODING_STANDARDS.md](../../CODING_STANDARDS.md) (~1000 LOC, merge
or delete). Named pins stay named. Probe PNG / window JSON go gitignored
scratch — not an AGENTS novel.

## Traps (burned once)

- Sword cave is **NW** of spawn on 0x77. Cave = mode **11**. Pickup x≈120
  then UP; after cave exit ~(64,77): **DOWN first**.
- `$066F` low nibble is whole hearts, not `0xF` full. Full is `lo==hi`
  (`0x22`=3/3) plus `$0670=$FF`.
- L2 prefix: `37→38→48→58→59→49→4A`; never 0x79.
- Stuck nav: stand still (`*_wait`). Do not loop LEFT/RIGHT/DOWN wiggle.
- `$0656` B-item: **1=bombs, 2=arrows, 4=candle**.
- Do not poke doors/keys/undiscovered items. Do not grant Map/Whistle.
- L2 entry bombs=0; Survival count top-up until farm `rr-doua`.
- **Read the pin's `$066F` before quoting a heart.** `hi = containers-1`,
  `lo = whole hearts`, so a coherent byte has `lo <= hi`
  (`ram.health_byte_is_coherent`). The ROM does *not* clamp a byte that
  breaks this. **7 of 14 entrance pins are incoherent** (2026-09-14 audit):
  `Level2/3/4/5/6Entrance`, `Level5EntranceFromL4`,
  `Level8EntranceReconFixture` all hold `lo = 0xF`, so every Clean heart
  number on `l2_tf`-`l8_tf` is against a 2-7x inflated budget.
  `Level5Entrance` also holds `TF 0x00`, which no L5 arrival can. Rebuild
  from a measured arrival (`scripts/fixtures/capture_level6_entrance_fixture.py`
  is the pattern: continuous power-on spine, no state load, save at the stop);
  do not hand-write the byte.
- **A BFS goal inside geometry has no path, and standing is never the
  answer.** `OccupancyWalker(retarget_blocked_goal=True)` moves a blocked goal
  to `grid.nearest_open` and counts it (`retargets`); `measured_walker`
  defaults it on. **Off by default, and keep it that way for anything L1
  touches** — retargeting shifts arrival frames and the L1 chain is
  frame-perfect: turning it on globally took Clean M5 from a green TF `0x01`
  to a red `aquamentus_heart` at 18830f. L2 `0x6e` aimed its key-door band
  walk at `(120,113)`, which is inside a diamond: 3999 of 4000 frames in
  `band_wait` without moving a pixel. Check a hand-written waypoint against
  the tile map before adding another one.
- **A bare `OccupancyWalker()` knows no walls.** It learns each one by
  bumping it and, non-sticky, forgets it again. Build one with
  `walk.physics.measured_walker(env.get_ram())` (live `$6530`, sticky by
  default). That loop is what spent `enter_6f_key`'s whole 4,000f budget in
  `band_wait`, and what the entry-route replan hit before it was made sticky.
- **A walled entry route used to be a silent timeout.**
  `dungeon/route_entry.py` now watches for 24 identical-pose frames on a held
  `entry_route` button, notes the pose and the leg, then replans off the live
  `$6530` map (sticky walker) or drops the leg. Below the threshold the axis
  walk is unchanged, so green chains are frame-identical. Every controller
  also keeps a records-only `reason_counts` + 30-frame `tail` in `report()`.
- **Overworld waves are one-shot.** `0x4A` is empty after its tektites die
  and stays empty through depth-1 (`0x49`) *and* depth-2
  (`0x49`→`0x59`→`0x49`) round trips. `HeartFarmController` restock is a
  give-up detector (`farm_screen_dead`), not a heart supply. Do not budget
  hearts against a farm.

## Pointers

[docs/STATUS.md](docs/STATUS.md) · [docs/plan.md](docs/plan.md) ·
[docs/ASSIST_CONTRACT.md](docs/ASSIST_CONTRACT.md) ·
[docs/HYGIENE.md](docs/HYGIENE.md) · session skill `zelda-session`.
Clean leftover: [docs/tasks/HANDOFF-2026-09-11-orchestrator.md](docs/tasks/HANDOFF-2026-09-11-orchestrator.md).
