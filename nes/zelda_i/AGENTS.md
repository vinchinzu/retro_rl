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

# Clean M5 (do not overwrite) — 2/2 TF 0x01 @ 18909f
uv run python zelda_i/scripts/run_level1_complete.py --natural-entry --trials 2

uv run python nes/zelda_i/scripts/clean_tip.py
uv run python nes/zelda_i/scripts/clean_tip.py --adoption
uv run python nes/zelda_i/scripts/audit_pins.py

uv run pytest zelda_i/tests -q
```

`--no-video` on spine CLIs. Leave proof is RAM + `zelda_i.screen_glance`,
not an MP4. Segment CLIs: `docs/plan.md`.

## Traps (burned once)

- **M5 Clean 18909f is live.** Re-measure after walker changes. Do not overwrite.
- Arrive-short top-up is `overworld/topup.py` (`bomb_topup` stage): 0x6F UP→**0x5F** / LEFT→**0x6E**, measured out and back (`scratch/probe_6f_neighbours.py` n4). Never a lap back west — `RoomHistory` is six slots and nothing behind 0x6F respawns inside five screens.
- Pre-L1 dest is **0x6F** south coast `shop_p7` (`0x77→…→0x7F→0x6F`), not inland **0x4A** (`bomb_shop.py`). Stop is `ADDR_BOMBS>=1`; red on purpose until the 4-pack.
- **Read `report()["reason_by_screen"]` before tuning anything on a hop table.** It is a per-screen count of the stem of every `FrameAction.reason`. Four separate 20k-frame stalls were found in it in one sitting, none by tuning a number.
- 0x7B / 0x7C / 0x7D scroll east from **every** row: those three hops carry `SCREEN_ANY_ROW_BAND`, **not** an `align_y`. An align nothing needs is a vertical shuffle in a leever swarm (394 of 0x7B's 1863 frames).
- Every rung needs a budget. The contact strike had none and one body owned 24877 frames (`ScreenHunter._strike_budget`); local caps do not compose, so the hop carries its own per-screen one (`_grinding`, 4000f) and drops scoop/hunt/occupied-lane past it.
- `_after_hops` answers a declining hunt with an **idle**, and the hunt declines for the whole guard branch — that pair is a 2400-frame stand on the destination screen. `destination_hunted` is True in guard.
- `ScreenHunter.cleared` ≠ `done`: a budget retire is `done` with the wave still alive. Only `cleared` scoops money.
- 0x79 east is **y=165 beach only**. 0x7A east is `y_band (133,141)`. 0x7E east is `y_band (137,145)`. `align_y` + `y_tol=5` accepted the dead rows (27501f stall). Do not BFS `OccupancyWalker` for a lane.
- ButtonsPressed is an **edge**; held A does not re-swing. Hunt `_a_edge` / travel press then idle.
- Sword shot is `beam.py`; travelling frames must offer `ScreenHunter.take_beam` above stall-escape (600f commits used to zero the weapon).
- Zora facing **0x03** is missing from `threat._FACING_AXIS`; `0x55` spit is that hole, not a shop_p7 special case.
- Scoop vs restock are separate: scoop_rupees vs `need_rupees`. Coast scoops; `need_rupees=0` so no 0x78 farm loop.
- `laps=0` by default; `RoomHistory` six slots, out-and-back evicts nothing.
- Do not poke Food/bombs/keys. `--through pre-l1` forces assist and pokes off.
- Sword cave is **NW** of spawn on 0x77. Cave = mode **11**. Pickup x≈120 then UP; after cave exit ~(64,77): **DOWN first**.
- `$066F` lo nibble is whole hearts minus one (`0x22`=3/3). Use `ram.whole_hearts`. Read the pin's `$066F` before quoting a heart (`lo <= hi`).
- Map rooms from `$6530`, never `$049E`. A bare `OccupancyWalker()` knows no walls — `measured_walker(env.get_ram())`. `retarget_blocked_goal` stays off for anything L1 touches.
- `$0656` B-item: **1=bombs, 2=arrows, 4=candle**. L2 prefix: `37→38→48→58→59→49→4A`; never 0x79.

## Pointers

[docs/STATUS.md](docs/STATUS.md) · [docs/plan.md](docs/plan.md) ·
[docs/PRE_L1.md](docs/PRE_L1.md) ·
[docs/tasks/rr-8t4.4-residual.md](docs/tasks/rr-8t4.4-residual.md) ·
[docs/ASSIST_CONTRACT.md](docs/ASSIST_CONTRACT.md) ·
[docs/HYGIENE.md](docs/HYGIENE.md) · session skill `zelda-session`.
