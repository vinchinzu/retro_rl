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
- An out-and-back hop lands on the **reverse** arrival edge. `on_arrival_edge` names the *travel* edge, so the back hop skips the hunt and `recover_off_edge` allows the retrace: live t1 left 0x5F in 87f with `peak_live` 0. `RupeeTopUpController._extra_hop_action` holds (`topup_hold`) until `hunter.done`. Do not BFS occupancy on 0x6E (bush maze; t2 stood 214f on the east line).
- Pre-L1 dest is **0x6F** south coast `shop_p7` (`0x77→…→0x7F→0x6F`), not inland **0x4A** (`bomb_shop.py`). Stop is `ADDR_BOMBS>=1`; red on purpose until the 4-pack.
- **A tape is a tape of the code that ran it.** `pre_l1_anyrow1` is the only
  pass that reached 0x6F and `hunt.py` / `path.py` were edited *after* it and
  before the next tape (`7b364caf`). Check source mtimes against the report's
  mtime before quoting an old run as this walk's behaviour.
- **`reason_by_screen` is a histogram; it cannot say whether Link moved.**
  0x7C read 4301 frames of `hop`/`hop_lane` and was 3593 frames at x ∈ {24,25}
  — the peel steps LEFT while the push steps RIGHT. `scratch/probe_walk_trace.py`
  traces `(frame, screen, x, y, reason)` per frame; use it on any screen the
  census cannot explain. Lane budget is `_lane_no_gain` (travel-axis progress),
  not a consecutive-steer count — every push frame zeroes `_OCCUPIED_LANE_STEER_CAP`.
- **Read `report()["reason_by_screen"]` before tuning anything on a hop table.** It is a per-screen count of the stem of every `FrameAction.reason`. Four separate 20k-frame stalls were found in it in one sitting, none by tuning a number.
- 0x7B / 0x7C / 0x7D scroll east from **every** row. 0x7B and 0x7C hops carry `SCREEN_ANY_ROW_BAND`, **not** an `align_y` (394 of 0x7B's 1863 frames were `hop_ay`). 0x7D's own exit is every-row too, but the hop that *leaves* 0x7D carries `SCREEN_7E_EAST_BAND`: 0x7E's 133 row is dead (`l1` stood at 131 and never scrolled), and `pre_l1_topup_live` died `(40,131)` after entering it. Do not put ANY_ROW back on that hop.
- Every rung needs a budget. The contact strike had none and one body owned 24877 frames (`ScreenHunter._strike_budget`); local caps do not compose, so the hop carries its own per-screen one (`_grinding`, 4000f) and drops scoop/hunt/occupied-lane past it.
- `_after_hops` answers a declining hunt with an **idle**, and the hunt declines for the whole guard branch — that pair is a 2400-frame stand on the destination screen. `destination_hunted` is True in guard.
- `ScreenHunter.cleared` ≠ `done`: a budget retire is `done` with the wave still alive. Only `cleared` scoops money.
- 0x79 east is **y=165 beach only**. 0x7A east is `y_band (133,141)`. 0x7E east is `y_band (137,145)` on the hop that leaves 0x7D *and* the hop that leaves 0x7E. `align_y` + `y_tol=5` accepted the dead rows (27501f stall). Do not BFS `OccupancyWalker` for a lane.
- ButtonsPressed is an **edge**; held A does not re-swing. Hunt `_a_edge` / travel press then idle.
- **A turn and a swing cannot share a frame.** `nes_action(face, "A")` from a walking Link keeps the *old* facing 22 times in 64 (`scratch/probe_turn_swing.py`, `turn4`) and the blade goes out along the axis the body is not on — a miss plus 13 pinned frames. Holding the direction alone turns him in 1-4 frames, always. Press A only once `$0098` agrees (`hunt._a_edge`, `common.swing_or_turn`). Not the **shot**: gating the beam the same way (`zfixG`) fired 4 where the baseline fired 36 — a wrong-way beam is still a screen-long projectile.
- **The blade has a near end.** `scratch/probe_blade.py` (`blade1`) ledgers every press against the hp drops that follow: `fwd >= 10` landed 11 of 41, `fwd <= 9` landed **1 of 13**, `abs(lat) >= 16` landed 0 of 4. The sword is an object placed *in front of* Link. `blade_lands` = `in_sword_hitbox` + `HUNT_BLADE_MIN_FWD`; inside that the answer is `_peel`, never the press. `pad <= MIN_DODGE_BODY` is the softlock rule, not a sword rule.
- **A leever with `ObjState` 0 is not a body.** Under the sand it never moved (3844 still frames to 45), never lost hp (every drop was state 2 or 3) and never hurt Link (49 frames overlapping him, no arm of its own) — five contact tapes. Six sit on 0x7B and seven on 0x7C, and the blade, the evader and `closest_body` all treated them as the wave. `combat.dormant_body`; states 1-2 are the rise and stay threats.
- **A dataclass field default cannot be overridden after the class exists.** `@dataclass` copies each default into the generated `__init__`, so `fields(C)[i].default = v` and `setattr(C, name, v)` are both no-ops on every instance. `probe_contact.py` printed `overrides:` over exactly that from its first sitting until 2026-09-16 — **every `--shield` / `--off-line` / `--min-hearts` / `--screen-budget` ablation before that measured the unablated hunt.** Wrap `__init__` instead.
- **Standing on a floor drop is not instant**: 20 frames at 3 px from a 1-rupee on live 0x7B, with a leever closing. `common._stand_on_drop` hands the frame back while anything is inside `MIN_DODGE_BODY`.
- Sword shot is `beam.py`; travelling frames must offer `ScreenHunter.take_beam` above stall-escape (600f commits used to zero the weapon).
- Zora facing **0x03** is missing from `threat._FACING_AXIS`; `0x55` spit is that hole, not a shop_p7 special case. Do not paint that axis: `scratch/zora1.json` says the shot is aimed at **launch** and flies straight, and five contact tapes put its motion at 466 frames horizontal / 248 diagonal / 172 vertical, sharing the Zora's row in 247 of 886 — there is no row to stand out of, so a firing *line* is the wrong shape. **Never engage a Zora — dodge and run.** The rung is `OverworldPathController._spit_duck`, above the evader and the hunt, and `prey.SKIP_TYPES` keeps it off the face (`common.walk_or_swing`) as well as off the chase.
- A dodge is a **walk**, and it has to be measured as one. `_EVADE_BOUNDS` is the scroll rectangle: it cannot see a rock, so a direction that moves Link nowhere for `_SPIT_DUCK_STILL_CAP` frames (excluding `link_busy`, which is the ROM pinning him through the sword animation) is written off per `(direction, 16 px cell)` — screen-wide poisoned 0x7B for eight hits. A body already inside `MIN_DODGE_BODY` outranks the shot (`_body_first`); a step cannot clear it there anyway.
- `common.answer_projectile` used to sidestep perpendicular to the **travel** axis and flip at the edge, which reverses the step *into* the shot on a wall. It crosses the bearing to the shot now (`common.perpendicular`, shared with `hunt`), and `perpendicular` asks for a **pad** of room, not one 2 px step.
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
