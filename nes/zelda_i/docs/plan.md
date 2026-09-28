# Plan — Zelda I

## Now: the credits tape, faster and cheaper (2026-09-28)

C11 is green. `clean_poweron_c12` plays power-on to the credits in one
session (334,763 frames, TF `0xFF`, no refill, no RAM write, no death). Its
only state loads are rollout lookahead restores, which the owner allowed for
development on 2026-09-28. The MP4 is that run's button tape replayed from
power-on with no lookahead (`scripts/replay_tape.py`). The replay makes no
load and no write, its system RAM is checked every frame, and it encodes at
about 12x real time. Numbers: [STATUS.md](STATUS.md).

### Next, in order

Claim one bead. `bd ready -l zelda_i` is this list.

1. **The policy at real time.** The live run is the dev loop, and it must
   not fall behind the NES. Each stage report carries `x_realtime`. The
   lookahead stages (`PolicyGuard` on the post-L8 walk and in Level 9,
   `PatraBlade`, and the 0x04 `Rollout.walk`) are the slow ones. Cut rollouts
   per frame before you cut the horizon: gate the 24-frame self-roll on a
   hazard in reach, as `RolloutEvader` does.
2. **Level 9 health.** Room 0x10 (Wizzrobe magic `$58`/`$59` and Bubble
   contacts, 0.25-8.25h over the offsets) and the Spectacle Rock bomb
   (0-3h) are the largest Clean costs left. Score changes on the offsets
   (`eval_l9_hop_offsets.py`), not on one tape.
3. **Optional route levers.** The Red Ring detour is on main but off
   (`red_ring=True`). It needs a ninth bomb, so reroute 0x31 west through
   0x51 -> 0x50 -> 0x40 -> 0x30 first. Skip the 0x4A detour when the wallet
   is short (`GatedLeg` with `POST_L8_TO_LEVEL9_HOPS`, about 1,000 frames).
4. **Lookahead-free Clean, only if the owner asks for it.** Under
   `docs/BENCHMARK_SPEC.md`, strict Clean allows no emulator-state mutation
   during the attempt. The replay meets that; the live policy does not.
   Replacing `PatraBlade`, `room04_west_plan` and the `PolicyGuard` users with
   model-only policies is its own epic.

Potion (rr-thlc), extra keys (rr-qb6w), and the 0x3a heart bleed
(rr-rgum.1) wait until a Clean tape actually dies there.

### Clean ladder

A rung is green once one power-on `--clean` run reaches it.

| rung | stop | green on |
|---|---|---|
| C1 | White Sword taken | `clean_poweron64` |
| C2 | Blue Ring, L1 mouth 0x37 | `clean_poweron64` |
| C3 | L1 Triforce (new M5 on the gathered route) | `clean_poweron64` |
| C4 | L2 Triforce | `clean_poweron64` (TF `0x03`) |
| C5 | L3 Triforce | `clean_poweron69` (121,388f) |
| C6 | L4 Triforce | `clean_poweron74` (152,416f) |
| C7 | L5 Triforce | `clean_poweron76` (190,444f, TF `0x1F`) |
| C8 | L6 Triforce | `clean_poweron83` (235,596f, TF `0x3F`) |
| C9 | L7 Triforce | `clean_poweron84` (264,227f, TF `0x7F`) |
| C10 | L8 Triforce | `clean_poweron98` (296,423f, TF `0xFF`) |
| C11 | Ganon, Zelda, credits | `clean_poweron_c12` (334,763f; lookahead disclosed) |

## The Gathering (route order)

Zelda Dungeon calls this The Gathering. Order:

1. Wooden sword on `0x77`.
2. South-coast walk to bombs at `0x6F`. Stop when `ADDR_BOMBS >= 1`.
3. Heart at `0x7B`, the 0x39 pond, then **the red potion** at the `0x2C`
   take-any (left item, (88,149); 2026-09-25, was the container).
4. Northeast cluster: 30R `0x2D`, 100R `0x0F`, letter `0x0E`, candle `0x0C`,
   30R `0x28`.
5. Burn row before the sword: 30R `0x48`, heart `0x47` (5 containers), back
   up the x=120 cut to `0x28`, White Sword `0x0A`, back down to `0x48`.
6. Blue Ring at `0x34`, paid by the hidden rupee caves (`0x5B`, `0x6B`,
   `0x56`); 0x62's 100R after it buys Bait at `0x34`. Arrows are bought
   after L4, and 0x67 funds Level 9's second bomb pack.
7. The 0x39 pond again, then the Level 1 mouth at `0x37` with 5 containers
   and usually one potion charge.

The 18909f wooden-sword oracle is historical; the gathered Clean L1
Triforce (`clean_poweron64`) replaces it as the M5 evidence.

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --no-infinite-life --no-video --trials 1   # gather → L1 TF
uv run python nes/zelda_i/scripts/run_survival_spine.py --through pre-l1 --no-video --trials 1
```

That command forces the health assist off. `--rollout` stays opt-in.

## Open on the bomb errand

`rr-ttyu.3` is still open. One flagged `--rollout` trial, tag `pre_l1_7e_band1` on 2026-09-20, reached cave `0x6F` with bombs 4. That trial is not the default arm and not a STATUS result. The next measurement is flag-off against flag-on damage on `0x7B`, `0x7C`, and the Zora shot `0x55`, with no new stall. Do not promote rollout. Do not retry a fixed y=133 coast row.

A short arrival is not the gate. The walk tries to bank more than 20 on the coast. If `0x6F` is still short, `bomb_topup` keeps hunting and comes back, and only then does the buy run. The gate is `ADDR_BOMBS >= 1`.

## Not this sitting

Level 2's east mouth, the dungeon lanes, and Clean continuous play have notes under `docs/tasks/`. `spine/clean_tip.py` still points at those files. None of them is the next open row. Survival tapes and their run history live in [RUN_METRICS.md](RUN_METRICS.md).

Session loop: `.grok/skills/zelda-session/SKILL.md`. Claim one bead. Overwrite [PRE_L1.md](PRE_L1.md) with the leftover. Glance leave is room, mode, x/y, sword, bombs, rupees, and hearts. A pin is not leave proof.
