---
name: sm-room-policy
description: >
  Optimize one Super Metroid room or boss hop: research the public RTA/TAS
  policy, benchmark the current autobot from a live enter pin, implement,
  then benchmark again. Always report frames and seconds. Use when the user
  says "optimize this room", "room policy", "bench before and after",
  "faster hop", "Ceres Ridley fight", "minimize room time", or runs
  /sm-room-policy.
---

# SM room policy (bench before / after)

One room per turn. Same enter pin for before and after. Do not STATUS-promote
a new continuous tip from a pin bench.

## Time contract

Every bench row must go through `super_metroid.room_timer.format_segment_time`:

| Field | Meaning |
|-------|---------|
| `frames` | emulator steps (source of truth) |
| `seconds` | `frames / 60.0988` (NTSC, plan.md tables) |
| `clock` | `mm:ss.cc` via `fmt_tracker` |

Print a three-row table: **before** / **after** / **Δ**. Negative Δ is faster.

## Loop

1. **Name the hop.** Room hex + from→to + items. `skill_bank.make_hop_key`.
2. **Research.** `wiki.supermetroid.run` first, then TAS / VOD. Write the
   public policy in the controller docstring (one home). Do not invent a
   fight if the wiki says take damage / skip / wait.
3. **Capture the enter pin.** Natural predecessor, not a door-warp.
4. **Bench BEFORE** the current product body from that pin. Save the JSON.
5. **Implement** a RAM-driven policy (seat → window → exit). Unit-test
   actions without the emulator. Keep the module under ~1000 LOC
   ([CODING_STANDARDS.md](../../../CODING_STANDARDS.md)).
6. **Bench AFTER** from the **same pin**. Overwrite `scratch/<hop>_bench.json`
   (not `_vN` / `_window_*`). If it is not faster and successful, do not
   wire it. Three red windows on the same checkbox → BLOCKED, stop.
7. **Wire** the winner only after the **next hop** still clears from the
   new leave pin (faster fights change Ceres elev debris phase). Glance
   the leave with `hop_glance` — not an MP4. Re-record the continuous tip
   before any STATUS claim. If the next hop dies, keep the old product
   body and leave the new policy behind a flag. Never overwrite
   `recordings/<tip>.json` on a red run. Session gates: `sm-session`.

## Probe shape

Mirror a surviving room probe such as
`snes/super_metroid/scripts/probe/ws_main.py` (`bench` / `dump` / `pure`
subcommands):

```bash
uv run python snes/super_metroid/scripts/probe/<room>.py bench
```

`bench` must reload the enter pin before each policy. A boss hop that runs
inside the station spine is benched through the spine CLI instead
(`python -m super_metroid.routes.kpdr.ceres.spine station`).

## Ceres Ridley (worked example)

Public policy: energy **< 30** starts the escape; five right-wall **tail**
hits. Shooting 100 times is slower.
https://wiki.supermetroid.run/Ridley#Ceres_Station

```bash
# The fight runs inside the full-station play; hops report vs TAS.
PYTHONPATH=snes uv run python -m super_metroid.routes.kpdr.ceres.spine station
uv run pytest snes/super_metroid/tests/test_ceres_ridley_combat.py -q
```

Controller: `combat/ceres_ridley.py` (`CeresRidleyStrategy`; product default
is `fresh_fifth_jump`, no route flag). Same-pin fight bench is
`routes/kpdr/ceres/data/ceres_ridley_bench.json`; the numbers live in
`docs/plan.md` § Ceres Ridley fight — do not copy them here.

Traps: energy assist is already off on Ceres; do not leave the wall on hit
count alone (weak hits do not cross 30); countdown is not HP-zero.

## Tests

Unit-test seat / action / "don't fire at 0 ammo" / countdown-stop without the
emulator. Emulator proof is the bench JSON (`success`, frames, seconds, clock),
with the before and after runs loading the same enter pin.
