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
   actions without the emulator. Keep files ≤ ~500 lines.
6. **Bench AFTER** from the **same pin**. If it is not faster and successful,
   do not wire it.
7. **Wire** the winner only after the **next hop** still clears from the
   new leave pin (faster fights change Ceres elev debris phase). Re-record
   the continuous tip before any STATUS claim. If the next hop dies, keep
   the old product body and leave the new policy behind a flag. Write
   experimental continuous reports under `scratch/` — never overwrite
   `recordings/<tip>.json` on a red run.

## Probe shape

Mirror `snes/super_metroid/scripts/probe/ceres_ridley_combat.py`:

```bash
# capture | dump | strategy --policy <name> | bench
uv run python snes/super_metroid/scripts/probe/<room>_combat.py bench
```

`bench` must reload the pin between policies.

## Ceres Ridley (worked example)

Public policy: energy **< 30** starts the escape; five right-wall **tail**
hits. Shooting 100 times is slower.
https://wiki.supermetroid.run/Ridley#Ceres_Station

```bash
uv run python snes/super_metroid/scripts/probe/ceres_ridley_combat.py capture
uv run python snes/super_metroid/scripts/probe/ceres_ridley_combat.py dump --frames 400
uv run python snes/super_metroid/scripts/probe/ceres_ridley_combat.py bench
```

Controller: `combat/ceres_ridley.py`. Product flag:
`routes/kpdr/ceres/outbound.py` `CERES_RIDLEY_POLICY`. Pin-bench numbers live
in `docs/plan.md` § Ceres Ridley fight — do not copy them here.

Traps: energy assist is already off on Ceres; do not leave the wall on hit
count alone (weak hits do not cross 30); countdown is not HP-zero.

## Tests

Unit-test seat / action / "don't fire at 0 ammo" / countdown-stop without the
emulator. Emulator proof is the bench JSON (`success`, frames, seconds, clock,
same `state` path on both rows).
