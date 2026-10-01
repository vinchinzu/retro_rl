# Assist contract — ALTTP

Published ALTTP results are Clean unless this file and the run report say
otherwise. A save-state pin is not continuous. Only a power-on session with
no mid-run state load is continuous.

The shape follows `nes/zelda_i/docs/ASSIST_CONTRACT.md` (Clean versus
Survival, what-if pins, rollout disclosure). NES addresses, inventory
top-ups, and retired Zelda I pokes are not ALTTP rules. Zelda I lookahead
permission is not a ruling for ALTTP.

## Clean

No RAM write, no mid-run state load, and no HP refill. Do not write keys,
rupees, HP, items, or follower bytes (including `$F3CC`) in a run quoted as
Clean or continuous. Natural entry is `--natural` from the real predecessor.
Work-queue pins are development only.

A Clean quote is a power-on tape replay that itself makes no state load and
no RAM write. Do not mix a Survival or what-if green into a Clean row.

## Survival (development only)

HP refill at the last heart. Development only. Disclose every use in the
run report: assist enabled, write count, hearts restored, and the first
frame the assist was active. Survival does not write inventory, keys,
rupees, capacity, or progression. `progression_writes` and
`capacity_writes` stay 0.

A Survival refill is not a Clean claim and is not continuous evidence.

## What-if pins (development only)

A what-if pin sizes a lever. Write the item into a pin copy, run the
segment, decide, then delete the pin. What-if pins are development only
and are not quoted as Clean or continuous.

Pokes are allowed only in pins named `Dev*`, `Whatif_*`, or `*Entry`, and
each needs a `.provenance.json` that says `poke`. Delete the what-if pin
after the decision.

## Rollout lookahead

Lookahead saves the core, plays a few frames ahead, and restores the same
live frame. When a development run uses it, disclose those restores as
`rollout_restores`. Count `mid_run_state_load` only for other loads. This
is the Zelda I disclosure shape. It is **not yet ruled** for ALTTP.

The owner allowed lookahead for Zelda I on 2026-09-28. That permission is
not a ruling for ALTTP.

Owner ruling:

## Required disclosure

Any assisted or lookahead result records this contract path and:

- assist enabled, write count, hearts restored
- `rollout_restores` when lookahead ran
- `progression_writes` and `capacity_writes` (must stay 0 on Survival)
