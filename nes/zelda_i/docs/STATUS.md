# Status — Zelda I

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M5 |
| Best verified result | Clean power-on to Level 1 Triforce, triforce `0x01`, 18909f, 2/2 natural-entry |
| Last verification | 2026-09-14 |
| Runtime class | Bronze |
| Intervention class | Clean |
| Evidence | `spine/clean_tip.py` row `l1_tf`. Published chain file: [level1_complete_natural.json](../recordings/level1_complete_natural.json). Isolated chain: [level1_complete_isolated.json](../recordings/level1_complete_isolated.json). |
| Not the gate | Gathering, Survival dungeon tapes, and any `--rollout` trial. `--through pre-l1` is the open prefix and has no Clean claim. |

The 18909f figure is the clean-tip oracle recorded on 2026-09-14. This doc pass did not re-run the ROM. The older natural JSON is the published chain file. Do not treat its frame total as 18909.

Re-measure with `scripts/run_level1_complete.py --natural-entry` and no health refill. Do not overwrite the oracle from a gathering or Survival run.

## 2026-09-23: Survival power-on → credits, one continuous run

`run_survival_spine.py --through level9-credits --save-points Full --no-video`
went from power-on to the credits in one emulator session: `ok=True`,
`set_state_count=0`, 292,742 frames, 234 stages, TF `0xFF`, 12 containers,
Ganon and Zelda, final mode 19. Tape: `recordings/full_poweron11.json`.

This is **Survival, not Clean**: the health refill is on (768 hearts of
damage absorbed), bomb/key/rupee counts are topped up at declared gates
(`SPINE_*_RETOPUP`), and L7's Bait is the disclosed Food fixture. The M5
Clean gate above is unchanged. One run, not repeated yet.

How it got there: continuous runs exposed stalls that the resumed save-point
tapes hid (each fix moves every later frame). Almost every stall was a hand
walk disagreeing with the ROM, or two rules swapping 1-2 px each frame; the
fixes route through the ROM lattice (`dungeon/hop_controller.py`:
`stairs_step`, `block_push_step`, `door_nodes`, `ow_edge_band_step`,
`inland_lattice_step`) behind a stall gate. Heart containers are route
history, not a handoff gate.

## What is open

`spine/clean_tip.py` `next_open()` is `pre_l1`. Route and the one live tape are in [PRE_L1.md](PRE_L1.md). One flagged rollout trial bought bombs. The default walk is not accepted.

## What is not written here

Per-hop Survival ledgers used to live in this file. The JSON files under `recordings/` still hold those runs. They are development tapes. They do not move the M5 gate. Lane notes under `docs/tasks/` are the same kind of record. Fixture pins with 15 hearts in too few containers are not Clean measurements. `scripts/audit_pins.py` is how a pin gets refused.
