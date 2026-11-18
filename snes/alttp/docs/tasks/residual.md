# Eastern B9 north exit — 2026-10-25

The verified Clean forward tape now runs **power-on → Sanctuary → Eastern
Palace room `0xA9`**, beyond the cannonball hall. C2 (Eastern prize) remains open. No item row beyond 1–4
of ORDER is earned; no bow, Eastern big key, pendant or credits claim.
STATUS is planner-owned and was not promoted.

## Continuous forward proof

`recordings/b9_exit_20261025/power_on_replay.json`: **47,930 frames**,
`route_goal_ok=true`, `synchronization_ok=true`, `clean_chain=true`,
`continuous=true`; **desync 0, state loads 0, rollout restores 0, RAM writes 0,
refills 0, deaths 0**. Blank SRAM was prepared before the audited power-on
run. Every committed frame matches the full-RAM CRC recorded in development.
The exact A9 pose, HP, sword/lamp, keys, story and preserved B9 inventory
are independently checked by `recordings/b9_exit_20261025/power_on.py`.

Canonical new tape: `recordings/b9_exit_20261025/power_on.tape.npz`.
SHA256: `15416156acfeda12cd92669fe0f442a860fffd8c967a5a790b7ab8195e359dbe`.
Next development predecessor: `Dev_B9PowerOnA9_20261025`, captured only after
that successful replay. A later state load is development; the tape is the
continuous proof. New stop `eastern_a9` is named in this artifact's metadata;
it is not yet registered as a `spine.hops` CLI stop.

The fresh development controller used 11,277 forecast restores, no RAM
writes, and took 682 frames for B9 from the actual power-on predecessor.
It reused the old lobby tape as a fixed-input policy and recorded fresh CRCs,
rather than waiving the failed old-tape join. The final replay uses none
of those restores. Existing Sanctuary/Eastern-entry tapes remain intact.

## Verified room segment

Controller: `alttp.dungeon.nav:reach_cannonball_hall_exit`.
Source `Play_eog9_7`: room `0xB9`, layer 0, `(4856,6104)`, control ready,
HP 28/32, sword/lamp 1, follower 0, keys 0. Source SHA256:
`dce836a15025bd5e9fa06d99a72393e0b366178beb96a65518c66b8d535de1ef`.

The controller composes the earned lower crossing, passes below the east
stair's side wall, climbs to layer 0 and takes the north door. Map geometry
stays in `maps/room_b9.json`. Full room: **714 frames**, HP 28 throughout.
Development used **13,491 forecast restores**, no RAM writes. Independent
forward replay: **desync 0, loads 0, restores 0, writes 0**.
Endpoint `Dev_B9Exit20261025_predecessor` is resumed development evidence.
Tape SHA256: `4029ada1fa0a412ba916d5adba0f7bcb05abc464df54fc50414b1dbb6f99b0ae`.

Leave glance: indoors room `0xA9`, module/submodule `0x07/0x00`, control
ready, layer 0, `(4856,5592)`, HP 28/32, sword/lamp 1, `$F3CC=0`, keys 0,
rupees 5, inherited big-key word 16384 (not an earned Eastern big key).
Screenshots confirm the big-chest hall's south entrance.

## Attempts and tape joins

Two geometry hypotheses. First: straight east at y=5744; stopped after
51 frames at `(4928,5744)` on the stair side wall, HP unchanged. Second:
bypass below the wall, then climb; 318 frames from `Dev_B9Channel20261024`
to A9, independently replayed with zero desync/loads/writes. The stair
sets submodule `0x08`, changes layer 1→0 at `(4952,5742)`, and returns
control at `(4952,5703)`. No global classification of byte `0x1E` changed.

The old 46,646-frame power-on Eastern-entry tape verifies, but its direct
join to the old 602-frame lobby tape desynchronizes at the first lobby
frame despite matching arrival coordinates. That failed join is retained;
it cannot establish continuity. Successful regeneration from the actual power-on
predecessor and its independent replay are recorded above. Original accepted tapes are untouched.

Evidence: `recordings/b9_exit_20261025/` holds attempt/controller/replay
reports, full-RAM CRC tapes, transition screenshots, source snapshots,
failed-join reports and the regeneration harness. Model-call/token usage
is not exposed by this execution environment. Assist contract:
`docs/ASSIST_CONTRACT.md`; no health refill, inventory or progression write.

## Next

Continue A9 from `Dev_B9PowerOnA9_20261025`, resolve its reachable route and
required item path before C2. Register the earned Eastern suffix in the
normal spine dispatcher when composing the next campaign leg. Preserve prior pins and tapes. No A, sword
swing or inventory write was needed for this B9 segment. The old south
stair return, 29-step straight line and knockback-gap claims remain withdrawn.

Tracker: `bd ready` reports no database; board is archived. No bead was
updated or replacement initialized. Focused nav/map/docs checks: 22 passed;
room-fixture/tilemap checks: 5 passed. ROM evidence is the replays above.
