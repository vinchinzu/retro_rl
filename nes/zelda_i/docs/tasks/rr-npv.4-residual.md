# rr-npv.4 — Clean L8 Entrance→TF no retopup

Fixture-live only. `route_eligible=false`. Do not STATUS. Do not close the bead.

## Landed this sitting

- Replaced 0x5E waist-clamp / UP-refusal with active side-stepping, entry-column peel, flank/rear attacks, and pulsed sword thrusts (`_slash`).
- Eliminated phantom obstacle accumulation inland: cleared inferred blocks during open-floor combat and set `last_dir=None` on combat maneuvering.
- Fixed side-step direction logic to step monotonically away from enemy centerlines instead of oscillating across hardcoded coordinates.
- Room 0x5E is **completely cleared** under live Clean play: all 5 type 0x0C Darknuts killed with 0 deaths, center small key picked up, exited through north shutter door into 0x4E, unlocked north door with key, and reached room 0x3E.
- 237 passed across all `test_level8*.py` unit tests.

## ROM glance — first red (stop, new leftover)

One trial. `Level8InteriorReconFixture` play `0x7E` `(120,205)` TF `0x7F`,
`--from-enter --clean --no-video`. Tag `l8clr_lab`.

```text
[0] level8_north_manhandla_bomb: succ=True failed=False f=1700 -> 0x5e [120, 189] m5 hc=3 notes=['arrived_0x5e_120_189']
[1] level8_darknut_key_up: succ=False failed=True f=1723 -> 0x3e [72, 166] m17 hc=3 notes=['link_death']
```

Final: L8 `0x3e` `(72,166)` mode 17 TF `0x7F` MK 0 keys 9 bombs 7 rupees 255
hc 3 health `0x20`. Deaths 1. Writes 0.

**0x5E Blocker Resolved**: The 1107f death in room 0x5E at `(104,117)` is resolved.
Link clears 0x5E, collects the key, transitions through 0x4E, and enters 0x3E.
Died at frame 1723 at `(72,166)` in room 0x3E against Blue Darknuts. Stop predicate reached (1 ROM trial completed, stopped at first red).

## Leftover

Clean 0x3E Blue Darknuts clear without infinite life. Entry to 0x3E from 0x4E south
door at keys 9, bombs 7, hc 3. Next: 0x3E combat policy tuning against Blue Darknuts
and Bomb-N wall navigation to 0x2E.

## Blockers

- Room 0x3E Blue Darknuts combat under live Clean health (`hc=3`, no retopup).
- Full Entrance→TF without retopup now reaches 0x3E (previously blocked at 0x5E).
