## Residual — SM-ROOM-SEG-21

### Result
RED

### Files changed
- `custom_integrations/SuperMetroid-Snes/room_d6d0_from_d8c5.state` — doorway-natural entry fixture for `0xD6D0` from `0xD8C5`.
- `custom_integrations/SuperMetroid-Snes/room_d6d0_from_d8c5.provenance.json` — bootstrap provenance and entry contract for this fixture.
- `policies/room_clears/room_d6d0_from_d8c5_to_d8c5.json` — generated room policy scaffold; remains unverified.
- `docs/tasks/SM-ROOM-SEG-21-residual.md` — isolated-run residual and next action.

### Verify paste
```text
uv run python super_metroid/scripts/room/run_problem.py teleport room_d6d0_from_d8c5_to_d8c5
exit code: 0
stdout (relevant):
{"state":{"frame":1,"game_state":8,"phase":"ordinary_gameplay","room_id_hex":"0xD6D0","samus_x":64,"samus_y":121,"pose":1,"door_transition":0,"morph_ball":true,"bombs":true,"max_health":199,"max_missiles":10,"max_super_missiles":0,"max_power_bombs":0}}
stderr: empty

uv run python super_metroid/scripts/room/run_problem.py run room_d6d0_from_d8c5_to_d8c5
exit code: 1
stdout (relevant):
{"success":false,"failure":"room objective incomplete: collected_items did not change","crossingFrame":786,"settledFrame":938,"totalFrames":938,"targetRoomIdHex":"0xD8C5","finalState":{"room_id_hex":"0xD8C5","samus_x":984,"samus_y":139,"pose":10,"door_transition":0,"collected_items":4100},"assist":{"progression_writes":0,"capacity_writes":0},"policy":{"status":"generated_unverified"}}
stderr: empty
```

The natural post-Spore fixture has Morph Ball, Bombs, and Missiles, but no
Gravity Suit or X-Ray. The room's canonical item route requires Gravity plus
the bomb route; the alternate suitless R-jump route requires X-Ray and a
specialized floor clip. The scaffold reaches the return door but never
collects Spring Ball.

### Acceptance
- [x] Isolated run produced an honest residual with a probe pin; no promote was attempted after the RED run.
- [x] Only this problem's fixture, provenance, policy, and residual were changed by this task; the ignored room report is generated verification output.
- [x] Residual states the dual-track non-claim.
- [x] Next card ID and one change are filled below.

### Residual risks
- The policy has no GREEN isolated clear and remains `generated_unverified`; practice promotion is unavailable.
- The current doorway-natural source lacks the capabilities needed to reach the Spring Ball PLM without an equipment/progression forge.
- The return-door crossing is not item-clear evidence: `collected_items` stayed at `0x1004`.
- This is not pure-green, continuous evidence, STATUS evidence, or a natural-entry full-run result.

### Next action (required)
- **Next card ID:** SM-ROOM-SEG-21-R1
- **One change:** Capture one controllable doorway-natural source with the capabilities required for the Spring Ball item route, without writing progression or capacity RAM.
- **Source state:** needs capture: `SM-ROOM-SEG-21-SRC`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Not continuous evidence; this is dual-track room practice only.

### Probe pin
room=0xD8C5 pose=10 x=984 y=139 door_transition=0
