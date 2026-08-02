## Residual — SM-ROOM-SEG-09

### Result
GREEN

### Files changed
- `policies/room_clears/room_9c5e_from_9ad9_to_9c89.json` — replaced the generated scaffold with the tested upper-platform, lower-route, and bottom-left-door sequence; sha-gated practice promotion is recorded.
- `docs/tasks/SM-ROOM-SEG-09-residual.md` — records the isolated green run, promotion, pin, and dual-track non-claims.

The problem-specific doorway fixture at `custom_integrations/SuperMetroid-Snes/room_9c5e_from_9ad9.state` already existed and was teleport-verified; it was not changed by this card.

### Verify paste
```text
uv run python scripts/room/run_problem.py teleport room_9c5e_from_9ad9_to_9c89
exit 0
problemId=room_9c5e_from_9ad9_to_9c89
room=0x9C5E game_state=8 phase=ordinary_gameplay
x=704 y=121 pose=2 door_transition=0
stateSha256=f5c0cfdb9f53e372a8d1f8bbbea57dd4187ebcb31bcf48f23eead64f26ed7bba

uv run python scripts/room/run_problem.py run room_9c5e_from_9ad9_to_9c89
exit 0
success=true
crossingFrame=650 settledFrame=768 totalFrames=768
finalRoom=0x9C89 finalPose=10 finalX=216 finalY=139 door_transition=0
progression_writes=0 capacity_writes=0 deaths=0
policyStatus=generated_unverified

uv run python scripts/room/run_problem.py run room_9c5e_from_9ad9_to_9c89 --promote
exit 0
success=true promoted=true
crossingFrame=650 settledFrame=768 totalFrames=768
finalRoom=0x9C89 finalPose=10 finalX=216 finalY=139 door_transition=0
progression_writes=0 capacity_writes=0 deaths=0
policyStatus=verified_development_state
reportPath=recordings/room_clears/room_9c5e_from_9ad9_to_9c89.json
```

### Acceptance
- [x] Isolated run **GREEN + promote** — the policy crossed to `0x9C89` and the sha-gated `--promote` run returned `promoted=true`.
- [x] Only own-files touched — only the target policy and this target residual were durable edits; the existing target fixture was unchanged. Required harness output remains development-only/generated.
- [x] Dual-track non-claim — this is isolated room practice only and is not continuous evidence.
- [x] Next card ID + one change filled — `none`; no follow-up change is required for this card.

### Residual risks
- The fixture is a doorway-natural development state loaded by the room-practice harness, not a natural predecessor state in a continuous run.
- The green result does not establish continuous integrity, STATUS, queue promotion, or KPDR route readiness.
- Unlimited energy/ammo assists were enabled by the room harness; the report records one missile refill write and zero progression/capacity writes.
- Practice promotion is complete; no remaining risk blocks this card's practice acceptance.

### Next action (required)
- **Next card ID:** none
- **One change:** No follow-up one-knob change is required for this room-practice card.
- **Source state:** `custom_integrations/SuperMetroid-Snes/room_9c5e_from_9ad9.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression, capacity, door, event, or boss RAM.
- Did not claim continuous green; this is dual-track isolated practice only.

### Probe pin (if pure/geometry)
room=0x9C89 pose=10 x=216 y=139 door_transition=0
frames=768 dwell=118 last_pin=room=0x9C89 pose=10 x=216 y=139 door_transition=0

Superseded scaffold RED pin: room=0x9C5E pose=42 x=284 y=313 door_transition=0.
