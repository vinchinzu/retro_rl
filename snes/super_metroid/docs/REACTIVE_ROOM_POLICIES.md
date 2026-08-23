# Reactive room policies

Reactive room policies are the room-local control layer between human tapes
and route automation. They do not hill-climb a multi-room frame array. Each
policy contains one or more sparse reference trajectories and equipment
contracts, then projects the live Samus state onto the best trajectory.

**Repertoire spine:** practice-hack preset sessions (`practice_repertoire.py`,
product category `kpdr25`) name the ordered pins for **policy tune → graduate**,
**path stitch**, and **autopilot recovery** when no skill is loaded. See
[PRACTICE_ROM.md](PRACTICE_ROM.md).

The live feature vector includes position, X/Y velocity, X momentum, pose,
facing, movement type, vertical direction, inventory, current room, prior
room, and intended exit. This makes one room skill reusable after different
door speeds and after a human hands control back in the middle of the room.

## Runtime model

- Identical actions are stored as timed spans capped at eight frames.
- During a span, the action is cached; there is no RAM parse or trajectory
  search in `RoomAutopilot`.
- At a span boundary, bounded forward projection corrects ordinary drift.
- At takeover or large drift, global projection selects the closest point
  across every applicable take.
- If that score is still poor, a short frame-costed beam planner searches
  timed 1/2/4-frame input pulses from the exact emulator state. It restores
  the live state before returning the adapter plan.
- The registry loads verified policies by default. Missing rooms fall through
  to human control instead of emitting blind input.

This keeps the execution loop suitable for 60 FPS interactive play and more
than 300 FPS headless replay. The Climb policy compiled on 2026-08-12 measured
651–769 FPS in dual live-anchor verification, including state parsing and the
practice assist. The compiler enforces 300 FPS with `--min-fps 300` by default;
pass `--min-fps 0` only when collecting diagnostics on a slower machine.

## Compile and verify Climb

Base physics (Morph, explicitly no Hi-Jump):

```bash
uv run python snes/super_metroid/scripts/tools/optimize_room_policy.py \
  --body snes/super_metroid/tasks/full_start_v1_hops/hop_09_Climb.json \
  --room 0x96BA --from-room 0x975C --exit-room 0x92FD \
  --variant base --takeover-sweep
```

Hi-Jump physics from the late-game tape:

```bash
uv run python snes/super_metroid/scripts/tools/optimize_room_policy.py \
  --task snes/super_metroid/tasks/g4_tourian_human_mb.json --hop 5 \
  --room 0x96BA --from-room 0x975C --exit-room 0x92FD \
  --variant hi_jump --takeover-sweep
```

The first command creates the room policy. The second merges another
equipment variant into the same JSON instead of replacing it. A variant is
invalidated when its source trajectory changes; the policy returns to
`verified_live_anchor` only after every variant has its own dual-green report.

`--takeover-sweep` engages at 25%, 50%, and 75% of the source hop after four
extra idle frames have changed timing and kinematics. Use
`--takeover-perturb N` to change that disturbance. Candidate policies can be
written with `--no-verify`, but normal autopilot will not load them.

## Red Tower: checkpoint tree, not one tape

Red Tower `0xA253` uses a room-local checkpoint graph because the Ice
platforms move with live Ripper patrol phase and the room is ten screens tall.
The readable source is
`routes/kpdr/data/red_tower_ice_checkpoint_plan.json`; it separates verified
edges, observed human seats, planned edges, and fall-recovery funnels.

The first implemented edge is deliberately small:

```text
bottom_floor ~(216,2443)
  → track lowest Ripper X and freeze in a bounded launch band
  → right-wall spin + consecutive WJ (20/4/8, 14/2/6)
  → steer back to frozen support
  → lower_ripper_1 y=2351, grounded, freeze timer ≥30
  → freeze r2 at offset (UP+X, no walk), standing Hi-Jump, drift from above
  → lower_ripper_2 y=2255, grounded, freeze timer ≥30
  → freeze r3 on the facing (right) side, standing Hi-Jump, drift from above
  → lower_ripper_3 y=2159, grounded, freeze timer ≥30
  → freeze r4 on the facing (right) side, crouch-jump (136px), drift from above
  → lower_ripper_4 y=2023, grounded, freeze timer ≥30
```

Edge 1 waits for horizontal clearance before its first rising wall arc, is
dual-exact from the natural Bat→Red predecessor, and passed 31 patrol phases
total (offsets `0..240`, step 8) in **233–380 policy frames**. Edge 2 is
dual-exact **156f** ×2 from the Ice-pin r1 pin. Edge 3 is dual-exact
**108f** ×2 from the Ice-pin r2 pin (wait for r3 to pass, then freeze
right-side). Edge 4 is dual-exact **141f** ×2 from the Ice-pin r3 pin
(standing hop is ~3px short; crouch-jump clears). p165 chain bottom→r4
is **809f**. Hellway is not claimed.

```bash
uv run python snes/super_metroid/scripts/probe/red_ice_climb.py --edge 4 \
  --source snes/super_metroid/scratch/red_ice_p165_ripper3.state
uv run python snes/super_metroid/scripts/probe/red_ice_climb.py --edge chain \
  --source snes/super_metroid/scratch/bat_zero_settle_eq216_leave.state \
  --phase-offsets 0
```

**Human-only for product autopilot.** Red Tower Ice/WJ is not wired into
`RoomAutopilot` — no route-specific checkpoint hardcode. Climb with
`./play` / guided human (or the probe scripts below). A verified reactive
room policy may re-enter AP later; until then AP falls back to human and
may only show a repertoire recovery hint.

```bash
uv run python snes/super_metroid/scripts/probe/red_ice_climb.py --save
uv run python snes/super_metroid/scripts/export/red_ice_route_plan.py
```

Local visual plans (ROM-derived PNGs remain gitignored):

- `docs/tasks/refs/red_tower_ice_first_edge.png`
- `docs/tasks/refs/red_tower_ice_checkpoint_plan.png`

## Human hot-swap

Run the normal recorder:

```bash
./snes/super_metroid/play
```

It starts in human mode. Press backquote on the keyboard, or L+R+Select on a
controller, to engage autopilot at the current frame. Use the same control to
return immediately to human input. The HUD shows the selected policy,
equipment variant, projection score, and adapter status.

Development flags:

- `--no-autopilot` disables policy loading.
- `--autopilot-candidates` permits unverified policies.
- `--autopilot-policy-dir PATH` selects another registry directory.

## Room-by-room and randomizer use

Add multiple trajectories under one variant to preserve alternate human or
planned takes. On attach, the controller selects the best live kinematic fit;
on later feedback boundaries it may switch takes if recovery requires it.
Policies are keyed by route, room, predecessor room, exit room, and equipment
contract. `sm_rando` and `smz3` can therefore reuse the policy format while
providing different route IDs, door graph choices, and inventory contracts.

The optimizer's planned unit remains one natural-entry room hop. Route-level
code chooses the next exit; the room adapter finds a minimum-frame bounded
rejoin. This separation avoids fragile open-loop optimization across several
rooms and keeps policy failures local and diagnosable.
