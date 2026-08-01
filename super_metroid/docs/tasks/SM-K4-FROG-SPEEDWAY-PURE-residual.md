## Residual — SM-K4-SPEEDWAY-PURE

### Result

RED

### Files changed

- `routes/kpdr/k4_norfair.py` — replaces the unarmed Frog Save scaffold with
  the bounded `frog_save_to_speedway_door` beam-shot right-door attempt.
- `routes/kpdr/__init__.py` — exports the one Frog Save → Speedway controller.
- `routes/kpdr/registry.py` — registers `frog_save_to_speedway` for pure use.
- `scripts/probe/kpdr.py` — exposes controller-only
  `pure frog-save-to-speedway`.
- `tests/test_k4_norfair_scaffold.py` — locks the pure registry binding.
- `docs/tasks/SM-K4-FROG-SPEEDWAY-PURE-residual.md` — records this source-backed
  RED result.

### Verify paste

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure frog-save-to-speedway \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_continuous.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state \
  --pin-json super_metroid/debug/frog_save_to_speedway_pure_pin.json
# exit 1
# success=false roomIdHex=0xB167 frames=400 controllerOnly=true developmentOnly=false
# error=frog_save_to_speedway: right door missed before room 0xB106;
#       room=0xB167 pose=137 xy=(107,171) door_transition=0

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
# exit 0; 5 passed

uv run ruff check super_metroid/routes/kpdr/k4_norfair.py \
  super_metroid/routes/kpdr/__init__.py \
  super_metroid/routes/kpdr/registry.py \
  super_metroid/scripts/probe/kpdr.py \
  super_metroid/tests/test_k4_norfair_scaffold.py
# exit 0; All checks passed!
```

### Acceptance

- [x] The cataloged `post_frog_continuous` fingerprint loaded in `0xB167`.
- [ ] The pure controller reached ordinary `0xB106` without placement or warp.
- [ ] No successor state was captured; the requested output path is written only
  after a successful probe.
- [x] The focused import/registry test is green.
- [x] This residual has one next card, one change, and no continuous or STATUS
  claim.

### Residual risks

- The current source starts at the Frog Save reload band but the full-height
  beam-shot right run stalls at `x=107, y=171`; it never reaches a transition.
- The failed controller must not promote `frog_save_to_speedway` from
  `unverified`, and it cannot extend `continuous.py`.

### Next action (required)

- **Next card ID:** SM-K4-FROG-SPEEDWAY-R1
- **One change:** Replace only the `frog_save_to_speedway_door` full-height
  right-run phase with a bounded right-door approach that first reaches a
  measured door-height band before beam-shooting.
- **Source state:** `scratch/post_frog_continuous.state` (expected `0xB167`,
  reload x=60/y=139/pose=1).

### Non-claims

- Did not STATUS-promote or change graph, catalog, or continuous-tip wiring.
- Did not forge progression, capacity, door, event, boss, or room-state RAM.
- Not continuous evidence; the pure probe is RED.

### Probe pin

room=0xB167 pose=137 x=107 y=171 door_transition=0
frames=400
dwell=n/a
last_pin=room=0xB167 pose=137 x=107 y=171 door_transition=0
