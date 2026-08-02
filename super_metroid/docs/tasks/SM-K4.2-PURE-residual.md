## Residual — SM-K4.2-PURE

### Result
RED

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — replaced `play_speedway_to_farm` scaffold with bounded RIGHT+B+X door hold; timeout names boost-block stall + `max_x`.
- `super_metroid/routes/kpdr/registry.py` — registered `speedway_to_farm`.
- `super_metroid/routes/kpdr/__init__.py` — export `play_speedway_to_farm`.
- `super_metroid/scripts/probe/kpdr.py` — pure choice `speedway-to-farm`.
- `super_metroid/tests/test_k4_norfair_scaffold.py` — registration contract for `speedway_to_farm`.
- `super_metroid/docs/tasks/SM-K4.2-PURE-residual.md` — this residual.

### Verify paste

```text
uv run python super_metroid/scripts/probe/kpdr.py pure speedway-to-farm \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_speedway_to_farm_pure.state \
  --pin-json super_metroid/debug/speedway_to_farm_pure_pin.json
exit 1
success=false error="speedway_to_farm: right door missed before room 0xAF72; room=0xB106 pose=137 xy=(795,139) door_transition=0 max_x=795 (boost-block stall; no Speed)"
roomIdHex=0xB106 samusX=795 samusY=139 pose=137 doorTransition=0 frames=1100 controllerOnly=true
collected_items=0x1105 (Varia+Morph+HiJump+Bombs; no Speed)
pinJson=super_metroid/debug/speedway_to_farm_pure_pin.json
redDiag=super_metroid/debug/red_diag/20260802T020715Z_speedway-to-farm

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
exit 0
6 passed in 0.14s
```

### Acceptance

- [x] Source fingerprint loads at `0xB106`.
- [ ] Pure controller reaches ordinary `0xAF72` without placement, warp, or item grants.
- [ ] Successor source captured only if pure GREEN.
- [x] Focused unit test green.
- [x] Residual PROCESS fields; next card ID + one change; no continuous/STATUS claim.

### Residual risks

- Mid-room Boost Blocks solid-lock at **max_x=795** with continuous loadout `0x1105` (no Speed). Flat run, offline Hi-Jump pulses, and morph-roll all fail to exceed x=795.
- sm-json left→right Base strat requires `h_getBlueSpeedMaxRunway`; non-Speed options are glitch/G-Mode/Grapple only — out of pure continuous caps.
- Chicken-and-egg vs K4 order: Speed is past Bubble, but Frog Speedway needs Speed to reach Farm/Bubble via this tunnel.
- No successor state written (pure RED).

### Next action (required)

- **Next card ID:** `SM-K4.2-PURE-R1`
- **One change:** At stall band x∈[780,820], add a named reverse→spin-jump / morph wall-probe phase that fails closed if `max_x` stays ≤795 (prove or disprove a gap); do **not** grant Speed; do not edit `frog_save_to_speedway`.
- **Source state:** `custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state`

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM (no Speed grant).
- Not continuous evidence.
- Did not re-record continuous tip or edit `continuous.py` / catalog / progression verification.

### Probe pin (if pure/geometry) — mandatory metrics

```text
room=0xB106 pose=137 x=795 y=139 door_transition=0
frames=1100 dwell=not reported last_pin=room=0xB106 pose=137 x=795 y=139 door_transition=0
max_x=795 (boost-block stall; no Speed)
pin=super_metroid/debug/speedway_to_farm_pure_pin.json
redDiag=super_metroid/debug/red_diag/20260802T020715Z_speedway-to-farm
snapshot=super_metroid/debug/red_diag/20260802T020715Z_speedway-to-farm/door_plm_snapshot.json
frameDumpDir=super_metroid/debug/red_diag/20260802T020715Z_speedway-to-farm/frames
```
