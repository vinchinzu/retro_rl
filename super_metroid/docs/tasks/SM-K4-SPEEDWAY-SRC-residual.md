## Residual — SM-K4-SPEEDWAY-SRC

### Result
GREEN

### Files changed
- `super_metroid/docs/SOURCE_STATES.md` — registered pure Speedway successor row; removed Frog→Speedway geometry-controller gap.
- `super_metroid/docs/tasks/SM-K4-SPEEDWAY-SRC-residual.md` — catalog residual.

### Verify paste

```text
test -f super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state
exit 0
# file present: 100144 bytes (2026-08-01)

# pin metrics from SM-K4-FROG-SPEEDWAY-PURE residual (no re-run emulator):
room=0xB106 pose=11 x=39 y=139 door_transition=0 frames=295
sourceId=post_frog_continuous statePath=.../scratch/post_frog_save_to_speedway_pure.state

rg -n "post_frog_save_to_speedway|0xB106|Speedway" super_metroid/docs/SOURCE_STATES.md
exit 0
47:| `post_frog_save_to_speedway_pure` | `scratch/post_frog_save_to_speedway_pure.state` | `0xB106` Frog Speedway (reload: x=39/y=139/pose=11; door_transition=0) | continuous-like pure successor of Frog Save from `post_frog_continuous` (pure GREEN, frames=295) | pure `speedway-to-farm` / K4.2; **not** continuous tip |
162:| pure bubble mountain entry (K4 Speed) | `0xACB3` Bubble Mountain | needs continuous-like capture after Frog Save→Speedway→Farm; no source exists | SM-SRC-BUBBLE |
```

### Acceptance

- [x] SOURCE_STATES has a Speedway pure-successor row with repo-relative path
- [x] Room/fingerprint notes honest (GREEN pure only if state verified)
- [x] Residual next card ID + one change
- [x] Non-claims: not continuous evidence

### Residual risks

- Catalog only: no continuous re-record, graph edge, or STATUS promotion.
- Bubble Mountain / farm entry source still missing until Speedway→farm pure captures a successor.

### Next action (required)

- **Next card ID:** `SM-K4.2-PURE`
- **One change:** Implement pure Speedway → farm geometry from `post_frog_save_to_speedway_pure`.
- **Source state:** `custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_speedway_pure.state`

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin (if pure/geometry) — mandatory metrics

```text
room=0xB106 pose=11 x=39 y=139 door_transition=0
frames=295 dwell=not reported last_pin=room=0xB106 pose=11 x=39 y=139 door_transition=0
# metrics from predecessor pure residual; load path verified by file existence only
```
