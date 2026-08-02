## Residual — SM-ARCH-RED-DIAG

### Result
GREEN (tooling / diagnostics card — not a geometry pure)

### Files changed
- `super_metroid/scripts/probe/red_diag.py` — pure RED frame-dump + door/PLM snapshot helpers; repo-relative paths
- `super_metroid/scripts/probe/kpdr.py` — pure RED auto-capture (ring buffer + pin `redDiag`); `--no-red-diag` opt-out; green success path unchanged
- `super_metroid/tests/test_red_diag.py` — unit coverage without emulator fight
- `super_metroid/docs/tasks/SM-ARCH-RED-DIAG-residual.md` — this residual

### Verify paste
```bash
uv run pytest super_metroid/tests/test_red_diag.py -q
# ........                                                                 [100%]
# 8 passed in 0.16s
# exit 0

rg -n "pin.json|PLM|door_transition|clip|snapshot" \
  super_metroid/scripts/probe/kpdr.py super_metroid/scripts/probe/ | head -40
# exit 0  (matches red_diag + kpdr pure RED capture hooks)

uv run python super_metroid/scripts/probe/kpdr.py list
# exit 0  (import path OK after red_diag hook)
```

Manual pure RED exercise (optional; not run this session — no geometry claim):
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure <segment> \
  --source <continuous-like.state> \
  --pin-json super_metroid/debug/<segment>_pin.json
# On failure: report + pin include redDiag.snapshotPath / frameDumpDir /
# framePaths under super_metroid/debug/red_diag/<stamp>_<segment>/
```

### Acceptance
- [x] RED path produces artifact paths in pin or residual schema (`redDiag` keys)
- [x] Green pure path unchanged (success return has no redDiag side effects; capture only on `success=False`)
- [x] Residual next card ID + one change

### Residual risks
- Live PLM-record / blue-door-open WRAM still **blocked** (snapshot peeks validated door_def_ptr + door_transition + nav pin only; same gap as `kraid_door_plm_recon`).
- Frame dump is last ~45 RGB frames (or 3 idle frames on fingerprint fail), not a full-run video; `clipPath` reserved if ffmpeg clip is added later.
- Auto-capture best-effort: `redDiagError` may appear if PNG write fails; original pure error still primary.

### Next action (required)
- **Next card ID:** SM-K4-FROG-SPEEDWAY-R1 (or planner pure pick if frog re-verify not needed)
- **One change:** On pure RED, residual must cite `redDiag.snapshotPath` + `frameDumpDir` from auto-capture (do not debug dark).
- **Source state:** continuous-like source for the pure under test (see `docs/SOURCE_STATES.md`)

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM
- Not continuous evidence
- Did not change geometry controllers, continuous tip tables, combat primitives, or hop timings
- Did not claim any pure segment green

### Probe pin (if pure/geometry) — **mandatory metrics**
room=n/a pose=n/a x=n/a y=n/a door_transition=n/a
frames=n/a
dwell=n/a
last_pin=n/a (diagnostics card; unit-tested writer only)

### RED artifact schema (for next agent)
```json
{
  "redDiag": {
    "outDir": "super_metroid/debug/red_diag/<stamp>_<segment>",
    "snapshotPath": "super_metroid/debug/red_diag/.../door_plm_snapshot.json",
    "frameDumpDir": "super_metroid/debug/red_diag/.../frames",
    "framePaths": [".../frames/frame_000.png", "..."],
    "frameCount": 45,
    "clipPath": null,
    "pinPath": "super_metroid/debug/red_diag/.../pin.json",
    "manifestPath": "super_metroid/debug/red_diag/.../red_diag_manifest.json",
    "medium": "frame_dump"
  },
  "residualArtifactLine": "snapshot=... frames=N frameDumpDir=..."
}
```
