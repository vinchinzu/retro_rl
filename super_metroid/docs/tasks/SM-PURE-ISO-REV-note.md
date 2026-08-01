# SM-PURE-ISO-REV Harness Note

`kpdr.py iso-reverse` prints the cataloged reverse-hop commands below. These
are controller-only diagnostics. A successful command is not, by itself,
pure-green evidence: it does not establish natural predecessor entry, zero
state loads, zero progression writes, or continuous-tip integrity.

## Command Matrix

Run from the monorepo root. Each source path is relative to the repository
root, and the expected room is the room at probe start.

| Hop | Segment | Expected start room | Exact command |
|---|---|---:|---|
| K3.3 | `kraid-to-eye-return` | `0xA59F` | `uv run python super_metroid/scripts/probe/kpdr.py pure kraid-to-eye-return --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state` |
| K3.4 | `eye-to-baby-return` | `0xA56B` | `uv run python super_metroid/scripts/probe/kpdr.py pure eye-to-baby-return --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kraid_to_eye_return.state` |
| K3.5 | `baby-to-kihunter-return` | `0xA521` | `uv run python super_metroid/scripts/probe/kpdr.py pure baby-to-kihunter-return --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_eye_to_baby_return.state` |
| K3.6 | `kihunter-to-zeela-return` | `0xA4DA` | `uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state` |

The same matrix is available without copying commands:

```bash
uv run python super_metroid/scripts/probe/kpdr.py iso-reverse
```

## Smoke Results

No emulator smokes were run for this harness-only change. The source states
are gitignored runtime artifacts, and this card does not claim geometry green.
The CLI and matrix smoke output were verified:

```text
uv run python super_metroid/scripts/probe/kpdr.py pure --help
exit 0; reverse segment choices include kraid-to-eye-return,
eye-to-baby-return, baby-to-kihunter-return, and kihunter-to-zeela-return

uv run python super_metroid/scripts/probe/kpdr.py iso-reverse
exit 0; printed exact K3.3–K3.6 commands and expected source rooms
```

Run each exact command above when validating a geometry card and record the
last room, pose, coordinates, and door-transition result in that card's
residual.

## Residual — SM-PURE-ISO-REV

### Result
GREEN

### Files changed
- `scripts/probe/kpdr.py` — added the read-only `iso-reverse` command matrix.
- `docs/tasks/SM-PURE-ISO-REV-note.md` — documented K3.3–K3.6 source commands and diagnostic limits.

### Verify paste
- `uv run python super_metroid/scripts/probe/kpdr.py pure --help` — exit 0; listed all four reverse segment choices.
- `uv run python super_metroid/scripts/probe/kpdr.py iso-reverse` — exit 0; printed K3.3–K3.6 commands with expected rooms `0xA59F`, `0xA56B`, `0xA521`, and `0xA4DA`.
- `python -m py_compile super_metroid/scripts/probe/kpdr.py` — exit 0.

### Acceptance
- [x] Documented command matrix in note.
- [x] No controller geometry edits.
- [x] Residual next: R-02B if Zeela remains red.

### Residual risks
- The K3.3–K3.6 commands have not been emulator-smoked in this harness card.
- K3.6 remains geometry RED until the `kihunter-to-zeela-return` card proves the
  lower-alcove shot-block climb.
- Pure diagnostics do not establish continuous integrity or permit STATUS promotion.

### Next action (required)
- **Next card ID:** R-02B
- **One change:** Resolve the single K3.6 lower-alcove shot-block climb issue.
- **Source state:** `scratch/post_baby_to_kihunter_return.state`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin
room=N/A (smokes not run) pose=N/A x=N/A y=N/A door_transition=N/A
