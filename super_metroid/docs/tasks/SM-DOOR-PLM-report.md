# SM-DOOR-PLM Diagnostic Report

## Result

**Blocked: no safe live PLM or blue-door-open field was added.** The bounded
search found no source-confirmed or differential WRAM offset that can be
exposed in `ram.py` without inventing a field. No RAM constants or parser
fields were changed.

This is harness diagnostics only. It is not pure-green evidence, continuous
evidence, or a STATUS promotion.

## Searched

- `ram.py`, `tests/test_ram.py`, and `docs/ram_map.md`
- `docs/tasks/SM-DOOR-BLUE-report.md`
- Existing Kraid probes: `kraid_door_blue_recon.py`,
  `kraid_door_phase_recon.py`, and `kraid_left_door_recon.py`
- Local emulator integration and comments
- Monorepo-wide source/comments search for door, PLM, BTS, and projectile
  offsets

The generated room catalog contains ROM-side PLM IDs, but those IDs do not
identify a live WRAM PLM record or activation byte. The existing live fields
remain the only bounded observations: `door_transition` (`0x0797`),
`transition_direction` (`0x0791`), and `door_definition_ptr` (`0x078D`).

## Probe

The new `scripts/probe/kraid_door_plm_recon.py` wrapper reuses the established
read-only Kraid approach, four standing left shots, fuse waits, and bounded
spin-push. It samples every frame using the already validated fields and marks
the missing PLM/open-state sample table as empty rather than labeling an
unvalidated byte as a field.

Command:

```bash
uv run python super_metroid/scripts/probe/kraid_door_plm_recon.py \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state
```

Observed run output:

```text
frames=1200 rooms=['0xA59F'] door_transition!=0=False output=super_metroid/debug/kraid_door_plm_recon.json
last_pin={'room': '0xA59F', 'pose': 138, 'x': 37, 'y': 395}
field_status=blocked new_fields=[]
reason=no validated live PLM or blue-door-open WRAM offset
```

## Field Status

| Field | Status | Reason |
|---|---|---|
| Blue-door open/state-machine byte | blocked | No validated live WRAM offset |
| PLM record/activation byte | blocked | No validated live WRAM offset |
| Door BTS/tile collision metadata | blocked | No validated live WRAM offset |
| Existing `door_transition` | available | Already parsed at `0x0797`; remains zero in prior recon |
| Existing `door_definition_ptr` | available | Already peeked at `0x078D`; identifies a definition, not open state |

## Non-Claims And Residuals

- No new field was added and no fake field was sampled.
- No door, PLM, progression, capacity, event, boss, room, or position RAM was
  written.
- The probe cannot distinguish a closed door from a missed trigger geometry.
- Last pin: room `0xA59F`, pose `138`, `x=37`, `y=395`.
- No new-field shot sample table exists because no safe field was identified;
  the JSON contains the existing-field every-frame samples.

## Planner Next

Planner should commission external Super Metroid RAM-map/reverse-engineering
work to identify and differential-validate one live PLM or blue-door state
offset before another geometry card.
