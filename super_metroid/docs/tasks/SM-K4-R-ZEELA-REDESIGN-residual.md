## Residual — SM-K4-R-ZEELA-REDESIGN

### Result
GREEN

### Files changed
- `routes/kpdr/kraid_return.py` — redesign `play_zeela_to_warehouse_return`
- `scripts/probe/zeela_reverse_redesign_probe.py` — geometry diagnostic probe
- `progression.py` — `zeela_to_warehouse_return` → `controller_dev`
- `scripts/probe/kpdr.py` — wire pure `warehouse-to-business` (R-04)
- `docs/tasks/SM-K4-R-ZEELA-REDESIGN-residual.md` — this residual
- `docs/tasks/SM-K4-R-04-residual.md` — R-04 RED handoff
- `docs/SOURCE_STATES.md` / `docs/tasks/QUEUE.md` — live board

### Verify paste
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure zeela-to-warehouse-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_zeela_to_warehouse_return.state
```

Exit 0, multi-run green (~1800f):

```text
success: true
roomIdHex: 0xA6A1
samusX: 728
samusY: 139
frames: 1800
controllerOnly: true
```

Second pure without `--output` also green ~1800f.
`uv run pytest super_metroid/tests/test_controller_common.py -q` → 14 passed.

### Geometry (why R-03 cadence failed)

| Fact | Detail |
|------|--------|
| No Hi-Jump on reverse pure sources | `equipped_items=0x1005` (Varia+Morph+Bombs only) |
| Mid platform | narrow x≈90–107, y≈331 — reverse-shot + **RIGHT bias in hole** |
| Left-only reverse-shot | peaks shaft at x≈52, never lands mid |
| Below-platform lip | crouch-load from mid right-edge x≈107 → x≈69 y≈235 |
| Lip walk left blocked | hop-left to plant x≈37 y≈219 |
| Shot blocks respawn on reverse re-entry | without ~40 UP+X clear, wall climb caps y≈188 |
| Top door band | y≈112–139; firm land pose 1/2/137/138 (not walljump 26) |
| Floor-left | Energy Tank `0xA4B1` — fail-loud y>250 |

### Maneuver class (working)

1. Bottom morph-roll → second-drop lane x≈110–140
2. Reverse-shot + RIGHT bias → middle platform
3. Mid right-edge crouch-load LEFT → lip
4. Hop-left → wall plant x≈37 y≈219
5. Clear shot blocks (UP+X ×40) + left-wall spin climb → y≤150 grounded
6. Standing LEFT beams → Warehouse `0xA6A1`

### Acceptance
- [x] Pure green → ordinary `0xA6A1`
- [x] Floor-door guard retained
- [x] Source captured `post_zeela_to_warehouse_return.state`
- [x] Graph `controller_dev` (not continuous)
- [x] No STATUS claim

### Next action (required)
- **Next card ID:** SM-K4-R-04 (residual RED — see `SM-K4-R-04-residual.md`)
- **One change:** Warehouse right-door (x≈728) → elevator left traversal; super-stack opens from left only
- **Source state:** `scratch/post_zeela_to_warehouse_return.state` room `0xA6A1`

### Non-claims
- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
