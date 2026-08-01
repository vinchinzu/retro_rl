# TASK SM-K4-R-03: Pure `zeela-to-warehouse-return` (after R-02 source)

## Recipe step
1 pure controller

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_return.py` (`play_zeela_to_warehouse_return` only)

## Context
- **Unblocked:** SM-K4-R-CLIMB-REDESIGN pure green ~1716f; source captured
  `scratch/post_kihunter_to_zeela_return.state` (room `0xA471`, x≈403 y≈362)
- Scaffold exists: left spin-push toward Warehouse `0xA6A1`
- Forward geometry hints: `play_warehouse_to_zeela` reverse of morph tunnels
- Graph edge `zeela_to_warehouse_return` still `unverified`
- Do **not** retouch `play_kihunter_to_zeela_return` (Wave 9 green)

## Do
1. Pure-tune exit left into Warehouse Entrance from Zeela source only
2. Optional `--output` `scratch/post_zeela_to_warehouse_return.state`
3. No graph promote / continuous / STATUS

## Acceptance
- [ ] Pure green from post-R-02 source → `0xA6A1`
- [ ] Residual schema

## Verify
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure zeela-to-warehouse-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state
```
