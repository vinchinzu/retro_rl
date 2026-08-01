## Residual — SM-K4-R-04B (natural Varia lineage)

### Result

GREEN

### Verify paste

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure warehouse-to-business \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_continuous_to_warehouse.state
```

The right-ledge controller reaches ordinary Business:

```text
room=0xA7DE pose=155 x=128 y=65531 door_transition=0 frames=1421
```

The accepted route is a two-tier reverse stack traversal: left-spin from the
Zeela ledge for 120f, right-correct onto the lower lip (`x≈498/y≈315`), clear
the lower three-Super stack, take the two Hi-Jump ledges, clear the upper
left-facing stack, then reuse the normal elevator closeout. The left-side
power-on elevator path remains unchanged.

### Confirmed predecessor chain

- `post_varia_continuous_to_kihunter` → Kihunter→Zeela **GREEN** (1,759f)
- `post_varia_continuous_to_zeela` → Zeela→Warehouse **GREEN** (2,498f)
- `post_varia_continuous_to_warehouse` → Warehouse→Business **GREEN** (1,421f)
- Full accepted-Varia pure return chain → Business **GREEN** (9,343f)
- Power-on `--to business` → Business **GREEN** twice (113,723f; 0 loads /
  progression writes / capacity writes / deaths)

### Next action

- **Next card ID:** SM-K4-BUBBLE-PURE
- **One change:** Build `Business → Frog Save` from
  `scratch/post_business_continuous.state`.

### Non-claims

- No progression, capacity, door, event, boss, or room-state writes.
