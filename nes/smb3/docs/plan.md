# Plan

## Goal

Move from the verified World 1-1 clear toward a continuous World 1, then a
full scripted clear.

## Next

1. World 1-2 natural-entry clear. This is the documented gate. A 1-2 policy
   and `run_level1.py --level 1-2` exist. This file does not record that
   clear as verified.
2. World 1-3 natural-entry from `AfterLevel2`, only after 1-2 is verified.
3. World 1 chain from 1-1 through the fortress, without warp assists.
4. Wider RAM (cards, inventory, stage id) as later worlds need it.

## Notes

- Platform is NES fceumm. Shared ROM root is `roms/Nintendo/NES/`.
- Policies are `policies/level1_1.json` and `policies/level1_2.json`.
  Re-optimize from natural entry if boot or map timing changes.
