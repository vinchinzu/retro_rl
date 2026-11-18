# Maps

Run `uv run python -m snes.chrono_trigger maps` at the repository root.
`generated/locations.json` is a local ROM-derived exit graph. Records cover
all 512 IDs, including unused and non-field locations; the 814 exits are
static records, not 814 verified routes. Each edge starts `verified: false`.

Each run also produces `recordings/<run>/atlas.html` and `atlas.json`:

- Field layer-1 metatile IDs, raw attributes A/B, and exit indices, read
  with the correct 256-byte row stride and current map dimensions.
- Overworld layer-1 metatiles (96 by 64) and loaded entrance trigger table.
- Visited tiles, last positions, and observed scene changes for that run.

The HTML viewer is local and self-contained. Green means actually visited;
blue marks static exits or active overworld entrances. Dark tiles have
both half-tile “no level” bits set. Gray tiles may still be blocked by
level, shape, a wall, an NPC, or an event. Use movement evidence to plan
waypoints; an attribute color is not a traversability guarantee.

Keep ROM-derived maps under ignored `generated/` or `recordings/`.
The extractor and viewer source live in `../maps.py`.
