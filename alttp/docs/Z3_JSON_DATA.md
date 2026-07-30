# ALTTP — z3-json-data integration

Local, **opt-in** access to [vg-json-data/z3-json-data](https://github.com/vg-json-data/z3-json-data)
for region/connection/item/enemy JSON. The tree is **not** vendored (no
upstream LICENSE at repo root when this was wired; do not commit a wholesale
copy). Fetch once into a gitignored path.

## Setup

```bash
uv run python alttp/scripts/setup_z3_json_data.py
```

| Field | Value |
|-------|--------|
| Local path | `alttp/refs/z3-json-data/` (gitignored) |
| Upstream | `https://github.com/vg-json-data/z3-json-data.git` |
| Pin | `1eb7a785bda0d671136316c24f223c7ce12257e6` |

Options: `--force` re-clone, `--revision SHA` override pin, `--dest PATH`
override location. Normal `import alttp.z3_json_data` **never** downloads.

## Provenance

- Source project: community ALTTP logic/region JSON (randomizer-oriented).
- Pin lives in `alttp.paths.Z3_JSON_DATA_PIN`.
- Shape checks confirm expected files/keys only; they do **not** validate the
  full upstream JSON Schema or in-game correctness.

## Usage

```bash
# Checkout status (no ROM/emulator)
uv run python -m alttp.z3_json_data status
uv run python -m alttp.z3_json_data validate

# Opening-route rooms (Link's House, castle courtyard/ledge, escape, …)
uv run python -m alttp.z3_json_data list-regions --opening
uv run python -m alttp.z3_json_data list-connections --opening
uv run python -m alttp.z3_json_data show-room "Links House"
uv run python -m alttp.z3_json_data list-items -q Sword
uv run python -m alttp.z3_json_data list-enemies -q Crow
```

Python:

```python
from alttp.z3_json_data import Z3JsonData

data = Z3JsonData.load()  # raises if not fetched
house = data.room("Links House")
for conn in data.connections_for_room(house):
    print(conn.origin, "->", conn.destination)
```

## Focus

Useful for the existing boot/opening route (title → Link's House exit →
Hyrule Castle grounds / secret entrance / escape). Full dark-world and late
dungeon graphs are loadable but not curated here.

## Opening-route catalog

`alttp.opening_route_catalog` maps the confirmed Link's House → castle goal
to expected z3 rooms/nodes/connections and gameplay RAM checkpoints, then can
emit a structured artifact (default
`alttp/recordings/opening_route_catalog.json`).

```bash
uv run python -m alttp.opening_route_catalog status
uv run python -m alttp.opening_route_catalog validate
uv run python -m alttp.opening_route_catalog list-checkpoints -v
uv run python -m alttp.opening_route_catalog emit
uv run python -m alttp.opening_route_catalog emit \
  --from-boot-report alttp/recordings/boot_to_castle.json
```

- Missing checkout → actionable error pointing at `setup_z3_json_data.py`
  (never auto-downloads).
- `--from-boot-report` attaches only milestones actually present in the boot
  JSON (final castle-grounds acceptance); it does **not** invent intermediate
  screen visits.
- Catalog validation is structural presence against the local pin, not
  gameplay proof.

## Limitations

- Not a second source of truth for RAM screen IDs or emulator routing
  (`overworld.py` / `startup.py` remain authoritative for the scripted path).
- z3 room/node names are randomizer logic labels; they are **not** exact
  stable-retro screen coordinates.
- Item `data` fields and node addresses are echoed from upstream; semantics
  are not re-verified against the ROM.
- Enemy HP/damage arrays are not mapped into combat policy.
- No GUI, no silent network I/O, no commitment that upstream schema versions
  stay compatible beyond the pin.
