"""Write a ``Whatif_*`` copy of a pin with named inventory fields changed.

The source pin file is not written. Each ``--set`` names a field from
``ram.INVENTORY_FIELDS`` (``sword=2``, ``hp=24``). The copy's provenance
kind is ``whatif`` unless ``--kind poke`` is set. ``$F3CC`` is not an
inventory field and is never written. No step after the load.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/whatif_pin.py \\
        CastleMain --set sword=2 --out Whatif_CastleMain_sword2
"""

from __future__ import annotations

import argparse
import sys

from alttp.paths import GAME, GAME_DIR
from alttp.ram import FOLLOWER, INVENTORY_FIELDS, InventoryField, assign_u8, read_inventory
from alttp.spine.tape import read_pin_provenance, write_pin_provenance
from retro_harness.env import make_env, read_state_bytes, save_state, state_path
from retro_harness.segment_runner import configure_headless

_FIELDS = {field.name: field for field in INVENTORY_FIELDS}


def parse_set(text: str) -> tuple[InventoryField, int]:
    """``name=value``. Value is decimal or ``0x`` hex. Unknown names fail closed."""
    name, sep, raw = text.partition("=")
    name = name.strip()
    if not sep or not name or not raw.strip():
        raise SystemExit(f"--set must be field=value, got {text!r}")
    field = _FIELDS.get(name)
    if field is None:
        known = ", ".join(sorted(_FIELDS))
        raise SystemExit(f"unknown inventory field {name!r}; known: {known}")
    if _overlaps_follower(field):
        raise SystemExit(f"refusing to write $F3CC via {name}")
    try:
        value = int(raw.strip(), 0)
    except ValueError:
        raise SystemExit(f"--set {name} value is not an int: {raw!r}") from None
    if isinstance(value, bool) or value < 0 or value >= (1 << (8 * field.size)):
        raise SystemExit(
            f"--set {name}={value} does not fit {field.size} byte(s)"
        )
    return field, value


def _overlaps_follower(field: InventoryField) -> bool:
    follower = int(FOLLOWER)
    return any(int(field.offset) + i == follower for i in range(field.size))


def check_out(name: str) -> str:
    """Pokes belong on ``Whatif_*`` only. Reject a path or another prefix."""
    stem = name[:-6] if name.endswith(".state") else name
    if not stem.startswith("Whatif_") or stem != stem.strip() or "/" in stem or "\\" in stem:
        raise SystemExit(f"--out must be Whatif_<name>, got {name!r}")
    return stem


def assign_field(env: object, field: InventoryField, value: int) -> None:
    """Little-endian write of one inventory field. Does not step."""
    for i in range(field.size):
        assign_u8(env, int(field.offset) + i, (int(value) >> (8 * i)) & 0xFF)


def _refresh(env: object) -> None:
    data = getattr(env, "data", None)
    refresh = getattr(data, "update_ram", None)
    if refresh is not None:
        refresh()


def open_pin(name: str) -> object:
    """Boot headless and load the pin. Reset, then set_state. Do not step."""
    path = state_path(GAME_DIR, GAME, name)
    if not path.is_file():
        raise SystemExit(f"missing pin {name}")
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode=None)
    env.reset()
    env.em.set_state(read_state_bytes(path))
    _refresh(env)
    return env


def field_value(env: object, field: InventoryField) -> int:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return int(getattr(read_inventory(ram), field.name))


def write_whatif(
    env: object,
    source: str,
    sets: list[tuple[InventoryField, int]],
    out_name: str,
    *,
    kind: str = "whatif",
) -> None:
    """Apply ``sets``, save ``out_name``, and require the file to read back.

    ``kind`` is ``whatif`` or ``poke``. Either way ``$F3CC`` is not written
    and the source pin is not the save target.
    """
    if kind not in ("whatif", "poke"):
        raise SystemExit(f"provenance kind must be whatif or poke, got {kind!r}")
    out_name = check_out(out_name)
    source_path = state_path(GAME_DIR, GAME, source).resolve()
    out_path = state_path(GAME_DIR, GAME, out_name).resolve()
    if source_path == out_path:
        raise SystemExit(f"--out must differ from source pin {source!r}")
    for field, value in sets:
        assign_field(env, field, value)
        got = field_value(env, field)
        if got != value:
            raise SystemExit(
                f"{field.name} read {got} after write, wanted {value}"
            )
    path = save_state(env, GAME_DIR, GAME, out_name)
    write_pin_provenance(
        path,
        source_run=f"{kind}:{source}",
        frame=0,
        kind=kind,
    )
    env.em.set_state(read_state_bytes(path))  # type: ignore[attr-defined]
    _refresh(env)
    parts: list[str] = []
    for field, value in sets:
        got = field_value(env, field)
        if got != value:
            raise SystemExit(
                f"{out_name} {field.name} read back {got}, wanted {value}"
            )
        parts.append(f"{field.name}={got}")
    written = (read_pin_provenance(path) or {}).get("kind")
    if written != kind:
        raise SystemExit(f"{out_name} provenance kind {written!r}, wanted {kind}")
    print(f"{out_name} {' '.join(parts)} kind={written} source={source}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", help="source pin name, not modified")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="named inventory field (repeatable)",
    )
    parser.add_argument("--out", required=True, help="Whatif_<name> copy to write")
    parser.add_argument(
        "--kind",
        choices=("whatif", "poke"),
        default="whatif",
        help="provenance kind (default whatif). poke is a disclosed write",
    )
    args = parser.parse_args(argv)
    if not args.set:
        raise SystemExit("need at least one --set field=value")
    sets = [parse_set(item) for item in args.set]
    out_name = check_out(args.out)
    env = open_pin(args.state)
    try:
        write_whatif(env, args.state, sets, out_name, kind=args.kind)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
