"""Log every (module, submodule) change on one verified-tip run.

Invokes ``run_to_verified_tip.py`` once. The report and screenshot go to
``/tmp`` so ``recordings/verified_tip_run.json`` is not overwritten.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/module_census.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from alttp.ram import INDOORS, MODULE, ROOM_ID, SCREEN_ID, SUBMODULE, read_u16, read_u8

# snesrev kMainRouting / kDungeonSubmodules / kOverworldSubmodules / kMessagingSubmodules.
# A pair is named only when that table gives one of the census words.
_PAIR_NAME: dict[tuple[int, int], str] = {
    (0x07, 0x00): "play",
    (0x09, 0x00): "play",
    (0x0B, 0x00): "play",
    (0x07, 0x01): "room scroll",
    (0x07, 0x02): "room scroll",
    (0x07, 0x05): "shutters",
    (0x07, 0x0F): "landing wipe",
    (0x09, 0x01): "OW load",
    (0x09, 0x02): "OW load",
    (0x09, 0x03): "OW load",
    (0x09, 0x04): "OW load",
    (0x09, 0x05): "room scroll",
    (0x09, 0x06): "room scroll",
    (0x09, 0x07): "room scroll",
    (0x09, 0x08): "room scroll",
    (0x09, 0x0A): "exit walk",
    (0x07, 0x06): "stairs",
    (0x07, 0x07): "stairs",
    (0x07, 0x08): "stairs",
    (0x07, 0x0E): "stairs",
    (0x07, 0x10): "stairs",
    (0x07, 0x11): "stairs",
    (0x07, 0x12): "stairs",
    (0x07, 0x13): "stairs",
    (0x0E, 0x00): "dialog",
    (0x0E, 0x02): "dialog",
    (0x0E, 0x01): "menu",
    (0x0E, 0x03): "menu",
    (0x0E, 0x07): "menu",
    (0x0E, 0x0B): "menu",
    (0x06, 0x00): "dungeon load",
    (0x08, 0x00): "OW load",
    (0x08, 0x01): "OW load",
    (0x08, 0x02): "OW load",
    (0x0A, 0x00): "OW load",
    (0x0A, 0x01): "OW load",
    (0x0A, 0x02): "OW load",
}
_MODULE_NAME: dict[int, str] = {
    0x00: "intro",
    0x01: "file select",
    0x02: "copy file",
    0x03: "kill file",
    0x04: "name file",
    0x05: "load file",
    0x06: "dungeon load",
    0x07: "play",
    0x08: "OW load",
    0x09: "play",
    0x0A: "OW load",
    0x0B: "play",
    0x0E: "dialog",
    0x0F: "spotlight close",
    0x10: "spotlight open",
    0x11: "dungeon load",
}


def pair_name(module: int, submodule: int) -> str:
    """Census word for one pair. Unknown pairs stay unnamed."""
    named = _PAIR_NAME.get((module, submodule))
    if named is not None:
        return named
    return _MODULE_NAME.get(module, "unnamed")


@dataclass
class Change:
    frame: int
    module: int
    submodule: int
    room: int
    screen: int
    indoors: int
    name: str


class _Census:
    def __init__(self) -> None:
        self.frame = 0
        self.changes: list[Change] = []
        self._last: tuple[int, int] | None = None
        self._reset_seen = False

    def attach(self, env: object) -> object:
        orig_reset = env.reset  # type: ignore[attr-defined]
        orig_step = env.step  # type: ignore[attr-defined]

        def reset(*args: object, **kwargs: object) -> object:
            out = orig_reset(*args, **kwargs)
            if not self._reset_seen:
                self.frame = 0
                self._reset_seen = True
            self._note(env)
            return out

        def step(action: object) -> object:
            out = orig_step(action)
            self.frame += 1
            self._note(env)
            if self.frame % 2000 == 0:
                print(f"... frame {self.frame}", flush=True)
            return out

        env.reset = reset  # type: ignore[attr-defined]
        env.step = step  # type: ignore[attr-defined]
        return env

    def _note(self, env: object) -> None:
        ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
        module = read_u8(ram, MODULE)
        submodule = read_u8(ram, SUBMODULE)
        pair = (module, submodule)
        if pair == self._last:
            return
        self._last = pair
        row = Change(
            frame=self.frame,
            module=module,
            submodule=submodule,
            room=read_u16(ram, ROOM_ID),
            screen=read_u8(ram, SCREEN_ID),
            indoors=read_u8(ram, INDOORS),
            name=pair_name(module, submodule),
        )
        self.changes.append(row)
        print(
            f"{row.frame:5d} mod={row.module:02X} sub={row.submodule:02X} "
            f"room={row.room:04X} screen={row.screen:02X} in={row.indoors} {row.name}",
            flush=True,
        )


def _first_seen(changes: list[Change]) -> list[Change]:
    seen: set[tuple[int, int]] = set()
    rows: list[Change] = []
    for row in changes:
        key = (row.module, row.submodule)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def main() -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")

    import alttp.scripts.run_to_verified_tip as tip

    census = _Census()
    real_boot = tip.build_boot_env

    def boot(*args: object, **kwargs: object) -> object:
        return census.attach(real_boot(*args, **kwargs))

    tip.build_boot_env = boot  # type: ignore[assignment]
    code = tip.main(
        [
            "--report",
            "/tmp/ap008_verified_tip.json",
            "--screenshot",
            "/tmp/ap008_verified_tip.png",
        ]
    )
    print("--- first seen ---")
    for row in _first_seen(census.changes):
        print(
            f"first frame={row.frame} mod={row.module:02X} sub={row.submodule:02X} "
            f"room={row.room:04X} screen={row.screen:02X} {row.name}"
        )
    print(f"changes={len(census.changes)} steps={census.frame} exit={code}")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
