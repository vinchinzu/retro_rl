"""Link the verified shared Super Metroid ROM into SMRando-Snes.

```bash
uv run python -m sm_rando.scripts.setup_rom
```
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _SNES_IMPORT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sm_rando.paths import (  # noqa: E402
    INTEGRATION_DIR,
    SHARED_SM_ROM,
    SM_SHA1,
)


def setup_rom(*, rom: Path | None = None) -> Path:
    """Symlink ``rom.sfc`` and write ``rom.sha`` (SHA1). Return integration link."""
    source = Path(rom) if rom is not None else SHARED_SM_ROM
    if not source.is_file():
        raise FileNotFoundError(f"Missing Super Metroid ROM: {source}")

    digest = hashlib.sha1(source.read_bytes()).hexdigest()
    if digest != SM_SHA1:
        raise ValueError(
            f"ROM SHA1 mismatch for {source}: got {digest}, expected {SM_SHA1}"
        )

    INTEGRATION_DIR.mkdir(parents=True, exist_ok=True)
    target = INTEGRATION_DIR / "rom.sfc"
    rel = Path(os.path.relpath(source.resolve(), target.parent))
    if target.exists() or target.is_symlink():
        try:
            same = target.resolve() == source.resolve()
        except OSError:
            same = False
        if not same:
            target.unlink()
            target.symlink_to(rel)
    else:
        target.symlink_to(rel)

    (INTEGRATION_DIR / "rom.sha").write_text(f"{digest}\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    del argv  # unused; keep CLI shape stable
    try:
        target = setup_rom()
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"ROM ready: {target} -> {target.resolve()}")
    print(f"rom.sha = {SM_SHA1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
