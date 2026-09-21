"""ALttP TAS movie import + snes12_rle slice export.

Parse lsnes ``.lsmv``, BizHawk ``.bk2``, and Snes9x ``.smv`` movies into
SNES-12 env frames (same parsers as Super Metroid) and compress to
``snes12_rle`` seeds. Replay is **not** claimed — slices are button logs.

Ref movies live under ``tas/ref/`` (gitignored; re-fetch).

```bash
uv run python -m alttp.tas.fetch_refs
uv run python -m alttp.tas.export_slices --list
uv run pytest alttp/tests/test_tas.py -q
```
"""

from __future__ import annotations

from alttp.tas.bk2 import parse_bk2
from alttp.tas.lsmv import parse_lsmv
from alttp.tas.rle import (
    compress_snes12_rle,
    expand_snes12_rle,
    frames_to_snes12_rle_payload,
    load_snes12_rle_seed,
)
from alttp.tas.slice import SLICE_CATALOG, export_slice, load_movie_frames

__all__ = [
    "SLICE_CATALOG",
    "compress_snes12_rle",
    "expand_snes12_rle",
    "export_slice",
    "frames_to_snes12_rle_payload",
    "load_movie_frames",
    "load_snes12_rle_seed",
    "parse_bk2",
    "parse_lsmv",
]
