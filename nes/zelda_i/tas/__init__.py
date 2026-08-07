"""TAS tooling for The Legend of Zelda (NES).

Button-press movies from TASVideos (FCEUX ``.fm2``). Prefer import + route
adapt over blind room search. See ``docs/TAS_ADAPT.md``.

```bash
uv run python -m zelda_i.tas.fetch_refs
uv run python -m zelda_i.tas.import_fm2 --summary-only
```
"""

from zelda_i.tas.fm2 import Fm2Movie, fm2_to_nes9_frames, parse_fm2

__all__ = [
    "Fm2Movie",
    "fm2_to_nes9_frames",
    "parse_fm2",
]
