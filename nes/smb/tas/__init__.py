"""TAS tooling for Super Mario Bros.

- ``fm2``: import FCEUX movies (HappyLee warps, community RTA-rules)
- ``pipeline`` / ``search`` / ``windows``: 1-1 analyze + hill-climb polish

Prefer **import + adapt** public movies over blind hill-climb. See
``docs/TAS_ADAPT.md``.

Typical CLI::

    uv run python -m smb.scripts.import_fm2 --summary-only
    uv run python -m smb.scripts.tas_1_1 analyze
    uv run python -m smb.scripts.tas_1_1 optimize --iters 400
    uv run python -m smb.scripts.tas_1_1 verify
"""

from smb.tas.fm2 import Fm2Movie, fm2_to_nes9_frames, parse_fm2
from smb.tas.pipeline import OptimizeReport, ensure_completing_seed, optimize_1_1
from smb.tas.search import edge_shift_search, polish_systematic, systematic_delete_sweep
from smb.tas.trace import SeedTrace, TraceEvent, trace_seed
from smb.tas.windows import TasWindow, discover_windows

__all__ = [
    "Fm2Movie",
    "OptimizeReport",
    "SeedTrace",
    "TasWindow",
    "TraceEvent",
    "discover_windows",
    "edge_shift_search",
    "ensure_completing_seed",
    "fm2_to_nes9_frames",
    "optimize_1_1",
    "parse_fm2",
    "polish_systematic",
    "systematic_delete_sweep",
    "trace_seed",
]
