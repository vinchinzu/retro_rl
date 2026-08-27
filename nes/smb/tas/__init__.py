"""TAS tooling for Super Mario Bros.

- ``fm2``: import FCEUX movies (HappyLee warps, warpless #3728M)
- ``bk2``: NesHawk FM2→BK2 conversion (same mapping as ``*.fm2.bk2``)
- ``stages``: StageSpec table + control/goal predicates + HL indices
- ``slice``: probe / export / search from StageSpec
- ``chain``: reach_* / verify_* navigation (single path for all tracks)
- ``replay``: NES-9 helpers (``to_action9``, ``idle_until``; preserves L+R)
- ``pipeline`` / ``search`` / ``windows``: residual 1-1 hill-climb polish

Prefer **import + adapt** public movies over blind hill-climb. See
``docs/TAS_ADAPT.md``.

Typical CLI::

    uv run python -m smb.scripts.import_fm2 --summary-only
    uv run python -m smb.scripts.import_fm2 --verify-1-2-slice
    uv run python -m smb.tas.fetch_refs
    uv run python -m smb.scripts.convert_fm2
    uv run python -m smb.scripts.tas_1_1 analyze
"""

from smb.tas.fm2 import Fm2Movie, fm2_to_nes9_frames, parse_fm2, parse_movie
from smb.tas.pipeline import OptimizeReport, ensure_completing_seed, optimize_1_1
from smb.tas.replay import to_action9
from smb.tas.search import edge_shift_search, polish_systematic, systematic_delete_sweep
from smb.tas.slice import SliceProbe, export_stage_slice, probe_from_control
from smb.tas.stages import STAGES, StageSpec, get_stage
from smb.tas.trace import SeedTrace, TraceEvent, trace_seed
from smb.tas.windows import TasWindow, discover_windows

__all__ = [
    "Fm2Movie",
    "OptimizeReport",
    "SeedTrace",
    "SliceProbe",
    "StageSpec",
    "STAGES",
    "TasWindow",
    "TraceEvent",
    "discover_windows",
    "edge_shift_search",
    "ensure_completing_seed",
    "export_stage_slice",
    "fm2_to_nes9_frames",
    "get_stage",
    "optimize_1_1",
    "parse_fm2",
    "parse_movie",
    "polish_systematic",
    "probe_from_control",
    "systematic_delete_sweep",
    "to_action9",
    "trace_seed",
]
