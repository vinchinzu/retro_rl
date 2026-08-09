"""Super Metroid TAS movie import + harness replay/annotate.

Parse lsnes ``.lsmv`` and BizHawk ``.bk2`` movies into SNES-12 env frames,
compress to ``snes12_rle`` seeds, export named slices, and replay under
stable-retro with WRAM annotation (pose / x,y / velocity / rooms / items).

Ref movies live under ``tas/ref/`` (HappyLee-style vendored inputs).

```bash
uv run python -m super_metroid.tas.fetch_refs
uv run python -m super_metroid.tas.export_slices --finish
uv run python -m super_metroid.tas.replay --slice sniq_any_menu --annotate
```
"""

from __future__ import annotations

from super_metroid.tas.annotate import Annotator, TraceEvent
from super_metroid.tas.bk2 import parse_bk2
from super_metroid.tas.lsmv import parse_lsmv
from super_metroid.tas.rle import (
    compress_snes12_rle,
    expand_snes12_rle,
    frames_to_snes12_rle_payload,
    load_snes12_rle_seed,
)
from super_metroid.tas.slice import SLICE_CATALOG, export_slice, load_movie_frames
from super_metroid.tas.stages import (
    STAGE_CATALOG,
    RoomStageSpec,
    export_room_body_spec,
    get_stage,
    is_room_settled,
)
from super_metroid.tas.trace import (
    MovieTrace,
    resolve_frames,
    trace_frames,
    write_trace_artifacts,
)

# materialize_room_body: import from super_metroid.tas.materialize (CLI module;
# not re-exported here so ``python -m super_metroid.tas.materialize`` stays clean).

__all__ = [
    "Annotator",
    "MovieTrace",
    "RoomStageSpec",
    "SLICE_CATALOG",
    "STAGE_CATALOG",
    "TraceEvent",
    "compress_snes12_rle",
    "expand_snes12_rle",
    "export_room_body_spec",
    "export_slice",
    "frames_to_snes12_rle_payload",
    "get_stage",
    "is_room_settled",
    "load_movie_frames",
    "load_snes12_rle_seed",
    "parse_bk2",
    "parse_lsmv",
    "resolve_frames",
    "trace_frames",
    "write_trace_artifacts",
]
