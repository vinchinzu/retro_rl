"""Human guided-record anchors + hop extraction + open-loop replay.

Package split of the former flat modules:

- ``anchors`` — gzip pins, fingerprints, ``AnchorRecorder``, index match
- ``hops`` — room hops, skill groups, ``HopSlice`` / ``resolve_hop_slice``, extract
- ``replay`` — shared step loop, ``replay_hop``, green check, ``run_hop_replay``
- ``bodies`` — per-hop SNES-12 body export (hill-climb / bank seeds)
- ``compose`` — multi-hop pin→body chain (seam-safe open-loop)
- ``segment_archive`` — immutable segment tapes when reusing ``--name``
- ``midpoints`` — offline mid candidates + lockstep materialize
- ``trim`` — offline idle/retry hop trim

Public re-exports keep ``from super_metroid.human_tape import …`` working.
"""

from __future__ import annotations

from super_metroid.human_tape.anchors import (
    AnchorRecorder,
    _ANCHOR_KIND_RANK,
    anchor_rows,
    as_xy,
    fingerprint,
    fingerprint_from_trace_row,
    load_anchors_index,
    match_anchor,
    parse_items_value,
    parse_room_id,
    read_gzip_state,
    resolve_anchor_path,
    verify_end_against_trace,
    write_gzip_state,
)
from super_metroid.human_tape.bodies import (
    export_hop_bodies,
    export_hop_body,
    hop_bodies_dir,
)
from super_metroid.human_tape.compose import (
    ComposeHopResult,
    ComposeReport,
    compose_hops,
    compose_route_plan,
)
from super_metroid.human_tape.hops import (
    HopSlice,
    build_room_hops,
    default_skill_groups,
    extract_tape,
    hop_items_int,
    load_room_hops,
    load_room_names,
    load_task_json,
    resolve_hop_slice,
    resolve_hop_slice_typed,
    settle_room_hops,
)
from super_metroid.human_tape.midpoints import (
    lockstep_scan,
    materialize_lockstep_mid,
    propose_trace_midpoints,
)
from super_metroid.human_tape.replay import (
    check_hop_green,
    frame_action,
    iter_hop_steps,
    replay_hop,
    resolve_assist,
    run_hop_replay,
)
from super_metroid.human_tape.segment_archive import (
    archive_existing_take,
    list_archived_tapes,
    segments_dir_for,
)
from super_metroid.human_tape.product_chain import (
    build_product_chain_board,
    format_board_summary,
    write_product_chain_board,
)
from super_metroid.human_tape.pb_board import (
    PbBoard,
    format_pb_board_table,
    materialize_pb_board,
    pb_board_path,
)
from super_metroid.human_tape.rta_clock import (
    find_ceres_zero_frame,
    fmt_time as rta_fmt_time,
    load_archive_segments,
    product_chain_segments,
    resolve_rta_clock,
)
from super_metroid.human_tape.stitch import (
    format_pb_table,
    materialize_stitch,
    rezero_report_to_ceres,
    stitch_task_anchors,
)

__all__ = [
    "AnchorRecorder",
    "ComposeHopResult",
    "ComposeReport",
    "HopSlice",
    "_ANCHOR_KIND_RANK",
    "anchor_rows",
    "archive_existing_take",
    "as_xy",
    "build_product_chain_board",
    "build_room_hops",
    "check_hop_green",
    "compose_hops",
    "compose_route_plan",
    "default_skill_groups",
    "export_hop_bodies",
    "export_hop_body",
    "extract_tape",
    "format_board_summary",
    "find_ceres_zero_frame",
    "fingerprint",
    "fingerprint_from_trace_row",
    "PbBoard",
    "format_pb_board_table",
    "format_pb_table",
    "frame_action",
    "hop_bodies_dir",
    "hop_items_int",
    "iter_hop_steps",
    "list_archived_tapes",
    "load_anchors_index",
    "load_archive_segments",
    "load_room_hops",
    "load_room_names",
    "load_task_json",
    "lockstep_scan",
    "match_anchor",
    "materialize_lockstep_mid",
    "materialize_pb_board",
    "materialize_stitch",
    "parse_items_value",
    "parse_room_id",
    "pb_board_path",
    "product_chain_segments",
    "propose_trace_midpoints",
    "read_gzip_state",
    "replay_hop",
    "resolve_anchor_path",
    "resolve_assist",
    "resolve_hop_slice",
    "resolve_hop_slice_typed",
    "resolve_rta_clock",
    "rezero_report_to_ceres",
    "rta_fmt_time",
    "run_hop_replay",
    "segments_dir_for",
    "settle_room_hops",
    "stitch_task_anchors",
    "verify_end_against_trace",
    "write_gzip_state",
]
