"""Live gzip anchors: write/read, fingerprints, index match, AnchorRecorder."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# Prefer settled room_enter, then boot / manual / lockstep mid pins, then end.
# mid_lockstep ranks with manual (same usefulness for sub-hop boots).
_ANCHOR_KIND_RANK = {
    "room_enter": 0,
    "boot": 1,
    "manual": 2,
    "mid_lockstep": 2,
    "item_delta": 3,
    "end": 4,
}


def parse_room_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.lower().startswith("0x"):
            return int(s, 16)
        return int(s)
    except ValueError:
        return None


def parse_items_value(value: Any) -> int | None:
    """Parse collected-items bitmask from int or ``0x`` / decimal string.

    Shared by skill_bank ``parse_items`` and human_tape ``hop_items_int``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s, 0)
    except ValueError:
        return None


def as_xy(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [int(value[0]), int(value[1])]
    return None


def anchor_rows(
    anchors: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    """Accept full anchors index ``{\"anchors\": [...]}``, a bare list, or one fingerprint.

    A single mapping with ``kind`` or ``frame`` is treated as a one-row list
    (fingerprint dict without a nested ``anchors`` key).
    """
    if anchors is None:
        return []
    if isinstance(anchors, Mapping):
        rows = anchors.get("anchors")
        if rows is None:
            # Single fingerprint dict (has kind) — treat as one-row list
            if "kind" in anchors or "frame" in anchors:
                return [anchors]  # type: ignore[list-item]
            return []
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, Mapping)]
        return []
    return [r for r in anchors if isinstance(r, Mapping)]


# Back-compat private aliases (callers / internal)
_parse_room_id = parse_room_id
_as_xy = as_xy


def write_gzip_state(path: Path | str, state_bytes: bytes) -> Path:
    """Write emulator state as gzip (same format as ``RecordedTask.save``)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wb") as gz:
        gz.write(state_bytes)
    return out


def read_gzip_state(path: Path | str) -> bytes:
    with gzip.open(path, "rb") as gz:
        return gz.read()


def fingerprint(
    *,
    frame: int,
    room_id: int,
    x: int,
    y: int,
    pose: int = 0,
    items: int | None = None,
    beams: int | None = None,
    energy: int | None = None,
    kind: str = "pin",
    path: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact pin fingerprint stored in anchors index + task metadata."""
    row: dict[str, Any] = {
        "kind": kind,
        "frame": int(frame),
        "room": f"0x{int(room_id):04X}",
        "room_id": int(room_id),
        "xy": [int(x), int(y)],
        "pose": int(pose),
    }
    if items is not None:
        row["items"] = f"0x{int(items):04X}"
        row["grapple"] = bool(int(items) & 0x4000)
        row["gravity"] = bool(int(items) & 0x0020)
    if beams is not None:
        row["beams"] = f"0x{int(beams):04X}"
    if energy is not None:
        row["energy"] = int(energy)
    if path is not None:
        row["path"] = path
    if extra:
        row.update(dict(extra))
    return row


def fingerprint_from_trace_row(row: Mapping[str, Any], *, kind: str = "trace") -> dict[str, Any]:
    room = row.get("room")
    if room is None and row.get("room_hex"):
        room = int(str(row["room_hex"]), 16)
    items = row.get("items")
    if isinstance(items, str):
        items = int(items, 16) if items.startswith("0x") else int(items)
    beams = row.get("beams")
    if isinstance(beams, str):
        beams = int(beams, 16) if beams.startswith("0x") else int(beams)
    return fingerprint(
        frame=int(row.get("frame", 0)),
        room_id=int(room or 0),
        x=int(row.get("x", 0)),
        y=int(row.get("y", 0)),
        pose=int(row.get("pose", 0)),
        items=int(items) if items is not None else None,
        beams=int(beams) if beams is not None else None,
        energy=int(row["energy"]) if row.get("energy") is not None else None,
        kind=kind,
    )


def verify_end_against_trace(
    end_fp: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    *,
    xy_tol: int = 4,
) -> dict[str, Any]:
    """Compare a loaded end-state fingerprint to the last trace row."""
    if not trace:
        return {"ok": False, "reason": "empty_trace"}
    last = fingerprint_from_trace_row(trace[-1], kind="trace_end")
    room_ok = end_fp.get("room") == last.get("room")
    xy = end_fp.get("xy") or [0, 0]
    txy = last.get("xy") or [0, 0]
    xy_ok = abs(int(xy[0]) - int(txy[0])) <= xy_tol and abs(int(xy[1]) - int(txy[1])) <= xy_tol
    items_ok = True
    if "items" in end_fp and "items" in last:
        items_ok = end_fp["items"] == last["items"]
    ok = bool(room_ok and xy_ok and items_ok)
    return {
        "ok": ok,
        "room_ok": room_ok,
        "xy_ok": xy_ok,
        "items_ok": items_ok,
        "end": dict(end_fp),
        "trace_end": last,
    }


def load_anchors_index(
    task_path: Path | str,
    *,
    anchors_index_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Load ``tasks/<name>_anchors.json`` next to the task (or explicit path)."""
    if anchors_index_path is not None:
        p = Path(anchors_index_path)
    else:
        path = Path(task_path)
        candidates = [
            path.with_name(path.stem + "_anchors.json"),
            # Archived segment tapes live as sN/tape.json next to sN/anchors.json.
            path.with_name("anchors.json"),
        ]
        p = next((c for c in candidates if c.is_file()), candidates[0])
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return {"anchors": list(data) if isinstance(data, list) else [], "path": str(p)}
    data = dict(data)
    data.setdefault("path", str(p))
    return data


def resolve_anchor_path(
    anchor: Mapping[str, Any],
    *,
    anchors_index: Mapping[str, Any] | None = None,
    task_path: Path | str | None = None,
) -> Path | None:
    """Resolve an anchor row's ``path`` (absolute or basename under anchors_dir)."""
    raw = anchor.get("path")
    if not raw:
        return None
    p = Path(str(raw))
    if p.is_file():
        return p.resolve()

    candidates: list[Path] = []
    name = p.name
    anchors_dir = None
    if anchors_index:
        ad = anchors_index.get("anchors_dir")
        if ad:
            anchors_dir = Path(str(ad))
            candidates.append(anchors_dir / name)
    if task_path is not None:
        tp = Path(task_path)
        candidates.append(tp.with_name(tp.stem + "_anchors") / name)
        if anchors_dir is None:
            candidates.append(tp.parent / (tp.stem + "_anchors") / name)
        candidates.append(tp.parent / name)

    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def match_anchor(
    anchors_index: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    frame_or_start_index: int,
    room_id: int | None = None,
    *,
    task_path: Path | str | None = None,
    prefer_kinds: Sequence[str] | None = None,
    settle_window: int = 256,
) -> dict[str, Any] | None:
    """Pick the best live anchor for a hop start.

    Preference order (lower sort key wins):
    1. Same ``room_id`` (when given)
    2. ``room_enter`` with frame in ``[target, target+settle_window]``
       (door-settle dumps land slightly *after* hop start_index)
    3. room_enter/boot/manual/mid_lockstep with frame **at or before** target (latest)
    4. Any remaining same-room / other-room by kind + |frame-target|

    Returns the anchor row with resolved ``path`` (str) when a file exists,
    else ``None``.
    """
    if anchors_index is None:
        return None
    if isinstance(anchors_index, Mapping):
        rows = list(anchors_index.get("anchors") or [])
        index_map: Mapping[str, Any] | None = anchors_index
    else:
        rows = list(anchors_index)
        index_map = None
    if not rows:
        return None

    target = int(frame_or_start_index)
    want_room = int(room_id) if room_id is not None else None
    window = max(0, int(settle_window))
    kind_filter = set(prefer_kinds) if prefer_kinds else None

    def _kind_rank(kind: str) -> int:
        return _ANCHOR_KIND_RANK.get(kind, 50)

    def _resolved(row: Mapping[str, Any]) -> Path | None:
        return resolve_anchor_path(row, anchors_index=index_map, task_path=task_path)

    scored: list[tuple[tuple[Any, ...], Mapping[str, Any], Path]] = []
    for row in rows:
        path = _resolved(row)
        if path is None:
            continue
        kind = str(row.get("kind") or "pin")
        if kind_filter is not None and kind not in kind_filter:
            continue
        frame = int(row.get("frame") or 0)
        rid = row.get("room_id")
        if rid is None:
            rid = parse_room_id(row.get("room"))
        rid_i = int(rid) if rid is not None else None
        room_match = want_room is None or rid_i == want_room
        at_or_before = frame <= target
        enter_in_window = (
            kind == "room_enter"
            and target <= frame <= target + window
        )
        # tier 0: room_enter just after hop start (settle)
        # tier 1: any preferred pin at or before target
        # tier 2: everything else
        if enter_in_window:
            tier = 0
            key = (
                0 if room_match else 1,
                tier,
                frame - target,
                _kind_rank(kind),
            )
        elif at_or_before:
            tier = 1
            # Prefer **latest** pin at-or-before target so lockstep mid /
            # manual F6 pins beat a far-earlier room_enter for sub-hops.
            key = (
                0 if room_match else 1,
                tier,
                -frame,
                _kind_rank(kind),
            )
        else:
            tier = 2
            key = (
                0 if room_match else 1,
                tier,
                abs(frame - target),
                _kind_rank(kind),
            )
        scored.append((key, row, path))

    if not scored:
        return None
    scored.sort(key=lambda t: t[0])
    _key, best, path = scored[0]
    out = dict(best)
    out["path"] = str(path)
    out["resolved_path"] = str(path)
    return out


@dataclass
class AnchorRecorder:
    """Live dump of gzip states during a guided human session."""

    task_name: str
    anchors_dir: Path
    enabled: bool = True
    settle_ordinary_only: bool = True
    _last_room: int | None = field(default=None, init=False)
    _last_items: int | None = field(default=None, init=False)
    _boot_dumped: bool = field(default=False, init=False)
    anchors: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.enabled:
            self.anchors_dir.mkdir(parents=True, exist_ok=True)

    def _maybe_settle_ok(self, st: Any) -> bool:
        if not self.settle_ordinary_only:
            return True
        if int(getattr(st, "door_transition", 0) or 0) != 0:
            return False
        phase = getattr(st, "phase", None)
        if phase is None:
            return True
        label = str(
            getattr(phase, "name", None)
            or getattr(phase, "value", None)
            or phase
        )
        return "ordinary" in label.lower()

    def dump(
        self,
        *,
        env: Any,
        st: Any,
        frame: int,
        kind: str,
        label: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        room = int(st.room_id)
        items = int(st.collected_items)
        beams = int(st.collected_beams)
        slug = label or kind
        fname = f"f{int(frame):06d}_{slug}_0x{room:04X}.state"
        path = self.anchors_dir / fname
        blob = env.em.get_state()
        write_gzip_state(path, blob)
        fp = fingerprint(
            frame=frame,
            room_id=room,
            x=int(st.samus_x),
            y=int(st.samus_y),
            pose=int(st.pose),
            items=items,
            beams=beams,
            energy=int(st.health),
            kind=kind,
            path=str(path),
        )
        self.anchors.append(fp)
        return fp

    def on_frame(self, *, env: Any, st: Any, frame: int) -> list[dict[str, Any]]:
        """Call after each assist-applied step. Returns new anchors (0–2).

        Room-enter fires on the **first settled ordinary frame** in a new room.
        Do **not** update ``_last_room`` during door/load phases — that would
        swallow the enter (room already matches when settle resumes).
        """
        if not self.enabled:
            return []
        if not self._maybe_settle_ok(st):
            return []

        dumped: list[dict[str, Any]] = []
        room = int(st.room_id)
        items = int(st.collected_items)

        if not self._boot_dumped:
            fp = self.dump(env=env, st=st, frame=frame, kind="boot", label="boot")
            if fp:
                dumped.append(fp)
            self._boot_dumped = True
            self._last_room = room
            self._last_items = items
            return dumped

        if self._last_room is not None and room != self._last_room:
            fp = self.dump(
                env=env,
                st=st,
                frame=frame,
                kind="room_enter",
                label=f"enter_0x{room:04X}",
            )
            if fp:
                dumped.append(fp)
        self._last_room = room

        if self._last_items is not None and items != self._last_items:
            fp = self.dump(
                env=env,
                st=st,
                frame=frame,
                kind="item_delta",
                label=f"items_0x{items:04X}",
            )
            if fp:
                dumped.append(fp)
        self._last_items = items
        return dumped

    def manual_pin(self, *, env: Any, st: Any, frame: int) -> dict[str, Any] | None:
        return self.dump(env=env, st=st, frame=frame, kind="manual", label="manual")

    def write_index(self, path: Path | str, *, extra: Mapping[str, Any] | None = None) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "task": self.task_name,
            "anchors_dir": str(self.anchors_dir),
            "count": len(self.anchors),
            "anchors": self.anchors,
        }
        if extra:
            payload.update(dict(extra))
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return out
