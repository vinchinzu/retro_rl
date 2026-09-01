"""Ceres Elevator (0xDF45) pixel occupancy + wall-jump faces.

Source of truth is the editor ROM export (clip + BTS), checked against live
stable-retro pins — not sm-json-data (that file is two nodes and no collision).

    PYTHONPATH=snes uv run python -m super_metroid.scratch.ceres_elev_wj.allmap

Room is 16×48 tiles (256×768 px). The only slope in the shaft is shape 1
(half-solid side) at $94:8B2B. Editor ``drawSlopeOverlay`` indexes
``col = lx if xflip else (15-lx)``; live pins match that, not "ROM col 0 =
left, no flip":

* BTS 0x01 col 13: right 8px solid, inner face **x=216**, spin rest x=211.255
* BTS 0x41 col  2: left  8px solid, inner face **x=40**,  spin rest x=45
* Standing yr=21 → stand_y = floor_top - 21 (475/363/267/171/75/571)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from super_metroid.generalist.solid import editor_rooms_dir

HERE = Path(__file__).resolve().parent
ROOM_HEX = "DF45"
TILE = 16
W, H = 16 * TILE, 48 * TILE
SPIN_XR, SPIN_YR = 5, 12
STAND_YR = 21
WJ_REACH = 8
CLIP_AIR, CLIP_SLOPE, CLIP_SOLID, CLIP_DOOR = 0, 1, 8, 9

OUT_JSON = HERE / "allmap.json"
OUT_TXT = HERE / "allmap.txt"


_OCC: list[list[bool]] | None = None


def occupancy() -> list[list[bool]]:
    """Pixel solid map, cached. Out-of-room is treated as solid by ``is_solid``."""
    global _OCC
    if _OCC is None:
        _OCC = raster(load_editor())[0]
    return _OCC


def is_solid(x: int, y: int) -> bool:
    if not (0 <= x < W and 0 <= y < H):
        return True
    return occupancy()[y][x]


def editor_room_path() -> Path:
    rooms = editor_rooms_dir()
    if rooms is None:
        raise SystemExit("editor sm_nav/rooms not found; set SUPER_METROID_EDITOR_NAV")
    path = Path(rooms) / f"room_{ROOM_HEX}.json"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return path


def load_editor() -> dict:
    path = editor_room_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("roomIdHex") != f"0x{ROOM_HEX}":
        raise SystemExit(f"unexpected room {payload.get('roomIdHex')}")
    return payload


def is_solid_px(clip: int, bts: int, lx: int, ly: int) -> bool:
    """Pixel inside one tile. ``ly`` unused for shape-1 (full-height half)."""
    del ly
    if clip in (CLIP_SOLID, CLIP_DOOR):
        return True
    if clip != CLIP_SLOPE:
        return False
    shape = bts & 0x1F
    if shape != 1:
        raise SystemExit(f"unexpected slope shape {shape:#x} in Ceres elev")
    xflip = bool(bts & 0x40)
    return (lx < 8) if xflip else (lx >= 8)


def raster(payload: dict) -> tuple[list[list[bool]], list[list[int]], list[list[int]]]:
    coll, bts = payload["collision"], payload["bts"]
    solid = [[False] * W for _ in range(H)]
    for by, row in enumerate(coll):
        for bx, clip in enumerate(row):
            tb = int(bts[by][bx])
            for ly in range(TILE):
                py = by * TILE + ly
                for lx in range(TILE):
                    solid[py][bx * TILE + lx] = is_solid_px(int(clip), tb, lx, ly)
    return solid, coll, bts


def faces_on_row(row: list[bool]) -> tuple[list[int], list[int]]:
    """left_faces (air after solid, kick RIGHT) / right_faces (solid after air, kick LEFT)."""
    left, right = [], []
    for x in range(1, W):
        a, b = row[x - 1], row[x]
        if a and not b:
            left.append(x)
        elif (not a) and b:
            right.append(x)
    return left, right


def merge_runs(points: list[tuple[int, int]]) -> list[dict]:
    """``(y, face_x)`` → runs ``{face_x, y0, y1}`` inclusive."""
    by_x: dict[int, list[int]] = defaultdict(list)
    for y, fx in points:
        by_x[fx].append(y)
    runs = []
    for fx, ys in sorted(by_x.items()):
        ys = sorted(ys)
        y0 = prev = ys[0]
        for y in ys[1:]:
            if y == prev + 1:
                prev = y
                continue
            runs.append({"face_x": fx, "y0": y0, "y1": prev})
            y0 = prev = y
        runs.append({"face_x": fx, "y0": y0, "y1": prev})
    runs.sort(key=lambda r: (r["y0"], r["face_x"]))
    return runs


def ledges(solid: list[list[bool]]) -> list[dict]:
    """Top of a solid run with air above, grouped into x-spans per floor_top."""
    cells: dict[int, list[int]] = defaultdict(list)
    for x in range(W):
        for y in range(1, H):
            if solid[y][x] and not solid[y - 1][x]:
                cells[y].append(x)
    out = []
    for floor, xs in sorted(cells.items()):
        xs = sorted(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
                j += 1
            x0, x1 = xs[i], xs[j]
            out.append({
                "floor_top": floor,
                "stand_y": floor - STAND_YR,
                "x0": x0,
                "x1": x1,
                "row": floor // TILE,
            })
            i = j + 1
    return out


def chimneys(solid: list[list[bool]]) -> list[dict]:
    """Per-y air gap between the innermost left-wall and right-wall faces."""
    raw: list[tuple[int, int, int, int]] = []
    for y, row in enumerate(solid):
        left, right = faces_on_row(row)
        if not left or not right:
            continue
        lf, rf = max(left), min(right)
        if lf < rf:
            raw.append((y, lf, rf, rf - lf))
    if not raw:
        return []
    runs = []
    y0, lf, rf, gap = raw[0]
    prev = y0
    for y, a, b, g in raw[1:]:
        if y == prev + 1 and (a, b) == (lf, rf):
            prev = y
            continue
        runs.append({"y0": y0, "y1": prev, "left_face": lf, "right_face": rf, "air": gap})
        y0, lf, rf, gap, prev = y, a, b, g, y
    runs.append({"y0": y0, "y1": prev, "left_face": lf, "right_face": rf, "air": gap})
    return runs


def wj_ok(solid: list[list[bool]], x: int, y: int, away: str,
          xr: int = SPIN_XR, yr: int = SPIN_YR, reach: int = WJ_REACH) -> bool:
    """Occupancy: a solid pixel in the 8px horizontal box over ``[y-yr, y+yr]``.

    Not a latch oracle — ``wjmap.py`` still needs the 2-frame A release.
    """
    if away == "LEFT":
        xs = range(x + xr + 1, x + xr + reach + 1)
    else:
        xs = range(x - xr - reach, x - xr)
    for py in range(max(0, y - yr), min(H, y + yr + 1)):
        for px in xs:
            if 0 <= px < W and solid[py][px]:
                return True
    return False


def wj_band(face_x: int, side: str) -> list[int]:
    """Inclusive spin-x band that can latch this face (xr=5, reach=8)."""
    if side == "right":
        return [face_x - SPIN_XR - WJ_REACH, face_x - SPIN_XR]
    return [face_x + SPIN_XR, face_x + SPIN_XR + WJ_REACH]


def ascii_blocks(coll: list[list[int]]) -> list[str]:
    glyph = {0: ".", 1: "/", 8: "#", 9: "D"}
    lines = []
    for by, row in enumerate(coll):
        body = "".join(glyph.get(int(c), "?") for c in row)
        y0, y1 = by * TILE, by * TILE + 15
        stand = y0 - STAND_YR
        lines.append(f"{by:2d} y={y0:3d}-{y1:3d} stand={stand:3d} {body}")
    return lines


def self_check(solid: list[list[bool]]) -> list[str]:
    """Live pins from this sitting. Failures stay in the report."""
    checks = [
        ("right face 216 solid", solid[530][216] and not solid[530][215]),
        ("left face 40 air", (not solid[530][40]) and solid[530][39]),
        ("363 ledge left face 160", solid[390][160] and not solid[390][159]),
        ("475 floor 496 at x=156", solid[496][156] and not solid[495][156]),
        ("363 floor 384 at x=189", solid[384][189] and not solid[383][189]),
        ("267 floor 288 at x=107", solid[288][107] and not solid[287][107]),
        ("171 floor 192 at x=66", solid[192][66] and not solid[191][66]),
        ("75/ship floor 96 at x=113", solid[96][113] and not solid[95][113]),
        ("571 floor 592 at x=80", solid[592][80] and not solid[591][80]),
        ("row12 gap at x=120 y=192", not solid[192][120]),
        ("door tile 240,630 solid", solid[630][240]),
        ("open shaft x=128 y=530", not solid[530][128]),
    ]
    bad = [name for name, ok in checks if not ok]
    return bad


def compare_wjmap(solid: list[list[bool]]) -> dict:
    path = HERE / "wjmap.json"
    if not path.is_file():
        return {"skipped": True}
    grid = json.loads(path.read_text())
    xs = list(range(32, 229, 4))
    tp = fp = fn = 0
    misses = []
    for y_s, row in grid.items():
        y = int(y_s)
        for away in ("LEFT", "RIGHT"):
            got = {x for x in xs if wj_ok(solid, x, y, away)}
            exp = set(row.get(away, []))
            tp += len(got & exp)
            extra = sorted(got - exp)
            missing = sorted(exp - got)
            fp += len(extra)
            fn += len(missing)
            if extra or missing:
                misses.append({"y": y, "away": away, "extra": extra[:8], "missing": missing[:8]})
    return {
        "wjmap": str(path),
        "tp": tp, "fp": fp, "fn": fn,
        "n_disagree_rows": len(misses),
        "disagree_head": misses[:12],
    }


def name_ledge(row: dict) -> str:
    key = (row["floor_top"], row["x0"], row["x1"])
    names = {
        (96, 96, 159): "ship_pad",
        (192, 40, 95): "171_left",
        (192, 160, 215): "171_right",
        (288, 96, 159): "267",
        (384, 160, 215): "363",
        (400, 40, 95): "379_left",
        (496, 96, 159): "475",
        (592, 40, 111): "571",
        (672, 192, 239): "bottom_672",
        (688, 160, 191): "bottom_688",
        (704, 40, 159): "pit",
    }
    return names.get(key, "")


def build(payload: dict) -> dict:
    solid, coll, bts = raster(payload)
    left_pts, right_pts = [], []
    for y, row in enumerate(solid):
        left, right = faces_on_row(row)
        left_pts.extend((y, fx) for fx in left)
        right_pts.extend((y, fx) for fx in right)
    left_runs = [{**r, "side": "left", "kick": "RIGHT", "wj_x": wj_band(r["face_x"], "left")}
                 for r in merge_runs(left_pts)]
    right_runs = [{**r, "side": "right", "kick": "LEFT", "wj_x": wj_band(r["face_x"], "right")}
                  for r in merge_runs(right_pts)]
    ledge_rows = []
    for row in ledges(solid):
        row["name"] = name_ledge(row)
        ledge_rows.append(row)
    bad = self_check(solid)
    wj = compare_wjmap(solid)
    enemies = [
        {"id": e.get("idHex"), "name": e.get("name"), "x": e.get("pixelX"), "y": e.get("pixelY")}
        for e in payload.get("enemies") or []
    ]
    return {
        "room": "0xDF45",
        "name": "Ceres Elevator",
        "source": {
            "editor": str(editor_room_path()),
            "rom_slope_table": "$94:8B2B shape 1 (verified identical in editor rom.sfc)",
            "not_used": "sm-json-data/region/ceres/main/Ceres Elevator Room.json (nodes only)",
        },
        "size_px": [W, H],
        "spin": {"xr": SPIN_XR, "yr": SPIN_YR},
        "stand": {"yr": STAND_YR, "y": "floor_top - 21"},
        "wj_reach_px": WJ_REACH,
        "slope": {
            "shape": 1,
            "unflipped_bts": "0x01 right 8px, inner face 216",
            "xflip_bts": "0x41 left 8px, inner face 40",
            "live_pin": {"right_rest_x": "211.255", "left_rest_x": 45, "xr": 5},
        },
        "shaft": {"left_face": 40, "right_face": 216},
        "ledges": ledge_rows,
        "faces": left_runs + right_runs,
        "chimneys": chimneys(solid),
        "door": payload.get("doors") or [],
        "plms": payload.get("plms") or [],
        "enemies": enemies,
        "chain_hints": {
            "after_475": "box x=96-159 y=496-511; kick LEFT off 96 (wj 83-91) or RIGHT off 160 (wj 165-173)",
            "tas_404_chimney": "y=400-415 air 96..216; TAS latch (155,404) is in this gap, not a tight 8px shaft",
            "363_left_face": "y=384-399 right-wall face 160 (kick LEFT, wj 147-155) — 120px chimney from left face 40",
            "entry_wj": "y=608+ right slope is gone (door); the 475 WJ is off face 216 at y<=607",
        },
        "ascii_blocks": ascii_blocks(coll),
        "self_check_failures": bad,
        "wjmap_compare": wj,
    }


def write_txt(report: dict) -> str:
    lines = [
        "Ceres Elevator 0xDF45 all-map  (editor clip+BTS, live-checked)",
        f"source: {report['source']['editor']}",
        "sm-json-data was not used (no collision in that file).",
        "",
        "Shaft walls: left face x=40 (BTS 0x41)  right face x=216 (BTS 0x01)",
        "Spin xr=5 → rest x=45 / 211.  Standing yr=21 → stand_y = floor_top-21.",
        "",
        "Ledges (product seats in names):",
    ]
    for row in report["ledges"]:
        tag = f"  {row['name']:16s}" if row["name"] else "  (unnamed)       "
        lines.append(
            f"{tag} floor={row['floor_top']:3d} stand_y={row['stand_y']:3d} "
            f"x={row['x0']:3d}-{row['x1']:3d} tile_row={row['row']}"
        )
    lines += ["", "Wall faces (kick away from the solid):"]
    for face in report["faces"]:
        lines.append(
            f"  {face['side']:5s} face_x={face['face_x']:3d} y={face['y0']:3d}-{face['y1']:3d} "
            f"kick {face['kick']:5s} wj_x={face['wj_x'][0]}-{face['wj_x'][1]}"
        )
    lines += ["", "Chain hints (329f TAS gap):"]
    for k, v in report["chain_hints"].items():
        lines.append(f"  {k}: {v}")
    lines += ["", "Chimneys (innermost left_face → right_face):"]
    for ch in report["chimneys"]:
        lines.append(
            f"  y={ch['y0']:3d}-{ch['y1']:3d}  {ch['left_face']:3d}..{ch['right_face']:3d}  "
            f"air={ch['air']}px"
        )
    lines += ["", "Block map  (. air  / slope  # solid  D door)"]
    lines.extend(report["ascii_blocks"])
    wj = report["wjmap_compare"]
    if not wj.get("skipped"):
        lines += [
            "",
            f"wjmap compare tp={wj['tp']} fp={wj['fp']} fn={wj['fn']} "
            f"disagree_rows={wj['n_disagree_rows']}",
        ]
    if report["self_check_failures"]:
        lines += ["", "SELF-CHECK FAILED: " + ", ".join(report["self_check_failures"])]
    else:
        lines += ["", "self-check: all live pins match occupancy."]
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = load_editor()
    report = build(payload)
    slim = {k: v for k, v in report.items() if k != "ascii_blocks"}
    OUT_JSON.write_text(json.dumps(slim, indent=1) + "\n")
    text = write_txt(report)
    OUT_TXT.write_text(text)
    print(text)
    print(f"json: {OUT_JSON}")
    print(f"txt:  {OUT_TXT}")
    if report["self_check_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
