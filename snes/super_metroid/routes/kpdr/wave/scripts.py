"""Open-loop RLE tables for K4 Wave branch (Double Chamber gate open).

Source: human tape frames 4650–5200. Product loads
``routes/kpdr/data/double_chamber_gate_open_rle.json`` — do not invent shot
cadences or re-inline the tuple here.
"""

from __future__ import annotations

from pathlib import Path

from super_metroid.routes.rle import RleScript, load_rle_json

_GATE_OPEN_RLE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "double_chamber_gate_open_rle.json"
)

# Canonical human open sequence (tape frames 4650–5200).
HUMAN_GATE_OPEN_RLE: RleScript = load_rle_json(_GATE_OPEN_RLE_PATH)

__all__ = [
    "RleScript",
    "HUMAN_GATE_OPEN_RLE",
]
