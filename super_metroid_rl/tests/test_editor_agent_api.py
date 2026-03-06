from __future__ import annotations

import json
from pathlib import Path

import pytest

from super_metroid_rl.editor_agent_api import EditorAgentApi


pytest.importorskip("stable_retro")


def test_editor_agent_api_starts_zebes_and_observes_landing_site(tmp_path: Path) -> None:
    export_dir = tmp_path / "sm_export"
    export_dir.mkdir()
    (export_dir / "nav_graph.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "roomId": 0x91F8,
                        "roomIdHex": "0x91F8",
                        "handle": "landingSite",
                        "name": "Landing Site",
                        "area": 0,
                        "areaName": "Crateria",
                        "mapX": 23,
                        "mapY": 0,
                        "widthScreens": 9,
                        "heightScreens": 5,
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )

    with EditorAgentApi() as api:
        hello = api.request("hello", include_frame=False)
        assert hello["capabilities"]["supportsAgentControl"] is True

        configured = api.configure(nav_export_dir=str(export_dir), control_mode="watch")
        assert configured["session"]["controlMode"] == "watch"

        started = api.start_session(
            "ZebesStart",
            nav_export_dir=str(export_dir),
            control_mode="watch",
            include_frame=False,
        )
        snapshot = started["snapshot"]
        assert snapshot["roomId"] == 0x91F8
        assert snapshot["roomName"] == "Landing Site"
        assert snapshot["areaName"] == "Crateria"
        assert snapshot.get("frameRgb24Base64") is None

        left = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        stepped = api.step(left, repeat=8, include_frame=False)
        step_snapshot = stepped["snapshot"]
        assert step_snapshot["roomId"] == 0x91F8
        assert len(step_snapshot["trace"]) >= 2
        assert step_snapshot["samusX"] <= snapshot["samusX"]

        closed = api.request("close_session", include_frame=False)
        assert closed["session"]["active"] is False
