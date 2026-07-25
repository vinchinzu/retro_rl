"""Tests for shared editor bridge and launcher utilities."""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from retro_harness.editor.cursor_agent import (
    EditorAgentContext,
    build_agent_prompt,
    compact_snapshot,
    format_editor_context,
    format_sdk_message,
)
from retro_harness.editor.bridge_protocol import json_response, write_stdio_payload
from retro_harness.editor.bridge_runtime import EditorBridgeRuntime
from retro_harness.editor.bridge_server import handle_bridge_command
from retro_harness.editor.script_segments import normalize_script_segment
from retro_harness.editor_registry import get_editor_project, registered_editor_projects


class FakeEnv:
    def __init__(self, *, width: int = 256, height: int = 224) -> None:
        self.width = width
        self.height = height
        self.frame = 0
        self.state_bytes = b"\x01" * 32
        self.em = mock.Mock()
        self.em.save_state = mock.Mock()

    def reset(self):
        self.frame = 0
        frame = self.render()
        return frame, {}

    def step(self, action):
        del action
        self.frame += 1
        return self.render(), 0.0, False, False, {}

    def render(self):
        value = self.frame % 255
        return np.full((self.height, self.width, 3), value, dtype=np.uint8)

    def close(self) -> None:
        return None


def _build_fake_runtime(tmp_path: Path) -> EditorBridgeRuntime:
    env_holder: dict[str, FakeEnv | None] = {"env": None}

    def make_env(state_name: str | None) -> FakeEnv:
        del state_name
        env_holder["env"] = FakeEnv()
        return env_holder["env"]

    def read_wram(env: object) -> bytes:
        return bytes(getattr(env, "state_bytes", b""))

    def build_snapshot(
        env: object,
        obs: object,
        info: dict[str, object],
        frame_counter: int,
        logical_action: list[int],
    ) -> dict[str, object]:
        del env, obs, info
        return {
            "frameCounter": frame_counter,
            "logicalAction": logical_action,
            "mapName": "TestMap",
        }

    return EditorBridgeRuntime(
        project_root=tmp_path,
        states_dir=tmp_path / "states",
        capture_dir=tmp_path / "captures",
        hot_save_path=tmp_path / "hot.state",
        button_order=("B", "A"),
        make_env=make_env,
        read_wram=read_wram,
        build_snapshot=build_snapshot,
    )


class BridgeProtocolTests(unittest.TestCase):
    def test_json_response_includes_request_metadata(self) -> None:
        payload = json_response(request_id="abc", ok=True, message="ready")
        self.assertEqual(payload["id"], "abc")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["message"], "ready")

    def test_write_stdio_payload_appends_binary_tail(self) -> None:
        payload = {"ok": True}
        writes: list[bytes] = []

        class FakeBuffer:
            def write(self, data: bytes) -> None:
                writes.append(data)

            def flush(self) -> None:
                return None

        fake_stdout = mock.Mock()
        fake_stdout.buffer = FakeBuffer()
        with mock.patch("sys.stdout", fake_stdout):
            write_stdio_payload(payload, frame_rgb=b"rgb", wram=b"ram")
        combined = b"".join(writes)
        self.assertTrue(combined.startswith(b"{"))
        header, binary_tail = combined.split(b"\n", 1)
        self.assertIn(b'"frameBinaryLength":3', header)
        self.assertIn(b'"wramBinaryLength":3', header)
        self.assertEqual(binary_tail, b"rgbram")


class BridgeRuntimeTests(unittest.TestCase):
    def test_step_returns_snapshot_and_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _build_fake_runtime(Path(tmpdir))
            runtime.start_session(state_file="NONE")
            snapshot, frame_rgb, step_ms = runtime.step(
                action=[0] * 12,
                repeat=1,
                include_frame=True,
            )
            self.assertGreaterEqual(step_ms, 0.0)
            self.assertEqual(snapshot["mapName"], "TestMap")
            self.assertEqual(snapshot["wramRaw"], b"\x01" * 32)
            self.assertIsNotNone(frame_rgb)
            self.assertEqual(len(frame_rgb or b""), 256 * 224 * 3)

    def test_step_can_omit_wram_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _build_fake_runtime(Path(tmpdir))
            runtime.start_session(state_file="NONE")
            snapshot, frame_rgb, _step_ms = runtime.step(
                action=[0] * 12,
                repeat=1,
                include_frame=True,
                include_wram=False,
            )
            self.assertNotIn("wramRaw", snapshot)
            self.assertIsNotNone(frame_rgb)

    def test_controller_merge_mutates_action_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _build_fake_runtime(Path(tmpdir))
            runtime.set_controller(object())
            runtime._last_controller_scan_frame = 0
            fake_pygame = types.SimpleNamespace(
                event=types.SimpleNamespace(pump=lambda: None),
            )

            def fake_controller_action(_controller: object, action: list[int]) -> None:
                action[1] = 1

            action = [0] * 12
            with mock.patch.dict("sys.modules", {"pygame": fake_pygame}):
                with mock.patch("retro_harness.controls.controller_action", fake_controller_action):
                    runtime._merge_controller_input(action)

            self.assertEqual(action[1], 1)


class BridgeServerTests(unittest.TestCase):
    def test_discover_lists_reset_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _build_fake_runtime(Path(tmpdir))
            result = handle_bridge_command(
                runtime,
                request_id="1",
                command="discover",
                payload={},
            )
            self.assertTrue(result["ok"])
            states = result["states"]
            self.assertIsInstance(states, list)
            self.assertEqual(states[0]["path"], "NONE")

    def test_set_autoplay_delegates_to_runtime_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _build_fake_runtime(Path(tmpdir))

            def set_autoplay(*, enabled: bool, state_name: object | None = None) -> dict[str, object]:
                return {
                    "autoplayEnabled": enabled,
                    "autoplayStateName": state_name,
                }

            runtime.set_autoplay = set_autoplay  # type: ignore[attr-defined]
            result = handle_bridge_command(
                runtime,
                request_id="1",
                command="set_autoplay",
                payload={"enabled": True, "stateName": "latest", "includeFrame": False},
            )

            self.assertTrue(result["ok"])
            self.assertTrue(result["autoplayEnabled"])
            self.assertEqual(result["autoplayStateName"], "latest")


class ScriptSegmentTests(unittest.TestCase):
    def test_normalize_script_segment_expands_plus_syntax(self) -> None:
        segment = normalize_script_segment("A+B")
        self.assertEqual(segment["buttons"], ["A", "B"])
        self.assertEqual(segment["frames"], 1)


class CursorAgentTests(unittest.TestCase):
    def test_compact_snapshot_drops_frame_bytes(self) -> None:
        snapshot = {
            "mapName": "Onett",
            "frameCounter": 12,
            "frameRgb24Base64": "abc",
            "frameRgb24Raw": b"raw",
        }
        compact = compact_snapshot(snapshot)
        self.assertEqual(compact["mapName"], "Onett")
        self.assertNotIn("frameRgb24Base64", compact)
        self.assertNotIn("frameRgb24Raw", compact)

    def test_build_agent_prompt_includes_context_and_request(self) -> None:
        context = EditorAgentContext(
            title="Test Editor",
            summary="block (1, 2)",
            details={"selected_block": {"x": 1, "y": 2}},
        )
        prompt = build_agent_prompt(
            "Why is this NPC missing?",
            instructions=("Stay concise.",),
            context=context,
            published_context="Previously published notes",
        )
        self.assertIn("Stay concise.", prompt)
        self.assertIn("Why is this NPC missing?", prompt)
        self.assertIn("Previously published notes", prompt)
        self.assertIn('"selected_block"', prompt)

    def test_format_sdk_message_renders_assistant_and_tool_lines(self) -> None:
        assistant = mock.Mock()
        assistant.type = "assistant"
        text_block = mock.Mock()
        text_block.type = "text"
        text_block.text = "hello"
        tool_block = mock.Mock()
        tool_block.type = "tool_use"
        tool_block.name = "Read"
        assistant.message.content = (text_block, tool_block)

        tool_call = mock.Mock()
        tool_call.type = "tool_call"
        tool_call.name = "Grep"
        tool_call.status = "completed"

        self.assertEqual(format_sdk_message(assistant), ["hello", "[tool request] Read"])
        self.assertEqual(format_sdk_message(tool_call), ["[tool completed] Grep"])

    def test_format_editor_context_is_markdown_json(self) -> None:
        rendered = format_editor_context(
            EditorAgentContext(title="EarthBound", summary="live map", details={"x": 1})
        )
        self.assertIn("### EarthBound", rendered)
        self.assertIn("```json", rendered)


class EditorRegistryTests(unittest.TestCase):
    def test_harvest_editor_is_registered(self) -> None:
        project = get_editor_project("harvest")
        self.assertEqual(project.editor_module, "harvest.tools.editor_app")
        self.assertEqual(project.bridge_module, "harvest.runtime.editor_bridge")

    def test_earthbound_editor_is_registered(self) -> None:
        project = get_editor_project("earthbound")
        self.assertEqual(project.editor_module, "earthbound_editor.__main__")
        self.assertEqual(project.bridge_module, "earthbound_editor.editor_bridge")

    def test_registry_entries_are_unique(self) -> None:
        ids = [project.project_id for project in registered_editor_projects()]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
