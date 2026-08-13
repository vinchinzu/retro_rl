"""Unit tests for shared editor emulator loop helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from retro_harness.emulator_session import (
    DEFAULT_TURBO_FRAME_PREVIEW_INTERVAL,
    EmulatorSpeedController,
    FrameTimingTracker,
    after_step_wram_flags,
    build_script_recording_document,
    compact_snapshot,
    format_speed_label,
    should_include_wram,
    should_preview_turbo_frame,
    step_delay_ms,
    step_repeat_for_speed,
)


class SpeedHelperTests(unittest.TestCase):
    def test_format_speed_label(self) -> None:
        self.assertEqual(format_speed_label(2.0), "2x")
        self.assertEqual(format_speed_label(1.5), "1.5x")

    def test_step_repeat_for_speed(self) -> None:
        self.assertEqual(step_repeat_for_speed(4.0, speed_uses_frame_repeat=True), 4)
        self.assertEqual(step_repeat_for_speed(4.0, speed_uses_frame_repeat=False), 1)
        self.assertEqual(step_repeat_for_speed(1.0, speed_uses_frame_repeat=True), 1)

    def test_turbo_preview_matches_play_session_cadence(self) -> None:
        self.assertEqual(DEFAULT_TURBO_FRAME_PREVIEW_INTERVAL, 8)
        self.assertTrue(should_preview_turbo_frame(8, turbo=True, interval=8))
        self.assertFalse(should_preview_turbo_frame(7, turbo=True, interval=8))
        self.assertTrue(should_preview_turbo_frame(3, turbo=False, interval=8))

    def test_step_delay_unthrottled(self) -> None:
        self.assertEqual(
            step_delay_ms(
                speed=16.0,
                repeat=1,
                frame_ms=2.0,
                target_frame_ms=1,
                base_frame_ms=16,
                unthrottled_speed_threshold=8.0,
            ),
            0,
        )

    def test_speed_controller_ladder(self) -> None:
        ctl = EmulatorSpeedController(levels=(1.0, 2.0, 4.0), default_index=0)
        self.assertEqual(ctl.multiplier, 1.0)
        self.assertTrue(ctl.increase())
        self.assertEqual(ctl.multiplier, 2.0)
        self.assertTrue(ctl.decrease())
        self.assertFalse(ctl.decrease())
        ctl.reset()
        self.assertEqual(ctl.multiplier, 1.0)

    def test_speed_controller_frame_skip(self) -> None:
        ctl = EmulatorSpeedController(
            levels=(1.0, 8.0),
            default_index=1,
            turbo_speed_threshold=4.0,
            turbo_frame_preview_interval=8,
            skip_frame_when_turbo=True,
        )
        included = [ctl.should_include_frame() for _ in range(8)]
        self.assertEqual(included.count(True), 1)
        self.assertTrue(included[-1])


class WramHelperTests(unittest.TestCase):
    def test_should_include_wram_interval(self) -> None:
        # Within interval, no tilemap force -> skip WRAM.
        self.assertFalse(
            should_include_wram(
                include_wram_when_stepping=True,
                force_wram_next_step=False,
                wram_sync_interval_frames=10,
                frame=5,
                last_wram_sync_frame=0,
                synced_tilemap=None,
                tilemap_id=None,
            )
        )
        # At/after interval -> include.
        self.assertTrue(
            should_include_wram(
                include_wram_when_stepping=True,
                force_wram_next_step=False,
                wram_sync_interval_frames=10,
                frame=10,
                last_wram_sync_frame=0,
                synced_tilemap=1,
                tilemap_id=1,
            )
        )
        # Tilemap change forces include even mid-interval.
        self.assertTrue(
            should_include_wram(
                include_wram_when_stepping=True,
                force_wram_next_step=False,
                wram_sync_interval_frames=10,
                frame=5,
                last_wram_sync_frame=0,
                synced_tilemap=1,
                tilemap_id=2,
            )
        )

    def test_after_step_wram_flags(self) -> None:
        frame, force = after_step_wram_flags(
            {"wramRaw": b"x"},
            wram_sync_interval_frames=10,
            synced_tilemap=1,
            tilemap_id=1,
            frame=42,
        )
        self.assertEqual(frame, 42)
        self.assertFalse(force)
        frame, force = after_step_wram_flags(
            {},
            wram_sync_interval_frames=10,
            synced_tilemap=1,
            tilemap_id=2,
            frame=42,
        )
        self.assertIsNone(frame)
        self.assertTrue(force)


class RecordingDocumentTests(unittest.TestCase):
    def test_build_script_recording_document(self) -> None:
        path = Path("/tmp/demo.json")
        doc = build_script_recording_document(
            name="demo",
            button_order=("B", "A"),
            state_file="/states/a.state",
            selected_state_file="/states/a.state",
            start_capture={"paths": {"state": "/states/a.state"}},
            room_label="Field",
            total_frames=12,
            segments=[{"buttons": ["A"], "frames": 12}],
            markers=[],
            last_snapshot={"frameCounter": 12},
            recording_format="editor-script-recording",
            recording_version=1,
            recording_tool="editor",
            bridge_module="game.bridge",
            recording_path=path,
            recorded_at="2026-01-01T00:00:00",
        )
        self.assertEqual(doc["name"], "demo")
        self.assertEqual(doc["totalFrames"], 12)
        self.assertIn("game.bridge", doc["aiUse"]["headlessCommand"])

    def test_compact_snapshot(self) -> None:
        snap = {"a": 1, "b": 2, "c": 3}
        self.assertEqual(compact_snapshot(snap, ("a", "c")), {"a": 1, "c": 3})
        self.assertEqual(compact_snapshot(snap, ()), snap)


class TimingTrackerTests(unittest.TestCase):
    def test_fps_status(self) -> None:
        tracker = FrameTimingTracker()
        text = tracker.status_text(frame_ms=16.0, speed=2.0, script_recording=True)
        self.assertIn("FPS", text)
        self.assertIn("2x", text)
        self.assertIn("script rec", text)


if __name__ == "__main__":
    unittest.main()
