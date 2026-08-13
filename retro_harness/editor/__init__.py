"""Game-agnostic Qt editor ↔ subprocess emulator bridge primitives."""

from retro_harness.editor.bridge_client import (
    EditorBridgeClient,
    bridge_python_command,
    read_exact,
)
from retro_harness.editor.bridge_worker import (
    BridgeController,
    BridgeReply,
    BridgeRequest,
    BridgeWorker,
)
from retro_harness.editor.bridge_runtime import EditorBridgeRuntime
from retro_harness.editor.bridge_server import (
    handle_bridge_command,
    run_stdio_bridge,
)
from retro_harness.editor.bridge_protocol import (
    emit_stdio_response,
    json_response,
    reset_env,
    step_env,
    write_stdio_payload,
)
from retro_harness.editor.map_rgba import (
    blend_rect,
    draw_layer_marker,
    fill_rect,
    outline_rect,
)
from retro_harness.editor.recording import (
    append_recording_marker,
    append_recording_segment,
    recording_buttons_for_action,
    safe_recording_slug,
)
from retro_harness.editor.script_segments import (
    normalize_script_segment,
    script_segments_from_file,
    script_segments_from_payload,
)
from retro_harness.editor.snapshot import (
    snapshot_frame_counter,
    snapshot_int,
    snapshot_without_frame,
)
from retro_harness.editor.gui_emulator_panel import (
    EmbeddedEmulatorPanelBase,
    EmulatorPanelConfig,
)
from retro_harness.editor.cursor_agent import (
    EditorAgentContext,
    build_agent_prompt,
    compact_snapshot,
    cursor_sdk_available,
    format_editor_context,
    format_sdk_message,
)
from retro_harness.editor.cursor_agent_panel import (
    CursorAgentPanel,
    CursorAgentPanelConfig,
    attach_cursor_agent_dock,
)
from retro_harness.editor.util import (
    frame_budget_ms_for_speed,
    int_value,
)
from retro_harness.emulator_session import (
    EmulatorSpeedController,
    FrameTimingTracker,
    format_speed_label,
    should_preview_turbo_frame,
)

__all__ = [
    "BridgeController",
    "BridgeReply",
    "BridgeRequest",
    "BridgeWorker",
    "EditorBridgeClient",
    "EditorBridgeRuntime",
    "EmbeddedEmulatorPanelBase",
    "EmulatorPanelConfig",
    "EditorAgentContext",
    "CursorAgentPanel",
    "CursorAgentPanelConfig",
    "attach_cursor_agent_dock",
    "build_agent_prompt",
    "compact_snapshot",
    "cursor_sdk_available",
    "format_editor_context",
    "format_sdk_message",
    "append_recording_marker",
    "append_recording_segment",
    "blend_rect",
    "bridge_python_command",
    "draw_layer_marker",
    "emit_stdio_response",
    "fill_rect",
    "EmulatorSpeedController",
    "FrameTimingTracker",
    "format_speed_label",
    "frame_budget_ms_for_speed",
    "should_preview_turbo_frame",
    "int_value",
    "json_response",
    "normalize_script_segment",
    "outline_rect",
    "handle_bridge_command",
    "read_exact",
    "recording_buttons_for_action",
    "reset_env",
    "run_stdio_bridge",
    "safe_recording_slug",
    "script_segments_from_file",
    "script_segments_from_payload",
    "snapshot_frame_counter",
    "snapshot_int",
    "snapshot_without_frame",
    "step_env",
    "write_stdio_payload",
]
