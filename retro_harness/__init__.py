"""
Shared harness for retro gaming RL projects.

Provides common abstractions for SNES emulation via stable-retro:
- Controls: Keyboard and controller input handling
- Protocol: Task interfaces for composable behaviors
- Env: Environment setup utilities
- RAM State: Declarative RAM reading
- Splits: Segment timing / speedrun splits
- Play Session: Generic pygame play loop
- Bot Runner: Task-based autopilot framework
"""

from retro_harness.controls import (
    SNES_A,
    SNES_B,
    SNES_BUTTON_NAME_TO_INDEX,
    SNES_BUTTON_NAMES,
    SNES_DOWN,
    SNES_L,
    SNES_LEFT,
    SNES_R,
    SNES_RIGHT,
    SNES_SELECT,
    SNES_START,
    SNES_UP,
    SNES_X,
    SNES_Y,
    CONTROLLER_MAP,
    controller_action,
    controller_debug_snapshot,
    describe_input_mapping,
    describe_controller,
    format_input_mapping,
    keyboard_action,
    action_from_snes_button_names,
    parse_snes_button_label,
    pressed_snes_buttons,
    sanitize_action,
    sanitize_action_multi,
    sanitize_action_offset,
    init_controller,
    init_controllers,
)

from retro_harness.protocol import (
    TaskStatus,
    WorldState,
    ActionResult,
    TaskResult,
    Task,
)

from retro_harness.env import (
    GameSpec,
    add_custom_integrations,
    make_env,
    get_available_states,
    read_state_bytes,
    save_state,
    state_path,
    write_state_bytes,
)
from retro_harness.actions import (
    ActionBuilder,
    SNES_ACTION_SIZE,
    action_names,
    buttons,
    buttons_multi,
    idle_action,
    idle_action_multi,
    indexed_action,
    multiplayer_action,
    snes_action,
)
from retro_harness.input_script import (
    FrameAction,
    InputStep,
    ScriptResult,
    StartupPlan,
    input_step,
    parse_input_script,
    press_button_sequence,
    run_input_steps,
    run_startup,
)
from retro_harness.runtime import reset_env, step_env
from retro_harness.emulator_pool import EmulatorPool, PoolState
from retro_harness.live_play import (
    play_game,
)
from retro_harness.recordings import (
    ensure_gzip_state,
    append_jsonl,
    iter_jsonl,
    find_latest_recording,
    find_latest_recording_from_manifest,
)
from retro_harness.recorder import (
    SavePointSet,
    RecordingSession,
    LabeledRecorder,
    list_labeled_states,
)
from retro_harness.video import (
    FrameVideoWriter,
    VideoCaptureConfig,
    VideoRecorder,
    concat_videos,
    format_snes_buttons,
    probe_video_evidence,
    render_button_footer,
    should_capture_frame,
)
from retro_harness.ram_state import (
    RAMSchema,
    RAMWatcher,
    read_u8,
    read_u16,
    read_u16_be,
    read_s8,
    read_s16,
)
from retro_harness.splits import (
    SplitTracker,
    SplitResult,
)
from retro_harness.bot_runner import (
    BotRunner,
    TaskSequencer,
    TaskRepeater,
)
from retro_harness.benchmark import (
    RuntimeObservationClass,
    InterventionClass,
    StartIdentity,
    PolicyIdentity,
    PolicyArtifact,
    PolicyArtifactError,
    policy_identity_for,
    EvaluationContract,
    AuditCapabilities,
    AttemptAudit,
    AuditedEnv,
    ClaimValidationError,
    validate_claim,
    BenchmarkTier,
    BenchmarkCase,
    BenchmarkAttemptResult,
    BenchmarkRunResult,
    IdlePolicy,
    RandomPolicy,
    run_benchmark,
    zero_action_for_env,
)
from retro_harness.solver import (
    ObservationRequirement,
    ProgressionDelta,
    SkillInstance,
    SkillOutcome,
    SkillOutcomeStatus,
    SkillPolicy,
    SkillSignal,
    SkillSpec,
    SkillStep,
    SolverLifecycle,
    SolverObservation,
    SolverResultStatus,
    SolverSession,
    SolverSessionResult,
    SolverActionEvent,
    SolverTraceEvent,
    canonical_action_record,
)
from retro_harness.contracts import (
    ActionContract,
    ActionEntry,
    ContractBundle,
    ContractError,
    ContractMismatchError,
    EnvironmentContract,
    ObservationContract,
    ObservationField,
    RewardComponent,
    RewardContract,
    WrapperContract,
    WrapperSpec,
)
from retro_harness.entry_states import (
    EntryStateCorpus,
    EntryStateCorpusBuilder,
    EntryStateError,
    EntryStateRecord,
    EntryStateSplit,
    SplitStrategy,
)
from retro_harness.trajectory import (
    CounterexampleLibrary,
    TRAJECTORY_SCHEMA_DIGEST,
    Trajectory,
    TrajectoryError,
    TrajectoryStep,
    counterexamples_from_solver_result,
    trajectory_from_solver_result,
)
from retro_harness.mission_control import (
    MissionSnapshot,
    MissionAware,
)
# PlaySession imported lazily (depends on pygame)

__all__ = [
    # Controls
    "SNES_A", "SNES_B", "SNES_DOWN", "SNES_L", "SNES_LEFT",
    "SNES_R", "SNES_RIGHT", "SNES_SELECT", "SNES_START",
    "SNES_UP", "SNES_X", "SNES_Y", "SNES_BUTTON_NAME_TO_INDEX", "SNES_BUTTON_NAMES", "CONTROLLER_MAP",
    "controller_action", "controller_debug_snapshot", "describe_input_mapping", "describe_controller",
    "format_input_mapping",
    "keyboard_action", "action_from_snes_button_names", "parse_snes_button_label",
    "pressed_snes_buttons", "sanitize_action", "sanitize_action_multi",
    "sanitize_action_offset", "init_controller",
    "init_controllers",
    # Protocol
    "TaskStatus", "WorldState", "ActionResult", "TaskResult", "Task",
    # Named actions and scripts
    "ActionBuilder", "FrameAction", "InputStep", "SNES_ACTION_SIZE",
    "ScriptResult", "StartupPlan", "action_names", "buttons", "buttons_multi",
    "idle_action", "idle_action_multi", "indexed_action", "input_step", "multiplayer_action",
    "parse_input_script", "press_button_sequence", "run_input_steps",
    "run_startup", "snes_action",
    # Env
    "GameSpec", "add_custom_integrations", "make_env", "get_available_states",
    "read_state_bytes", "reset_env", "save_state", "state_path", "step_env",
    "write_state_bytes", "play_game",
    # Deterministic parallel rollouts
    "EmulatorPool", "PoolState",
    # Recordings/logging
    "ensure_gzip_state", "append_jsonl", "iter_jsonl",
    "find_latest_recording", "find_latest_recording_from_manifest",
    # Labeled recorder
    "SavePointSet", "RecordingSession", "LabeledRecorder", "list_labeled_states",
    # Video capture (shared continuous / showcase MP4)
    "FrameVideoWriter", "VideoCaptureConfig", "VideoRecorder", "concat_videos",
    "format_snes_buttons", "probe_video_evidence", "render_button_footer",
    "should_capture_frame",
    # RAM state
    "RAMSchema", "RAMWatcher", "read_u8", "read_u16", "read_u16_be", "read_s8", "read_s16",
    # Splits
    "SplitTracker", "SplitResult",
    # Bot runner
    "BotRunner", "TaskSequencer", "TaskRepeater",
    # Benchmarks
    "RuntimeObservationClass", "InterventionClass", "StartIdentity", "PolicyIdentity",
    "PolicyArtifact", "PolicyArtifactError",
    "policy_identity_for",
    "EvaluationContract", "AuditCapabilities", "AttemptAudit", "AuditedEnv",
    "ClaimValidationError", "validate_claim",
    "BenchmarkTier", "BenchmarkCase", "BenchmarkAttemptResult", "BenchmarkRunResult",
    "IdlePolicy", "RandomPolicy", "run_benchmark", "zero_action_for_env",
    # Solver runtime
    "ObservationRequirement", "ProgressionDelta", "SkillInstance", "SkillOutcome",
    "SkillOutcomeStatus", "SkillPolicy", "SkillSignal", "SkillSpec", "SkillStep",
    "SolverLifecycle", "SolverObservation", "SolverResultStatus", "SolverSession",
    "SolverSessionResult", "SolverActionEvent", "SolverTraceEvent",
    "canonical_action_record",
    # Versioned model/environment contracts
    "ActionContract", "ActionEntry", "ContractBundle", "ContractError",
    "ContractMismatchError", "EnvironmentContract", "ObservationContract",
    "ObservationField", "RewardComponent", "RewardContract",
    "WrapperContract", "WrapperSpec",
    # Natural-entry state distributions
    "EntryStateCorpus", "EntryStateCorpusBuilder", "EntryStateError",
    "EntryStateRecord", "EntryStateSplit", "SplitStrategy",
    # Canonical experience and retained failures
    "CounterexampleLibrary", "TRAJECTORY_SCHEMA_DIGEST", "Trajectory",
    "TrajectoryError", "TrajectoryStep", "counterexamples_from_solver_result",
    "trajectory_from_solver_result",
    # Mission control
    "MissionSnapshot", "MissionAware",
]
