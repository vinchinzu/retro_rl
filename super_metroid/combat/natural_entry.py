"""Natural boss activation capture harness.

Captures a scratch save-state the first time a boss is *active* (or at
settled room-entry) on a continuous prefix or from a source save-state.
Development infrastructure — not continuous evidence by itself. Continuous
acceptance still requires full power-on integrity with natural boss flags.

Shared multi-boss CLI (preferred for non-BT bosses):

```bash
uv run python super_metroid/scripts/probe/natural_entry_cli.py list
uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural bomb_torizo
uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural \\
  kraid --from-state entry --mode room_entry
uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural \\
  phantoon --from-state path/to/phantoon_entry.state --mode room_entry
uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural \\
  botwoon --from-state path/to/botwoon_entry.state --mode room_entry
```

Bomb Torizo back-compat path:

```bash
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py capture-natural
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py prove-natural
```

Capture paths never write boss / event / item / capacity progression RAM.
See ``docs/BOSS_PIPELINE.md``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retro_harness.actions import idle_action
from retro_harness.env import make_env, read_state_bytes, write_state_bytes
from super_metroid.assist import UnlimitedAmmoAssist
from super_metroid.combat.features import (
    BOSS_CATALOG,
    BossCatalogEntry,
    bomb_torizo_catalog,
    features_from_state,
    get_boss_catalog,
    list_boss_catalog,
    validate_live_enemy,
)
from super_metroid.paths import GAME, GAME_DIR, INTEGRATION_DIR, SCRATCH_STATE_DIR
from super_metroid.progression import EARLY_GAME_GRAPH
from super_metroid.ram import parse_state
from super_metroid.routes.continuous import play_start_to_bombs
from super_metroid.routes.runtime import PlayContext, run_continuous

DEFAULT_NATURAL_ACTIVE_STATE = SCRATCH_STATE_DIR / "natural_bomb_torizo_active.state"
DEFAULT_PROVENANCE = SCRATCH_STATE_DIR / "natural_bomb_torizo_active.provenance.json"

# Capture predicates supported by :func:`is_capture_frame`.
CAPTURE_MODES: tuple[str, ...] = ("active", "full_hp", "room_entry", "statue")

# Bosses with a built-in continuous (power-on) play prefix.
CONTINUOUS_PREFIX_BOSSES: frozenset[str] = frozenset({"bomb_torizo"})

# Human / CLI aliases → catalog boss_id.
BOSS_ID_ALIASES: dict[str, str] = {
    "bt": "bomb_torizo",
    "torizo": "bomb_torizo",
    "bomb-torizo": "bomb_torizo",
    "bomb_torizo": "bomb_torizo",
    "spore": "spore_spawn",
    "spore_spawn": "spore_spawn",
    "kraid": "kraid",
    "phantoon": "phantoon",
    "phan": "phantoon",
    "botwoon": "botwoon",
    "botw": "botwoon",
    "draygon": "draygon",
    "dray": "draygon",
    "crocomire": "crocomire",
    "croc": "crocomire",
    "ridley": "ridley",
    "golden_torizo": "golden_torizo",
    "gt": "golden_torizo",
    "mother_brain": "mother_brain",
    "mb": "mother_brain",
}

# Optional short names for --from-state (resolved relative to scratch/integration).
KNOWN_SOURCE_ALIASES: dict[tuple[str, str], Path] = {
    ("kraid", "entry"): SCRATCH_STATE_DIR / "eye_hj_kraid_entry.state",
    ("kraid", "eye"): SCRATCH_STATE_DIR / "eye_hj_kraid_entry.state",
    ("kraid", "eye_hj"): SCRATCH_STATE_DIR / "eye_hj_kraid_entry.state",
    ("kraid", "natural"): SCRATCH_STATE_DIR / "eye_hj_kraid_entry.state",
    ("kraid", "composed"): SCRATCH_STATE_DIR / "warehouse_hijump_kraid_composed.state",
    ("kraid", "dev_kpdr_kraid_entry"): INTEGRATION_DIR / "dev_kpdr_kraid_entry.state",
    ("kraid", "dev_kraid_room_natural"): INTEGRATION_DIR / "dev_kraid_room_natural.state",
    ("phantoon", "entry"): SCRATCH_STATE_DIR / "phantoon_entry.state",
    ("botwoon", "entry"): SCRATCH_STATE_DIR / "botwoon_entry.state",
}

PlayPrefixFn = Callable[[PlayContext], None]


class BossActivationCaptured(Exception):
    """Internal stop signal once combat AI is live (or capture mode hits)."""

    def __init__(
        self,
        frame: int,
        features: dict[str, object],
        *,
        boss_id: str,
        settle: dict[str, object] | None = None,
    ) -> None:
        super().__init__(f"{boss_id}_active_at_frame_{frame}")
        self.frame = frame
        self.features = features
        self.boss_id = boss_id
        self.settle = settle


# Back-compat alias for Bomb Torizo call sites / tests.
TorizoActivationCaptured = BossActivationCaptured


@dataclass(frozen=True)
class NaturalCaptureResult:
    success: bool
    state_path: str
    provenance_path: str | None
    capture_frame: int | None
    features: dict[str, object] | None
    outcome: str
    total_frames: int
    boss_id: str = "bomb_torizo"
    development_only: bool = True
    settle: dict[str, object] | None = None
    source_state: str | None = None
    mode: str | None = None
    progression_writes: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "bossId": self.boss_id,
            "statePath": self.state_path,
            "provenancePath": self.provenance_path,
            "captureFrame": self.capture_frame,
            "features": self.features,
            "settle": self.settle,
            "sourceState": self.source_state,
            "mode": self.mode,
            "outcome": self.outcome,
            "totalFrames": self.total_frames,
            "progressionWrites": self.progression_writes,
            "developmentOnly": self.development_only,
            "acceptanceWarning": (
                "Scratch capture for strategy/RL iteration only; continuous "
                "acceptance still requires natural-entry power-on evidence."
            ),
        }


@dataclass(frozen=True)
class NaturalCaptureConfig:
    """Parameters for a reusable natural-entry capture."""

    boss_id: str
    catalog: BossCatalogEntry
    play_prefix: PlayPrefixFn
    output: Path
    provenance_path: Path | None = None
    max_prefix_frames: int = 60_000
    mode: str = "active"  # active | full_hp | room_entry | statue
    graph: Any = None
    kind: str = "natural_boss_activation"
    source_note: str = ""
    assist_factory: Callable[[], Any] | None = None


def settle_fingerprint(state: Any) -> dict[str, object]:
    """Room + pose + door-settle snapshot for provenance / residual pins."""
    return {
        "roomId": int(state.room_id),
        "roomIdHex": f"0x{int(state.room_id):04X}",
        "pose": int(getattr(state, "pose", 0)),
        "samusX": int(state.samus_x),
        "samusY": int(state.samus_y),
        "doorTransition": int(getattr(state, "door_transition", 0)),
        "gameState": int(getattr(state, "game_state", 0)),
        "enemy0Hp": int(state.enemy0_hp),
        "enemy0Spritemap": int(state.enemy0_spritemap),
        "enemy0SpritemapHex": f"0x{int(state.enemy0_spritemap):04X}",
        "numEnemies": int(getattr(state, "num_enemies", 0)),
    }


def normalize_boss_id(boss_id: str) -> str:
    """Map CLI aliases to catalog ``boss_id``; raise KeyError if unknown."""
    raw = boss_id.strip().lower().replace("-", "_")
    resolved = BOSS_ID_ALIASES.get(raw, raw)
    if resolved not in BOSS_CATALOG:
        known = ", ".join(sorted(BOSS_CATALOG))
        raise KeyError(f"unknown boss_id {boss_id!r}; known: {known}")
    return resolved


def has_continuous_prefix(boss_id: str) -> bool:
    """True when a power-on continuous play prefix is wired for ``boss_id``."""
    return normalize_boss_id(boss_id) in CONTINUOUS_PREFIX_BOSSES


def default_capture_mode(boss_id: str) -> str:
    """Default capture mode: BT uses ``active``; others prefer door settle."""
    bid = normalize_boss_id(boss_id)
    if bid == "bomb_torizo":
        return "active"
    return "room_entry"


def describe_capture_target(boss_id: str) -> dict[str, object]:
    """Machine-readable capture plan for one catalog boss (no emulator)."""
    bid = normalize_boss_id(boss_id)
    catalog = get_boss_catalog(bid)
    state_path, prov_path = default_scratch_paths(bid)
    return {
        "bossId": bid,
        "name": catalog.name,
        "roomId": catalog.room_id,
        "roomIdHex": f"0x{catalog.room_id:04X}",
        "maxHp": catalog.max_hp,
        "continuousStatus": catalog.continuous_status,
        "continuousPrefix": has_continuous_prefix(bid),
        "defaultMode": default_capture_mode(bid),
        "defaultStatePath": str(state_path),
        "defaultProvenancePath": str(prov_path),
        "requiresFromState": not has_continuous_prefix(bid),
        "knownSourceAliases": sorted(
            name for (b, name) in KNOWN_SOURCE_ALIASES if b == bid
        ),
        "notes": (
            "Use capture-natural with continuous prefix (BT only) or "
            "--from-state <path|alias> for doorway/settle capture. "
            "Never forges boss bits; not continuous evidence."
        ),
    }


def list_capture_targets() -> list[dict[str, object]]:
    """All catalog bosses as capture targets (KPDR priority order)."""
    return [describe_capture_target(e.boss_id) for e in list_boss_catalog()]


def resolve_source_state(boss_id: str, name_or_path: str | Path) -> Path:
    """Resolve ``--from-state`` to an existing path (aliases + search roots)."""
    bid = normalize_boss_id(boss_id)
    key = str(name_or_path).strip()
    alias = KNOWN_SOURCE_ALIASES.get((bid, key))
    if alias is not None:
        if alias.exists():
            return alias
        raise FileNotFoundError(
            f"known source alias {key!r} for {bid} missing at {alias}"
        )
    path = Path(key)
    if path.suffix == ".state" or "/" in key or path.exists():
        candidates = (
            path,
            GAME_DIR / path,
            INTEGRATION_DIR / path.name,
            SCRATCH_STATE_DIR / path.name,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"source state not found: {path}")
    # Bare name: try scratch / integration natural and entry names.
    bare_candidates = (
        SCRATCH_STATE_DIR / f"{key}.state",
        SCRATCH_STATE_DIR / key,
        INTEGRATION_DIR / f"{key}.state",
        INTEGRATION_DIR / key,
        SCRATCH_STATE_DIR / f"natural_{bid}_active.state",
    )
    for candidate in bare_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"source state {key!r} not found for {bid}; "
        f"known aliases: {sorted(n for (b, n) in KNOWN_SOURCE_ALIASES if b == bid)}"
    )


def is_capture_frame(
    state: Any,
    catalog: BossCatalogEntry,
    *,
    mode: str = "active",
) -> bool:
    """True when the live enemy0 snapshot is a usable fight start.

    ``mode``:
      - ``active``: combat AI live (features.enemy_active)
      - ``full_hp``: in room at catalog max HP, non-garbage spritemap
      - ``room_entry``: first settled ordinary frame in boss room
      - ``statue``: Bomb Torizo idle statue (legacy)
    """
    if state.room_id != catalog.room_id:
        return False
    if mode == "room_entry":
        return (
            getattr(state, "game_state", 8) == 8
            and getattr(state, "door_transition", 0) == 0
        )
    if mode == "statue":
        # Bomb Torizo-specific idle statue settle.
        if state.enemy0_hp != catalog.max_hp:
            return False
        if state.num_enemies > catalog.max_enemy_slots or state.enemy0_spritemap == 0:
            return False
        return state.enemy0_spritemap == 0x87D0 and state.samus_x < 400
    if mode == "full_hp":
        if catalog.max_hp > 0 and state.enemy0_hp != catalog.max_hp:
            return False
        if state.num_enemies > catalog.max_enemy_slots or state.enemy0_spritemap == 0:
            return False
        if (
            catalog.inactive_spritemaps
            and state.enemy0_spritemap in catalog.inactive_spritemaps
        ):
            return False
        return True
    if mode == "active":
        if catalog.max_hp > 0 and state.enemy0_hp != catalog.max_hp:
            # Prefer first active frame at full HP for strategy/RL starts.
            # Fall through to features.enemy_active if already damaged mid-fight.
            pass
        if state.num_enemies > catalog.max_enemy_slots or state.enemy0_spritemap == 0:
            return False
        feat = features_from_state(state, catalog)
        if catalog.max_hp > 0 and state.enemy0_hp == catalog.max_hp:
            return feat.enemy_active
        return feat.enemy_active
    raise ValueError(f"unknown capture mode: {mode!r}")


# Back-compat private name used by older imports.
def _is_capture_frame(state, catalog, *, mode: str) -> bool:
    return is_capture_frame(state, catalog, mode=mode)


def _write_provenance(
    *,
    provenance_path: Path,
    boss_id: str,
    catalog: BossCatalogEntry,
    mode: str,
    kind: str,
    source_note: str,
    capture_frame: int,
    state_path: Path,
    features: dict[str, object] | None,
    settle: dict[str, object] | None,
    source_state: str | None,
) -> str:
    provenance = {
        "schemaVersion": 1,
        "kind": kind,
        "bossId": boss_id,
        "mode": mode,
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "captureFrame": capture_frame,
        "statePath": str(state_path.resolve()),
        "sourceState": source_state,
        "features": features,
        "settle": settle,
        "source": source_note or f"natural capture for {boss_id}",
        "developmentOnly": True,
        "progressionWrites": 0,
        "forgedBossBits": False,
        "catalog": {
            "name": catalog.name,
            "roomId": catalog.room_id,
            "roomIdHex": f"0x{catalog.room_id:04X}",
            "maxHp": catalog.max_hp,
        },
        "notes": (
            f"mode={mode}: first matching frame. Room + pose + door settle "
            "recorded in settle. Room-load garbage filtered via catalog "
            "max_enemy_slots and inactive spritemaps when mode requires it. "
            "No progression / boss-bit forges."
        ),
    }
    provenance_path = Path(provenance_path)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return str(provenance_path.resolve())


def capture_natural_activation(
    config: NaturalCaptureConfig,
) -> NaturalCaptureResult:
    """Run ``config.play_prefix`` until capture predicate; write scratch state."""
    output = Path(config.output)
    catalog = config.catalog
    boss_id = config.boss_id
    captured: dict[str, object] = {}

    def play(ctx: PlayContext) -> None:
        session = ctx.session
        original_step = session.step

        def step_with_capture(action, reason: str):
            state = original_step(action, reason)
            if session.frame > config.max_prefix_frames:
                raise TimeoutError(
                    f"natural {boss_id} capture exceeded "
                    f"{config.max_prefix_frames} frames without activation "
                    f"(room 0x{state.room_id:04X}, hp={state.enemy0_hp}, "
                    f"spritemap 0x{state.enemy0_spritemap:04X})"
                )
            if is_capture_frame(state, catalog, mode=config.mode):
                # Extra validation for non-statue modes.
                if config.mode in {"active", "full_hp"}:
                    fails = validate_live_enemy(
                        state,
                        catalog,
                        require_active=(config.mode == "active"),
                        require_full_hp=(
                            config.mode == "full_hp" and catalog.max_hp > 0
                        ),
                    )
                    # validate_live_enemy may flag inactive for active mode
                    # if features disagree — still require empty failures.
                    if fails:
                        return state
                feat = features_from_state(state, catalog)
                settle = settle_fingerprint(state)
                write_state_bytes(output, session.env.em.get_state())  # type: ignore[attr-defined]
                captured["frame"] = session.frame
                captured["features"] = feat.to_dict()
                captured["settle"] = settle
                raise BossActivationCaptured(
                    session.frame,
                    feat.to_dict(),
                    boss_id=boss_id,
                    settle=settle,
                )
            return state

        session.step = step_with_capture  # type: ignore[method-assign]
        config.play_prefix(ctx)

    assist = (
        config.assist_factory()
        if config.assist_factory is not None
        else UnlimitedAmmoAssist(enabled=True)
    )
    result = run_continuous(
        play=play,
        assist=assist,
        graph=config.graph if config.graph is not None else EARLY_GAME_GRAPH,
        video_path=None,
        success_outcome=f"natural_{boss_id}_activation_captured",
    )

    progression_writes = int(
        getattr(getattr(assist, "telemetry", None), "progression_writes", 0) or 0
    )

    if isinstance(result.failure, BossActivationCaptured):
        settle = result.failure.settle or captured.get("settle")  # type: ignore[assignment]
        prov_path: str | None = None
        if config.provenance_path is not None:
            prov_path = _write_provenance(
                provenance_path=Path(config.provenance_path),
                boss_id=boss_id,
                catalog=catalog,
                mode=config.mode,
                kind=config.kind,
                source_note=config.source_note,
                capture_frame=result.failure.frame,
                state_path=output,
                features=result.failure.features,
                settle=settle,  # type: ignore[arg-type]
                source_state=None,
            )
        return NaturalCaptureResult(
            success=True,
            state_path=str(output.resolve()),
            provenance_path=prov_path,
            capture_frame=result.failure.frame,
            features=result.failure.features,
            outcome=f"natural_{boss_id}_activation_captured",
            total_frames=result.session.frame,
            boss_id=boss_id,
            settle=settle,  # type: ignore[arg-type]
            mode=config.mode,
            progression_writes=progression_writes,
        )

    if result.failure is not None:
        return NaturalCaptureResult(
            success=False,
            state_path=str(output),
            provenance_path=None,
            capture_frame=None,
            features=None,
            outcome=result.outcome,
            total_frames=result.session.frame,
            boss_id=boss_id,
            mode=config.mode,
            progression_writes=progression_writes,
        )

    return NaturalCaptureResult(
        success=False,
        state_path=str(output),
        provenance_path=None,
        capture_frame=None,
        features=captured.get("features"),  # type: ignore[arg-type]
        outcome="completed_without_capture",
        total_frames=result.session.frame,
        boss_id=boss_id,
        settle=captured.get("settle"),  # type: ignore[arg-type]
        mode=config.mode,
        progression_writes=progression_writes,
    )


def capture_natural_from_source_state(
    boss_id: str,
    source_state: Path | str,
    *,
    mode: str | None = None,
    output: Path | None = None,
    provenance_path: Path | None = None,
    max_frames: int = 3_000,
    source_note: str = "",
) -> NaturalCaptureResult:
    """Load a doorway / predecessor save; idle until capture mode; write scratch.

    Development infrastructure only. Uses ammo refill assist (no capacity /
    progression / boss-bit writes). Preferred path for Phantoon, Botwoon, and
    other bosses that lack a continuous power-on prefix.
    """
    bid = normalize_boss_id(boss_id)
    catalog = get_boss_catalog(bid)
    capture_mode = mode or default_capture_mode(bid)
    if capture_mode not in CAPTURE_MODES:
        raise ValueError(f"unknown capture mode: {capture_mode!r}")

    source = resolve_source_state(bid, source_state)
    state_path, default_prov = default_scratch_paths(bid)
    out = Path(output) if output is not None else state_path
    prov = (
        Path(provenance_path)
        if provenance_path is not None
        else default_prov
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedAmmoAssist(enabled=True)
    try:
        env.reset()
        env.em.set_state(read_state_bytes(source))  # type: ignore[attr-defined]
        state = parse_state(env.get_ram(), frame=0)  # type: ignore[attr-defined]
        assist.apply(env.data, state)  # type: ignore[attr-defined]

        for frame in range(1, max_frames + 1):
            env.step(idle_action())
            state = parse_state(env.get_ram(), frame=frame)  # type: ignore[attr-defined]
            assist.apply(env.data, state)  # type: ignore[attr-defined]
            if not is_capture_frame(state, catalog, mode=capture_mode):
                continue
            if capture_mode in {"active", "full_hp"}:
                fails = validate_live_enemy(
                    state,
                    catalog,
                    require_active=(capture_mode == "active"),
                    require_full_hp=(
                        capture_mode == "full_hp" and catalog.max_hp > 0
                    ),
                )
                if fails:
                    continue
            feat = features_from_state(state, catalog)
            settle = settle_fingerprint(state)
            write_state_bytes(out, env.em.get_state())  # type: ignore[attr-defined]
            note = source_note or (
                f"source-state settle capture from {source} "
                f"(mode={capture_mode})"
            )
            prov_path = _write_provenance(
                provenance_path=prov,
                boss_id=bid,
                catalog=catalog,
                mode=capture_mode,
                kind=f"natural_{bid}_source_capture",
                source_note=note,
                capture_frame=frame,
                state_path=out,
                features=feat.to_dict(),
                settle=settle,
                source_state=str(source.resolve()),
            )
            progression_writes = int(assist.telemetry.progression_writes)
            return NaturalCaptureResult(
                success=True,
                state_path=str(out.resolve()),
                provenance_path=prov_path,
                capture_frame=frame,
                features=feat.to_dict(),
                outcome=f"natural_{bid}_activation_captured",
                total_frames=frame,
                boss_id=bid,
                settle=settle,
                source_state=str(source.resolve()),
                mode=capture_mode,
                progression_writes=progression_writes,
            )

        progression_writes = int(assist.telemetry.progression_writes)
        settle = settle_fingerprint(state)
        return NaturalCaptureResult(
            success=False,
            state_path=str(out),
            provenance_path=None,
            capture_frame=None,
            features=features_from_state(state, catalog).to_dict(),
            outcome=(
                f"timeout_without_capture:mode={capture_mode}:"
                f"room=0x{state.room_id:04X}:door={state.door_transition}"
            ),
            total_frames=max_frames,
            boss_id=bid,
            settle=settle,
            source_state=str(source.resolve()),
            mode=capture_mode,
            progression_writes=progression_writes,
        )
    finally:
        env.close()


def capture_natural_bomb_torizo_activation(
    *,
    output: Path = DEFAULT_NATURAL_ACTIVE_STATE,
    provenance_path: Path | None = DEFAULT_PROVENANCE,
    max_prefix_frames: int = 60_000,
    mode: str = "active",
) -> NaturalCaptureResult:
    """Power-on continuous bombs prefix; save state at first real fight frame.

    Skips room-load garbage (leftover enemy0 from Flyway) by requiring
    ``enemy0_hp == 800`` and a non-zero spritemap with few enemy slots.
    """

    def play_prefix(ctx: PlayContext) -> None:
        play_start_to_bombs(ctx.session, ctx.splits, ctx.segments)

    return capture_natural_activation(
        NaturalCaptureConfig(
            boss_id="bomb_torizo",
            catalog=bomb_torizo_catalog(),
            play_prefix=play_prefix,
            output=Path(output),
            provenance_path=Path(provenance_path) if provenance_path else None,
            max_prefix_frames=max_prefix_frames,
            mode=mode,
            graph=EARLY_GAME_GRAPH,
            kind="natural_bomb_torizo_activation",
            source_note="continuous play_start_to_bombs prefix",
        )
    )


def default_scratch_paths(boss_id: str) -> tuple[Path, Path]:
    """Default ``(state, provenance)`` paths under scratch for ``boss_id``."""
    bid = boss_id if boss_id in BOSS_CATALOG else normalize_boss_id(boss_id)
    state = SCRATCH_STATE_DIR / f"natural_{bid}_active.state"
    prov = SCRATCH_STATE_DIR / f"natural_{bid}_active.provenance.json"
    return state, prov


def capture_config_for_boss(
    boss_id: str,
    play_prefix: PlayPrefixFn,
    *,
    mode: str = "active",
    max_prefix_frames: int = 120_000,
    graph: Any = None,
    source_note: str = "",
) -> NaturalCaptureConfig:
    """Build a :class:`NaturalCaptureConfig` for any registered catalog boss."""
    bid = normalize_boss_id(boss_id)
    catalog = get_boss_catalog(bid)
    state_path, prov_path = default_scratch_paths(bid)
    return NaturalCaptureConfig(
        boss_id=bid,
        catalog=catalog,
        play_prefix=play_prefix,
        output=state_path,
        provenance_path=prov_path,
        max_prefix_frames=max_prefix_frames,
        mode=mode,
        graph=graph,
        kind=f"natural_{bid}_activation",
        source_note=source_note or f"natural capture prefix for {bid}",
    )


def run_capture_natural(
    boss_id: str,
    *,
    from_state: str | Path | None = None,
    mode: str | None = None,
    output: Path | None = None,
    provenance_path: Path | None = None,
    max_prefix_frames: int = 60_000,
    max_source_frames: int = 3_000,
) -> NaturalCaptureResult:
    """Dispatch multi-boss capture-natural (continuous prefix or source state).

    Rules:
    - ``from_state`` set → idle settle from that save (any catalog boss).
    - ``bomb_torizo`` without ``from_state`` → continuous bombs prefix.
    - other bosses without ``from_state`` → structured failure (need source).
    """
    bid = normalize_boss_id(boss_id)
    capture_mode = mode or default_capture_mode(bid)

    if from_state is not None:
        return capture_natural_from_source_state(
            bid,
            from_state,
            mode=capture_mode,
            output=output,
            provenance_path=provenance_path,
            max_frames=max_source_frames,
        )

    if bid == "bomb_torizo":
        state_path, prov_path = default_scratch_paths(bid)
        return capture_natural_bomb_torizo_activation(
            output=Path(output) if output is not None else state_path,
            provenance_path=(
                Path(provenance_path) if provenance_path is not None else prov_path
            ),
            max_prefix_frames=max_prefix_frames,
            mode=capture_mode if capture_mode in ("active", "statue") else "active",
        )

    target = describe_capture_target(bid)
    return NaturalCaptureResult(
        success=False,
        state_path=str(target["defaultStatePath"]),
        provenance_path=None,
        capture_frame=None,
        features=None,
        outcome=(
            f"missing_from_state:{bid}: continuous prefix not wired; "
            "pass --from-state <doorway_or_predecessor.state> "
            f"(aliases={target['knownSourceAliases']})"
        ),
        total_frames=0,
        boss_id=bid,
        mode=capture_mode,
        progression_writes=0,
    )


# ---------------------------------------------------------------------------
# Argparse surface (thin CLI wrappers call these)
# ---------------------------------------------------------------------------


def add_capture_natural_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared ``capture-natural`` flags to a subparser."""
    parser.add_argument(
        "boss",
        help=(
            "Catalog boss id or alias "
            "(bomb_torizo, kraid, phantoon, botwoon, draygon, ...)"
        ),
    )
    parser.add_argument(
        "--from-state",
        type=str,
        default=None,
        help=(
            "Source save path or known alias (e.g. entry for kraid). "
            "Required for bosses without a continuous power-on prefix."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=CAPTURE_MODES,
        default=None,
        help="Capture predicate (default: active for BT, room_entry otherwise)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Scratch .state path (default: scratch/natural_<boss>_active.state)",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=None,
        help="Provenance JSON path (default: next to state)",
    )
    parser.add_argument(
        "--max-prefix-frames",
        type=int,
        default=60_000,
        help="Max frames for continuous power-on prefix (BT)",
    )
    parser.add_argument(
        "--max-source-frames",
        type=int,
        default=3_000,
        help="Max idle frames when capturing from --from-state",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print capture plan JSON without running the emulator",
    )


def build_cli_parser() -> argparse.ArgumentParser:
    """Full multi-boss natural-entry CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Multi-boss natural-entry capture (development infrastructure). "
            "Records room + pose + door settle without progression writes. "
            "Not continuous evidence."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser(
        "list",
        help="List catalog bosses and capture requirements",
    )
    p_list.add_argument(
        "--json",
        action="store_true",
        help="Emit full JSON (default: short table-ish JSON list)",
    )
    p_list.set_defaults(func=_cli_list)

    p_cap = sub.add_parser(
        "capture-natural",
        help=(
            "Capture natural room/pose/door settle for a boss "
            "(continuous prefix or --from-state)"
        ),
    )
    add_capture_natural_arguments(p_cap)
    p_cap.set_defaults(func=_cli_capture_natural)

    p_desc = sub.add_parser(
        "describe",
        help="Describe capture plan for one boss (no emulator)",
    )
    p_desc.add_argument("boss", help="Boss id or alias")
    p_desc.set_defaults(func=_cli_describe)

    return parser


def _cli_list(args: argparse.Namespace) -> int:
    targets = list_capture_targets()
    if args.json:
        print(json.dumps(targets, indent=2))
    else:
        rows = [
            {
                "bossId": t["bossId"],
                "roomIdHex": t["roomIdHex"],
                "continuousPrefix": t["continuousPrefix"],
                "requiresFromState": t["requiresFromState"],
                "defaultMode": t["defaultMode"],
                "continuousStatus": t["continuousStatus"],
            }
            for t in targets
        ]
        print(json.dumps(rows, indent=2))
    return 0


def _cli_describe(args: argparse.Namespace) -> int:
    try:
        target = describe_capture_target(args.boss)
    except KeyError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(target, indent=2))
    return 0


def _cli_capture_natural(args: argparse.Namespace) -> int:
    try:
        bid = normalize_boss_id(args.boss)
    except KeyError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        return 2

    if args.plan_only:
        plan = describe_capture_target(bid)
        plan["requestedMode"] = args.mode or plan["defaultMode"]
        plan["fromState"] = args.from_state
        plan["command"] = "capture-natural"
        plan["planOnly"] = True
        print(json.dumps(plan, indent=2))
        # plan-only is green when catalog resolves (including non-BT).
        return 0

    result = run_capture_natural(
        bid,
        from_state=args.from_state,
        mode=args.mode,
        output=args.output,
        provenance_path=args.provenance,
        max_prefix_frames=args.max_prefix_frames,
        max_source_frames=args.max_source_frames,
    )
    payload = result.to_dict()
    payload["command"] = "capture-natural"
    text = json.dumps(payload, indent=2)
    print(text)
    if args.report is not None:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(text + "\n", encoding="utf-8")
    if result.success:
        return 0
    # missing_from_state is a usage error (2); other failures are 1.
    if result.outcome.startswith("missing_from_state"):
        return 2
    return 1


def cli_main(argv: list[str] | None = None) -> int:
    """Entry for multi-boss natural-entry CLI."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return int(args.func(args))
