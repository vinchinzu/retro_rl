"""Shared play + record spine for randomizer / solver work.

Goals:
- Fast human iteration (windowed play, speed keys, savestates)
- Always-on machine-readable run manifests for demos and seed-robust reports
- Fun HUD (milestones, seed label, streak flair) without game-specific coupling

Game packages call :func:`run_play_spine` or compose the helpers below.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from retro_harness.live_play import play_game
from retro_harness.play_session import PlaySession

# Fun one-liners for HUD (rotates by frame bucket). Keep light.
_FLAIR = (
    "skill library loading…",
    "planner wants snacks",
    "seed gods watching",
    "no tape, only vibes",
    "L4 later · L1 now",
    "record → imitate → plan",
    "natural entry or bust",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def configure_display(*, headless: bool = False) -> None:
    """Set SDL drivers for interactive play or headless record."""
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")
        os.environ.setdefault("HEADLESS", "1")
        return
    if "SDL_VIDEODRIVER" not in os.environ:
        if os.environ.get("WAYLAND_DISPLAY"):
            os.environ["SDL_VIDEODRIVER"] = "wayland"
        else:
            os.environ.setdefault("SDL_VIDEODRIVER", "x11")


@dataclass
class RunManifest:
    """One play/record session — feeds seed-robust reports and demos."""

    game: str
    package: str
    started_at: str
    seed: str | None = None
    seed_settings: dict[str, Any] = field(default_factory=dict)
    start_state: str | None = None
    mode: str = "play"  # play | record | bot
    milestones: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    frames: int = 0
    outcome: str = "open"
    video_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def add_milestone(self, name: str, **extra: Any) -> None:
        entry = {"name": name, "at": utc_now_iso(), "frame": self.frames}
        entry.update(extra)
        self.milestones.append(entry)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def fun_hud_lines(
    *,
    package: str,
    seed: str | None,
    frame: int,
    milestone: str | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """Short HUD for play windows — informative + lightly fun."""
    flair = _FLAIR[(frame // 180) % len(_FLAIR)]
    lines = [
        f"{package} · frame {frame}",
        f"seed {seed or 'vanilla/unset'} · {flair}",
    ]
    if milestone:
        lines.append(f"tip: {milestone}")
    lines.extend(extra)
    return lines


def default_manifest_path(recordings_dir: Path, *, package: str, seed: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_part = (seed or "noseed").replace("/", "_")[:32]
    return Path(recordings_dir) / f"spine_{package}_{seed_part}_{stamp}.json"


def run_play_spine(
    *,
    game: str,
    game_dir: Path | str,
    package: str,
    state: str = "",
    title: str | None = None,
    seed: str | None = None,
    seed_settings: Mapping[str, Any] | None = None,
    recordings_dir: Path | str | None = None,
    scale: int = 3,
    headless: bool | None = None,
    bot: Callable | None = None,
    on_hud: Callable[[dict], list[str]] | None = None,
    on_step: Callable[[dict], None] | None = None,
    env_factory: Callable[[], Any] | None = None,
    write_manifest: bool = True,
    milestone: str | None = None,
    action_size: int = 12,
    **env_kwargs: Any,
) -> RunManifest:
    """Launch interactive (or bot) play and always produce a run manifest.

    Recording of video is left to the game CLI (optional); the spine guarantees
    a JSON run artifact for later multi-seed aggregation.
    """
    game_dir = Path(game_dir).resolve()
    recordings = Path(recordings_dir) if recordings_dir else game_dir / "recordings"
    is_headless = headless if headless is not None else (
        os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes")
    )
    configure_display(headless=is_headless)

    manifest = RunManifest(
        game=game,
        package=package,
        started_at=utc_now_iso(),
        seed=seed,
        seed_settings=dict(seed_settings or {}),
        start_state=state or None,
        mode="bot" if bot is not None else "play",
        meta={"title": title or f"{package}: {state or 'boot'}"},
    )
    if milestone:
        manifest.add_milestone(milestone)

    def _hud(info: dict) -> list[str]:
        frame = int(info.get("frame", 0) or 0)
        manifest.frames = max(manifest.frames, frame)
        if on_hud is not None:
            return on_hud(info)
        return fun_hud_lines(
            package=package,
            seed=seed,
            frame=frame,
            milestone=milestone,
        )

    # PlaySession hooks: wrap session_factory to attach on_step if needed.
    def _session_factory(env: Any, **kwargs: Any) -> PlaySession:
        session = PlaySession(env, **kwargs)
        if on_step is not None:
            original = getattr(session, "on_step", None)

            def _wrapped(info: dict) -> None:
                if callable(original):
                    original(info)
                on_step(info)
                manifest.frames = max(
                    manifest.frames, int(info.get("frame", 0) or session._frame_count)
                )

            session.on_step = _wrapped  # type: ignore[method-assign]
        return session

    try:
        play_game(
            game=game,
            state=state,
            game_dir=game_dir,
            title=title or f"{package} · seed {seed or '—'}",
            scale=scale,
            action_size=action_size,
            bot=bot,
            on_hud=_hud,
            env_factory=env_factory,
            session_factory=_session_factory,
            headless=is_headless,
            **env_kwargs,
        )
        manifest.outcome = "session_end"
    except SystemExit:
        manifest.outcome = "exit"
        raise
    except Exception as exc:  # noqa: BLE001 — surface in manifest then re-raise
        manifest.outcome = "error"
        manifest.notes.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if write_manifest:
            path = default_manifest_path(recordings, package=package, seed=seed)
            manifest.write(path)
            manifest.meta["manifest_path"] = str(path)

    return manifest
