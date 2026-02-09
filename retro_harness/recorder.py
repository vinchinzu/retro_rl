"""
Generic labeled recording infrastructure for stable-retro games.

Provides:
- Labeled save state sets with auto-incrementing indices
- Recording sessions with metadata
- F5 = QuickSave, F6 = auto-increment labeled save point
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import stable_retro as retro


@dataclass
class SavePointSet:
    """A labeled set of save points with auto-incrementing indices."""

    label: str
    game: str
    game_dir: Path
    save_dir: Path
    index: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_index()

    def _load_existing_index(self):
        """Find the highest existing index for this label."""
        pattern = f"{self.label}_*.state"
        existing = list(self.save_dir.glob(pattern))
        if existing:
            indices = []
            for p in existing:
                stem = p.stem
                suffix = stem[len(self.label) + 1 :]
                try:
                    idx = int(suffix)
                    indices.append(idx)
                except ValueError:
                    continue
            if indices:
                self.index = max(indices) + 1

    def save(self, env: retro.RetroEnv, extra_meta: dict | None = None) -> Path:
        """Save a new auto-incrementing save point."""
        state_data = env.em.get_state()
        name = f"{self.label}_{self.index:03d}"
        filename = f"{name}.state"

        # Save to save_dir
        save_path = self.save_dir / filename
        with gzip.open(save_path, "wb") as f:
            f.write(state_data)

        # Save to custom_integrations for loading
        ci_path = (
            Path(self.game_dir) / "custom_integrations" / self.game / filename
        )
        ci_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(ci_path, "wb") as f:
            f.write(state_data)

        # Save metadata
        meta = {
            "label": self.label,
            "index": self.index,
            "timestamp": time.time(),
            "name": name,
            **(extra_meta or {}),
        }
        meta_path = self.save_dir / f"{name}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self.index += 1
        return save_path

    def get_all(self) -> list[tuple[Path, dict]]:
        """Get all save points for this label with their metadata."""
        results = []
        pattern = f"{self.label}_*.state"
        for state_path in sorted(self.save_dir.glob(pattern)):
            meta_path = state_path.with_suffix(".json")
            meta = {}
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    # Skip corrupted metadata files
                    meta = {"error": "corrupted metadata"}
            results.append((state_path, meta))
        return results


@dataclass
class RecordingSession:
    """Tracks a recording session with labeled checkpoints."""

    label: str
    game: str
    game_dir: Path
    session_id: str = field(default_factory=lambda: str(int(time.time())))
    recordings_dir: Path | None = None
    _save_set: SavePointSet | None = field(default=None, repr=False)
    _manifest: list[dict] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.recordings_dir is None:
            self.recordings_dir = Path(self.game_dir) / "recordings"
        self.recordings_dir = Path(self.recordings_dir)

        # Create labeled subdirectory for this session's saves
        self.save_dir = self.recordings_dir / "states" / self.label
        self._save_set = SavePointSet(
            label=self.label,
            game=self.game,
            game_dir=self.game_dir,
            save_dir=self.save_dir,
        )

    @property
    def current_index(self) -> int:
        """Current save point index (next save will use this)."""
        return self._save_set.index

    def quick_save(self, env: retro.RetroEnv, name: str = "QuickSave") -> Path:
        """F5-style quick save (overwrites)."""
        state_data = env.em.get_state()
        filename = f"{name}.state"

        # Save to game dir
        save_path = Path(self.game_dir) / filename
        with gzip.open(save_path, "wb") as f:
            f.write(state_data)

        # Also to custom_integrations
        ci_path = (
            Path(self.game_dir) / "custom_integrations" / self.game / filename
        )
        ci_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(ci_path, "wb") as f:
            f.write(state_data)

        return save_path

    def checkpoint(
        self, env: retro.RetroEnv, extra_meta: dict | None = None
    ) -> Path:
        """F6-style auto-incrementing labeled checkpoint."""
        return self._save_set.save(env, extra_meta)

    def get_all_checkpoints(self) -> list[tuple[Path, dict]]:
        """Get all checkpoints for this label."""
        return self._save_set.get_all()


class LabeledRecorder:
    """
    Generic recorder with labeled save point sets.

    Usage:
        recorder = LabeledRecorder(
            game="DonkeyKongCountry-Snes",
            game_dir=Path("donkey_kong_country"),
            label="level_217",
        )
        recorder.start(env)

        # In game loop:
        recorder.handle_key(pygame.K_F5, env)  # Quick save
        recorder.handle_key(pygame.K_F6, env, meta={"level_id": 217})  # Checkpoint
    """

    def __init__(
        self,
        game: str,
        game_dir: Path,
        label: str,
        recordings_dir: Path | None = None,
        on_save: Callable[[Path, str], None] | None = None,
    ):
        self.game = game
        self.game_dir = Path(game_dir)
        self.label = label
        self.recordings_dir = recordings_dir
        self.on_save = on_save
        self.session: RecordingSession | None = None

    def start(self, env: retro.RetroEnv | None = None) -> RecordingSession:
        """Start a new recording session."""
        self.session = RecordingSession(
            label=self.label,
            game=self.game,
            game_dir=self.game_dir,
            recordings_dir=self.recordings_dir,
        )
        return self.session

    def handle_key(
        self,
        key: int,
        env: retro.RetroEnv,
        pygame,
        extra_meta: dict | None = None,
    ) -> Path | None:
        """
        Handle F5/F6 key presses.

        Returns path if a save was made, None otherwise.
        """
        if self.session is None:
            self.start(env)

        K_F5 = getattr(pygame, "K_F5", 286)
        K_F6 = getattr(pygame, "K_F6", 287)

        if key == K_F5:
            path = self.session.quick_save(env)
            if self.on_save:
                self.on_save(path, "quick")
            return path
        elif key == K_F6:
            path = self.session.checkpoint(env, extra_meta)
            if self.on_save:
                self.on_save(path, "checkpoint")
            return path

        return None

    @property
    def current_index(self) -> int:
        """Current checkpoint index."""
        if self.session is None:
            return 0
        return self.session.current_index


def list_labeled_states(
    game_dir: Path, label: str | None = None
) -> dict[str, list[tuple[Path, dict]]]:
    """
    List all labeled save states in a game directory.

    Args:
        game_dir: Path to game directory
        label: Optional filter for specific label

    Returns:
        Dict mapping labels to list of (path, metadata) tuples
    """
    states_dir = Path(game_dir) / "recordings" / "states"
    if not states_dir.exists():
        return {}

    results = {}
    for label_dir in states_dir.iterdir():
        if not label_dir.is_dir():
            continue
        if label and label_dir.name != label:
            continue

        items = []
        for state_path in sorted(label_dir.glob("*.state")):
            meta_path = state_path.with_suffix(".json")
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            items.append((state_path, meta))

        if items:
            results[label_dir.name] = items

    return results
