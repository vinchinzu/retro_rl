"""Imported legacy model registry for later route expansion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from super_metroid.paths import MODELS_DIR


@dataclass(frozen=True)
class LegacyModel:
    model_id: str
    filename: str
    role: str
    framework: str
    sha256: str
    source: str
    status: str

    @property
    def path(self) -> Path:
        return MODELS_DIR / "imported" / self.filename


def load_model_registry(path: Path = MODELS_DIR / "manifest.json") -> dict[str, LegacyModel]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["id"]: LegacyModel(
            model_id=item["id"],
            filename=item["filename"],
            role=item["role"],
            framework=item["framework"],
            sha256=item["sha256"],
            source=item["source"],
            status=item["status"],
        )
        for item in payload["models"]
    }

