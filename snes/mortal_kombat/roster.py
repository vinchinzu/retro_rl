"""Per-fight model roster for Liu Kang's 12-fight tournament.

Old pixel CNN checkpoints can fill a slot until a RAM v3 specialist exists.
Overnight training writes ``mk1_v3_<prefix>_ppo_final.zip`` and this module
prefers those when present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from mortal_kombat.paths import MODEL_DIR, ROSTER_PATH

STAGES: tuple[tuple[str, str, int], ...] = (
    ("Fight", "Match 1", 0),
    ("Match2", "Match 2", 1),
    ("Match3", "Match 3", 2),
    ("Match4", "Match 4", 3),
    ("Match5", "Match 5", 4),
    ("Match6", "Match 6", 5),
    ("Match7", "Mirror Match", 6),
    ("Endurance1", "Endurance 1 (opp 1)", 7),
    ("Endurance1B", "Endurance 1 (opp 2)", 8),
    ("Endurance2", "Endurance 2 (opp 1)", 9),
    ("Goro", "Goro (E2 opp 2)", 10),
    ("ShangTsung", "Shang Tsung", 11),
)

# Pixel fallbacks from the existing zoo. Not required; overnight retrains all.
PIXEL_FALLBACK: dict[str, str] = {
    "Fight": "mk1_multichar_ppo_2000000_steps.zip",
    "Match2": "mk1_ladder_ft_ppo_final.zip",
    "Match3": "mk1_ladder_ft_ppo_final.zip",
    "Match4": "mk1_match4_ppo_final.zip",
    "Match5": "mk1_ladder_ft_ppo_final.zip",
    "Match6": "mk1_ladder_ft_ppo_final.zip",
    "Match7": "mk1_match7_ppo_9500000_steps.zip",
    "Endurance1": "mk1_speedrun_ppo_final.zip",
    "Endurance1B": "mk1_speedrun_ppo_final.zip",
    "Endurance2": "mk1_speedrun_ppo_final.zip",
    "Goro": "mk1_goro_ppo_final.zip",
    "ShangTsung": "mk1_shangtsung_ppo_final.zip",
}

KIND_RAM_V3 = "ram_v3"
KIND_PIXEL = "pixel"
KIND_SCRIPT = "script"
SCRIPT_NAME = "scripted"


@dataclass
class StageSlot:
    """Best known agent for one tournament fight."""

    prefix: str
    display: str
    match_id: int
    model: str
    kind: str
    win_rate: float | None = None
    attempts: int = 0
    backups: list[str] = field(default_factory=list)

    @property
    def state_name(self) -> str:
        return f"{self.prefix}_LiuKang"


def v3_filename(prefix: str) -> str:
    return f"mk1_v3_{prefix}_ppo_final.zip"


def stage_prefixes() -> list[str]:
    return [prefix for prefix, _, _ in STAGES]


def load_roster(path: Path | None = None) -> dict:
    target = path or ROSTER_PATH
    if target.exists():
        return json.loads(target.read_text())
    return {"updated": None, "stages": {}}


def save_roster(data: dict, path: Path | None = None) -> Path:
    target = path or ROSTER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    target.write_text(json.dumps(data, indent=2) + "\n")
    return target


def record_stage(
    prefix: str,
    *,
    model: str,
    kind: str,
    win_rate: float | None,
    attempts: int,
    path: Path | None = None,
) -> dict:
    data = load_roster(path)
    stages = data.setdefault("stages", {})
    stages[prefix] = {
        "model": model,
        "kind": kind,
        "win_rate": win_rate,
        "attempts": attempts,
    }
    save_roster(data, path)
    return data


def resolve_model(prefix: str, model_dir: Path | None = None) -> tuple[Path, str]:
    """Prefer a v3 RAM specialist; else pixel fallback; else any listed backup.

    A roster entry with kind ``script`` (or model ``scripted``) is a zip-less
    RAM policy and is returned without the path existing.
    """
    directory = model_dir or MODEL_DIR
    roster = load_roster()
    entry = roster.get("stages", {}).get(prefix, {})
    entry_model = entry.get("model")
    entry_kind = entry.get("kind") or KIND_RAM_V3
    if entry_model and (entry_kind == KIND_SCRIPT or entry_model == SCRIPT_NAME):
        return directory / SCRIPT_NAME, KIND_SCRIPT
    candidates: list[tuple[str, str]] = []
    if entry_model:
        candidates.append((entry_model, entry_kind))
    candidates.append((v3_filename(prefix), KIND_RAM_V3))
    fallback = PIXEL_FALLBACK.get(prefix)
    if fallback:
        candidates.append((fallback, KIND_PIXEL))
    for name, kind in candidates:
        path = directory / name
        if path.exists():
            return path, kind
    raise FileNotFoundError(f"No model for stage {prefix} in {directory}")


def build_slots(model_dir: Path | None = None) -> list[StageSlot]:
    """Materialize the 12-fight roster, skipping missing models."""
    directory = model_dir or MODEL_DIR
    roster = load_roster()
    slots: list[StageSlot] = []
    for prefix, display, match_id in STAGES:
        try:
            path, kind = resolve_model(prefix, directory)
        except FileNotFoundError:
            continue
        if kind != KIND_SCRIPT and not path.exists():
            continue
        entry = roster.get("stages", {}).get(prefix, {})
        backups = []
        fallback = PIXEL_FALLBACK.get(prefix)
        if fallback and fallback != path.name:
            backups.append(fallback)
        slots.append(
            StageSlot(
                prefix=prefix,
                display=display,
                match_id=match_id,
                model=path.name,
                kind=kind,
                win_rate=entry.get("win_rate"),
                attempts=int(entry.get("attempts") or 0),
                backups=backups,
            )
        )
    return slots


def slot_for_match(match_id: int, opponent_id: int, slots: list[StageSlot]) -> StageSlot | None:
    """Pick the specialist for the live match / opponent (Goro=7, Shang=8)."""
    if opponent_id == 8:
        want = "ShangTsung"
    elif opponent_id == 7:
        want = "Goro"
    else:
        want = None
        for prefix, _display, mid in STAGES:
            if mid == match_id:
                want = prefix
                break
    if want is None:
        return slots[0] if slots else None
    for slot in slots:
        if slot.prefix == want:
            return slot
    return slots[0] if slots else None


def backup_on_round_loss(slot: StageSlot, model_dir: Path | None = None) -> str | None:
    """Next model to try after a lost round (round-level swap)."""
    directory = model_dir or MODEL_DIR
    for name in slot.backups:
        if name == SCRIPT_NAME:
            continue
        if (directory / name).exists():
            return name
    if SCRIPT_NAME in slot.backups:
        return SCRIPT_NAME
    return None
