"""Machine-checkable documentation integrity tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "docs" / "manifests"

# Top-level docs that participate in link checks (not vendored trees).
DOC_GLOBS = (
    "*.md",
    "docs/*.md",
    "snes_oneshot/*.md",
    "snes_oneshot/docs/*.md",
    "snes_oneshot/docs/archive/*.md",
    "ADDING_GAMES.md",
    "AGENTS.md",
    "README.md",
    "BENCHMARK_STATUS.md",
    "ARCHITECTURE_AND_CLEANUP_PLAN.md",
)

ACTIVE_GAME_DIRS = (
    "alttp",
    "battle_clash",
    "castlevania",
    "contra",
    "ducktales",
    "f_zero",
    "final_fight",
    "great_waldo_search",
    "joe_and_mac",
    "kirby_adventure",
    "magical_quest",
    "mega_man_2",
    "metroid",
    "pilotwings",
    "punch_out",
    "rival_turf",
    "smb",
    "smb3",
    "star_fox",
    "super_double_dragon",
    "super_metroid",
    "tmnt_i",
    "tmnt_ii",
    "tmnt_iii",
    "tmnt_iv",
    "zelda_i",
    "zelda_ii",
)

STALE_LINK_RE = re.compile(
    r"\[[^\]]*\]\((?:\.\./)*"
    r"(?:super_metroid_rl|super_mario_bros)(?:/[^)]*)?\)"
)

MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
BACKTICK_TOP_DIR_RE = re.compile(r"`([A-Za-z0-9_.-]+/)`")
MATURITY_RE = re.compile(
    r"Current maturity:\s*M[0-8]\b|"
    r"\|\s*Current maturity\s*\|\s*M[0-8]\b|"
    r"\|\s*Maturity\s*\|\s*M[0-8]\b",
    re.IGNORECASE,
)
LAST_VERIFIED_RE = re.compile(
    r"Last verification:\s*\d{4}-\d{2}-\d{2}|"
    r"\|\s*Last verification\s*\|\s*\d{4}-\d{2}-\d{2}|"
    r"last_verified:\s*\d{4}-\d{2}-\d{2}",
    re.IGNORECASE,
)


def _doc_files() -> list[Path]:
    files: list[Path] = []
    for pattern in DOC_GLOBS:
        files.extend(ROOT.glob(pattern))
    # Active game STATUS/plan/AGENTS
    for game in ACTIVE_GAME_DIRS:
        files.extend((ROOT / game).glob("AGENTS.md"))
        files.extend((ROOT / game).glob("docs/*.md"))
    # Deduplicate
    return sorted({path.resolve() for path in files if path.is_file()})


def _resolve_link(source: Path, target: str) -> Path | None:
    if target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    cleaned = target.split("#", 1)[0].strip()
    if not cleaned:
        return None
    return (source.parent / cleaned).resolve()


@pytest.fixture(scope="module")
def manifests() -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for path in sorted(MANIFEST_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), path
        items.append(data)
    assert items, "expected at least one game manifest"
    return items


def test_relative_markdown_links_resolve() -> None:
    missing: list[str] = []
    for path in _doc_files():
        if "archive" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for _label, target in MD_LINK_RE.findall(text):
            resolved = _resolve_link(path, target)
            if resolved is None:
                continue
            if not resolved.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not missing, "Broken relative Markdown links:\n" + "\n".join(missing)


def test_backticked_workspace_directories_exist() -> None:
    """Root program docs must not claim missing top-level workspaces."""
    known_top = {p.name for p in ROOT.iterdir() if p.is_dir()}
    retired = {
        "super_metroid_rl",
        "super_mario_bros",
    }
    allowed_future = {"adventure_common"}
    # Common game-local or generic folder names, not repo root workspaces.
    non_workspace = {
        "scripts",
        "models",
        "logs",
        "recordings",
        "maps",
        "docs",
        "tests",
        "roms",
            "custom_integrations",
            "manifests",
            "archive",
            "debug_*",
        }
    program_docs = {
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "ADDING_GAMES.md",
        ROOT / "BENCHMARK_STATUS.md",
        *sorted((ROOT / "docs").glob("*.md")),
        ROOT / "snes_oneshot" / "docs" / "STATUS.md",
        ROOT / "snes_oneshot" / "docs" / "GAME_SELECTION_NOTES.md",
        ROOT / "snes_oneshot" / "docs" / "FULL_RUN_PROCESS.md",
    }
    missing: list[str] = []
    for path in program_docs:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for match in BACKTICK_TOP_DIR_RE.findall(text):
            name = match.rstrip("/")
            if name in retired:
                # Explicit "do not use" callouts are allowed in program docs.
                continue
            if name in allowed_future or name in non_workspace:
                continue
            if name.startswith("debug_"):
                continue
            if name not in known_top:
                missing.append(f"{path.relative_to(ROOT)}: `{name}/`")
    assert not missing, "Missing top-level directories:\n" + "\n".join(missing)


def test_no_stale_directory_links_in_live_docs() -> None:
    """Reject Markdown links that point at retired workspace paths."""
    offenders: list[str] = []
    for path in _doc_files():
        if path.name == "ARCHITECTURE_AND_CLEANUP_PLAN.md":
            continue
        if "archive" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for match in STALE_LINK_RE.findall(text):
            offenders.append(f"{path.relative_to(ROOT)}: {match}")
        # Also catch direct relative targets without relying on findall groups.
        for _label, target in MD_LINK_RE.findall(text):
            cleaned = target.split("#", 1)[0]
            if any(
                part in cleaned
                for part in (
                    "super_metroid_rl/",
                    "super_mario_bros/",
                )
            ):
                # Allow glossary/status prose that is not a navigable link to a
                # missing tree — only flag real relative targets.
                if cleaned.startswith(("http://", "https://")):
                    continue
                offenders.append(f"{path.relative_to(ROOT)} -> {target}")
    # Deduplicate
    offenders = sorted(set(offenders))
    assert not offenders, "Stale directory links:\n" + "\n".join(offenders)


def test_active_games_have_required_docs() -> None:
    missing: list[str] = []
    for game in ACTIVE_GAME_DIRS:
        base = ROOT / game
        for rel in ("AGENTS.md", "docs/STATUS.md", "docs/plan.md"):
            path = base / rel
            if not path.is_file():
                missing.append(str(path.relative_to(ROOT)))
    assert not missing, "Missing required game docs:\n" + "\n".join(missing)


def test_status_docs_have_maturity_and_last_verified() -> None:
    missing: list[str] = []
    for game in ACTIVE_GAME_DIRS:
        path = ROOT / game / "docs" / "STATUS.md"
        text = path.read_text(encoding="utf-8")
        if not MATURITY_RE.search(text):
            missing.append(f"{path.relative_to(ROOT)}: maturity")
        if not LAST_VERIFIED_RE.search(text):
            missing.append(f"{path.relative_to(ROOT)}: last verification")
    assert not missing, "STATUS.md missing required fields:\n" + "\n".join(
        missing
    )


def test_manifests_reference_existing_paths(
    manifests: list[dict[str, object]],
) -> None:
    missing: list[str] = []
    for data in manifests:
        directory = data.get("directory")
        if directory:
            if not (ROOT / str(directory)).is_dir():
                missing.append(f"directory {directory}")
        for key in ("status_doc", "plan_doc", "assist_contract", "best_manifest"):
            rel = data.get(key)
            if rel in (None, "null"):
                continue
            path = ROOT / str(rel)
            if not path.exists():
                missing.append(f"{data['game']}.{key}={rel}")
        if str(data.get("maturity")) in {"M7", "M8"}:
            video = data.get("best_video")
            manifest = data.get("best_manifest")
            if not video and not manifest:
                missing.append(
                    f"{data['game']}: continuous clear needs video or manifest"
                )
        intervention = str(data.get("intervention_class", "clean"))
        if intervention != "clean" and "assisted" in intervention:
            assist = data.get("assist_contract")
            if not assist:
                missing.append(f"{data['game']}: assisted result needs assist_contract")
            elif not (ROOT / str(assist)).exists():
                missing.append(f"{data['game']}: missing assist contract file")
    assert not missing, "Manifest path problems:\n" + "\n".join(missing)


def test_game_matrix_is_generated() -> None:
    matrix = ROOT / "docs" / "GAME_MATRIX.md"
    assert matrix.is_file()
    text = matrix.read_text(encoding="utf-8")
    assert "Generated from `docs/manifests/*.yaml`" in text
    assert "tmnt_iv" in text
    assert "zelda_i" in text
    assert "smb" in text
    assert "Ladder rank" not in text
