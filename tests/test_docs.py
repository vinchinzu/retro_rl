"""Machine-checkable documentation integrity tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "docs" / "manifests"

# Auto-loaded agent context. Soft targets are ~45–60; these ceilings stop
# the 200+ line encyclopedias from coming back.
ROOT_AGENTS_MAX_LINES = 70
GAME_AGENTS_MAX_LINES = 80

# Top-level docs that participate in link checks (not vendored trees).
DOC_GLOBS = (
    "*.md",
    "docs/*.md",
    "retro_harness/docs/*.md",
    "AGENTS.md",
    "README.md",
)

# Workspace-relative game directories (under snes/ or nes/).
ACTIVE_GAME_DIRS = (
    "snes/alttp",
    "snes/alttp_rando",
    "snes/battle_clash",
    "nes/castlevania",
    "nes/contra",
    "nes/ducktales",
    "snes/f_zero",
    "snes/final_fight",
    "snes/great_waldo_search",
    "snes/joe_and_mac",
    "nes/kirby_adventure",
    "snes/magical_quest",
    "nes/mega_man_2",
    "nes/metroid",
    "snes/pilotwings",
    "nes/punch_out",
    "snes/rival_turf",
    "nes/smb",
    "nes/smb3",
    "snes/sm_rando",
    "snes/smz3",
    "snes/star_fox",
    "snes/super_double_dragon",
    "snes/super_metroid",
    "nes/tmnt_i",
    "nes/tmnt_ii",
    "nes/tmnt_iii",
    "snes/tmnt_iv",
    "nes/zelda_i",
    "nes/zelda_ii",
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


def _known_game_slugs() -> set[str]:
    slugs: set[str] = set()
    for console in ("snes", "nes"):
        base = ROOT / console
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                slugs.add(child.name)
    return slugs


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
    resolved = (source.parent / cleaned).resolve()
    # Probe media under recordings/ is gitignored and machine-local; STATUS
    # cites those paths as evidence labels, not durable schema requirements.
    if _is_local_recording(resolved):
        return None
    return resolved


def _is_local_recording(path: Path) -> bool:
    """True for gitignored ``**/recordings/**`` evidence paths."""
    return "recordings" in path.parts


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
    known_games = _known_game_slugs()
    retired = {
        "super_metroid_rl",
        "super_mario_bros",
        "snes_oneshot",  # folded into retro_harness/
    }
    allowed_future = {"earthbound"}
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
        # Game package names live under snes/ or nes/; backtick form `game/` is OK.
        *known_games,
    }
    program_docs = {
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        *sorted((ROOT / "docs").glob("*.md")),
        ROOT / "retro_harness" / "docs" / "TOOLSET.md",
        ROOT / "retro_harness" / "docs" / "EMULATOR_FEATURES.md",
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
            if key == "best_manifest" and _is_local_recording(path):
                continue
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


def test_agents_files_stay_under_hygiene_ceiling() -> None:
    """AGENTS.md is auto-loaded; keep commands + traps + pointers only."""
    offenders: list[str] = []
    root_agents = ROOT / "AGENTS.md"
    root_lines = len(root_agents.read_text(encoding="utf-8").splitlines())
    if root_lines > ROOT_AGENTS_MAX_LINES:
        offenders.append(f"AGENTS.md: {root_lines} > {ROOT_AGENTS_MAX_LINES}")
    for path in sorted(ROOT.glob("*/AGENTS.md")) + sorted(
        ROOT.glob("*/*/AGENTS.md")
    ):
        lines = len(path.read_text(encoding="utf-8").splitlines())
        if lines > GAME_AGENTS_MAX_LINES:
            offenders.append(
                f"{path.relative_to(ROOT)}: {lines} > {GAME_AGENTS_MAX_LINES}"
            )
    assert not offenders, "AGENTS.md over hygiene ceiling:\n" + "\n".join(
        offenders
    )


def test_game_matrix_is_generated() -> None:
    matrix = ROOT / "docs" / "GAME_MATRIX.md"
    assert matrix.is_file()
    text = matrix.read_text(encoding="utf-8")
    assert "Generated from `docs/manifests/*.yaml`" in text
    assert "tmnt_iv" in text
    assert "zelda_i" in text
    assert "smb" in text
    assert "Ladder rank" not in text


def test_external_route_research_policy_is_shared() -> None:
    process = (ROOT / "docs" / "FULL_RUN_PROCESS.md").read_text(
        encoding="utf-8"
    )
    benchmark = (ROOT / "docs" / "BENCHMARK_SPEC.md").read_text(
        encoding="utf-8"
    )
    assert "External route references are allowed" in process
    assert "approved development accelerators for every game" in process
    assert (
        "Offline route research is also separate from runtime class"
        in benchmark
    )
