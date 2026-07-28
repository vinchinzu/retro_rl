#!/usr/bin/env python3
"""Generate GAME_MATRIX.md from docs/manifests/*.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
OUTPUT = Path(__file__).resolve().parent / "GAME_MATRIX.md"

REQUIRED_KEYS = (
    "game",
    "title",
    "genre_tracks",
    "capability_phase",
    "project_state",
    "maturity",
    "runtime_class",
    "intervention_class",
    "blocker",
)


def load_manifests() -> list[dict[str, object]]:
    manifests: list[dict[str, object]] = []
    for path in sorted(MANIFEST_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected mapping")
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"{path}: missing keys {missing}")
        data["_path"] = str(path.relative_to(ROOT))
        manifests.append(data)
    return manifests


def _cell(value: object) -> str:
    if value is None:
        return "—"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def render(manifests: list[dict[str, object]]) -> str:
    lines: list[str] = [
        "# Game Matrix",
        "",
        "Generated from `docs/manifests/*.yaml`. Do not hand-edit the tables;",
        "edit the manifests and run:",
        "",
        "```bash",
        "uv run python docs/generate_game_matrix.py",
        "```",
        "",
        "Games are **parallel genre capability tracks**, not a single ranked",
        "ladder. Maturity uses M0–M8; runtime and intervention classes are",
        "independent. See [ROADMAP.md](ROADMAP.md),",
        "[DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md), and",
        "[BENCHMARK_SPEC.md](BENCHMARK_SPEC.md).",
        "",
        f"Manifest count: **{len(manifests)}**.",
        "",
        "## Active and scaffolded workspaces",
        "",
        "| Game | Genre track | Phase | State | Maturity | Runtime | Intervention | Full run | Blocker |",
        "| ---- | ----------- | ----- | ----- | -------- | ------- | ------------ | -------- | ------- |",
    ]

    in_repo = [m for m in manifests if m.get("directory")]
    planned = [m for m in manifests if not m.get("directory")]

    for m in sorted(
        in_repo,
        key=lambda item: (
            int(item["capability_phase"]),  # type: ignore[arg-type]
            str(item["game"]),
        ),
    ):
        tracks = ", ".join(str(t) for t in m["genre_tracks"])  # type: ignore[arg-type]
        full_run = "yes" if str(m["maturity"]) in {"M7", "M8"} else "no"
        directory = _cell(m.get("directory"))
        title = _cell(m["title"])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{directory}` / {title}",
                    _cell(tracks),
                    f"P{_cell(m['capability_phase'])}",
                    _cell(m["project_state"]),
                    _cell(m["maturity"]),
                    _cell(m["runtime_class"]),
                    _cell(m["intervention_class"]),
                    full_run,
                    _cell(m["blocker"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Planned / external",
            "",
            "| Game | Genre track | Phase | State | Maturity | Blocker |",
            "| ---- | ----------- | ----- | ----- | -------- | ------- |",
        ]
    )
    for m in planned:
        tracks = ", ".join(str(t) for t in m["genre_tracks"])  # type: ignore[arg-type]
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(m["title"]),
                    _cell(tracks),
                    f"P{_cell(m['capability_phase'])}",
                    _cell(m["project_state"]),
                    _cell(m["maturity"]),
                    _cell(m["blocker"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Scoring fields (in manifests)",
            "",
            "Each manifest may also carry `popularity`, `engineering_effort`,",
            "`transfer_value`, `ending_definition`, evidence paths, and",
            "`last_verified`. Those feed [PROGRAM_STATUS.md](PROGRAM_STATUS.md)",
            "and local status docs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    output = render(load_manifests())
    OUTPUT.write_text(output, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
