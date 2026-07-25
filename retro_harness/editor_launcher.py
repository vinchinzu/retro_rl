"""Launch registered game editors from one shared entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from retro_harness.editor.bridge_client import bridge_python_command
from retro_harness.editor_registry import get_editor_project, registered_editor_projects


def _format_project_list() -> str:
    lines = ["Registered editors:"]
    for project in registered_editor_projects():
        lines.append(f"  {project.project_id:12}  {project.display_name}")
        if project.description:
            lines.append(f"               {project.description}")
    return "\n".join(lines)


def _launch_editor(project_id: str, editor_args: list[str]) -> int:
    project = get_editor_project(project_id)
    command = [
        *bridge_python_command(project_root=project.project_root),
        "-m",
        project.editor_module,
        *editor_args,
    ]
    completed = subprocess.run(command, cwd=str(project.project_root))
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch a registered retro_rl game editor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_format_project_list(),
    )
    parser.add_argument(
        "project",
        nargs="?",
        help="Editor project id (for example: harvest)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered editor projects and exit",
    )
    parser.add_argument(
        "editor_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected editor module",
    )
    args = parser.parse_args(argv)

    if args.list or args.project is None:
        print(_format_project_list())
        if args.project is None and not args.list:
            return 2
        return 0

    editor_args = list(args.editor_args)
    if editor_args and editor_args[0] == "--":
        editor_args = editor_args[1:]
    return _launch_editor(args.project, editor_args)


if __name__ == "__main__":
    raise SystemExit(main())
