"""Fetch a pinned vg-json-data/z3-json-data checkout into alttp/refs/.

Never run automatically on import. Developers call this once (or after
changing the pin) to materialize the gitignored local tree used by
``alttp.z3_json_data``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, globals().get('_SNES_IMPORT_ROOT', _REPO_ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from alttp.paths import (  # noqa: E402
    Z3_JSON_DATA_DIR,
    Z3_JSON_DATA_PIN,
    Z3_JSON_DATA_REPO,
)
from alttp.z3_json_data import validate_source_shape  # noqa: E402


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _git_head(dest: Path) -> str:
    out = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=dest,
        text=True,
    )
    return out.strip()


def _revision_matches(head: str, revision: str) -> bool:
    return head == revision or head.startswith(revision) or revision.startswith(head)


def _fetch_and_checkout(worktree: Path, revision: str) -> str:
    """Fetch *revision* into an existing git worktree and check it out."""
    # Prefer a shallow tip fetch (works for commit SHAs on GitHub).
    try:
        _run(["git", "fetch", "--depth", "1", "origin", revision], cwd=worktree)
        _run(["git", "checkout", "--force", "FETCH_HEAD"], cwd=worktree)
        return _git_head(worktree)
    except subprocess.CalledProcessError:
        pass
    # Full fetch of the object (needed for some local remotes / shallow sources).
    _run(["git", "fetch", "origin", revision], cwd=worktree)
    _run(["git", "checkout", "--force", revision], cwd=worktree)
    return _git_head(worktree)


def _clone_pinned(dest: Path, repo: str, revision: str) -> str:
    """Create *dest* checked out at *revision*."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Strategy 1: init + shallow fetch of exact SHA (fast on GitHub).
    dest.mkdir(parents=True)
    try:
        _run(["git", "init"], cwd=dest)
        _run(["git", "remote", "add", "origin", repo], cwd=dest)
        return _fetch_and_checkout(dest, revision)
    except (subprocess.CalledProcessError, OSError) as first_exc:
        print(f"shallow pin fetch failed ({first_exc}); trying full clone")
        if dest.exists():
            shutil.rmtree(dest)

    # Strategy 2: full clone then checkout (most portable).
    _run(["git", "clone", repo, str(dest)])
    _run(["git", "checkout", "--force", revision], cwd=dest)
    return _git_head(dest)


def setup(
    *,
    dest: Path,
    repo: str,
    revision: str,
    force: bool = False,
) -> Path:
    """Clone or update *dest* to *revision*; return the resolved path."""
    dest = dest.expanduser()
    if force and dest.exists():
        print(f"Removing existing checkout: {dest}")
        shutil.rmtree(dest)

    if dest.is_dir() and (dest / ".git").exists():
        head = _fetch_and_checkout(dest, revision)
    elif dest.exists():
        raise RuntimeError(
            f"{dest} exists but is not a git clone; remove it or pass --force"
        )
    else:
        head = _clone_pinned(dest, repo, revision)

    if not _revision_matches(head, revision):
        raise RuntimeError(
            f"checkout HEAD {head} does not match requested revision {revision}"
        )

    print(f"z3-json-data ready at {dest}")
    print(f"revision: {head}")
    print(f"pin:      {revision}")
    validate_source_shape(dest)
    print("shape checks: OK")
    return dest.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Clone/checkout vg-json-data/z3-json-data at a pinned revision "
            f"(default pin {Z3_JSON_DATA_PIN[:12]}…)."
        )
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Z3_JSON_DATA_DIR,
        help=f"checkout directory (default: {Z3_JSON_DATA_DIR})",
    )
    parser.add_argument(
        "--repo",
        default=Z3_JSON_DATA_REPO,
        help=f"git remote URL (default: {Z3_JSON_DATA_REPO})",
    )
    parser.add_argument(
        "--revision",
        default=Z3_JSON_DATA_PIN,
        help=f"commit SHA to pin (default: {Z3_JSON_DATA_PIN})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete existing dest and re-clone",
    )
    args = parser.parse_args(argv)
    try:
        setup(
            dest=args.dest,
            repo=args.repo,
            revision=args.revision,
            force=args.force,
        )
    except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
        print(f"setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
