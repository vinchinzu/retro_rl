"""Ratchet against the suite's dominant growth failure mode.

Measured this session: a mutation campaign flipped nine semantic combat/hunt
constants (``THREAT_RADIUS`` 40->56, ``SWORD_REACH``, ``BEAM_REACH``, ...)
and each failed at most 0-2 of the 1666 tests in this package. A *cosmetic*
rename of a ``FrameAction.reason`` string with zero behaviour change failed
3-6 tests. The suite grew from 63 to 124 files mostly by adding tests whose
only assertion is ``act.reason == "some_string"`` — a log line, not a
behaviour.

This file is pure ``ast`` inspection: no game modules are imported, so it is
cheap and cannot itself rot into an integration test.

Two offender families, each ratcheted the same way (current-set subset of a
checked-in baseline, so tests deleted elsewhere only shrink the set and never
turn this red):

* ``reason_only`` -- every assert in the test touches only ``.reason`` or
  ``.notes``.
* ``meta_only`` -- every assert touches only report()-bookkeeping keys
  (route_eligible, writes, evidence, natural_entry, spec_id) or isinstance().

New offenders in either family fail loudly, by name, with the fix spelled
out. See ``reason_only_baseline.txt`` for the baseline lists and format.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

TESTS_DIR = Path(__file__).parent
BASELINE_PATH = TESTS_DIR / "reason_only_baseline.txt"

REASON_RE = re.compile(r"\.reason\b|\.notes\b")
META_KEYS = ("route_eligible", "writes", "evidence", "natural_entry", "spec_id")

REASON_SECTION = "# reason_only"
META_SECTION = "# meta_only"


def _assert_text(src: str, node: ast.Assert) -> str:
    """Source text of an assert's condition (and message, if any)."""
    parts = [ast.get_source_segment(src, node.test) or ""]
    if node.msg is not None:
        parts.append(ast.get_source_segment(src, node.msg) or "")
    return " ".join(parts)


def _is_meta_text(text: str) -> bool:
    if REASON_RE.search(text):
        return False
    if "isinstance(" in text:
        return True
    return any(key in text for key in META_KEYS)


def classify_test(src: str, func: ast.FunctionDef) -> str | None:
    """Classify a ``test_*`` function as 'reason_only', 'meta_only', or None.

    None means it has a substantive assertion (or no assertion at all --
    that's a different, real problem, but not this ratchet's job).
    """
    asserts = [n for n in ast.walk(func) if isinstance(n, ast.Assert)]
    if not asserts:
        return None
    texts = [_assert_text(src, a) for a in asserts]
    if all(REASON_RE.search(t) for t in texts):
        return "reason_only"
    if all(_is_meta_text(t) for t in texts):
        return "meta_only"
    return None


def scan_tests_dir(tests_dir: Path) -> tuple[set[str], set[str]]:
    """Return (reason_only, meta_only) sets of 'file.py::test_name'.

    Non-recursive: only direct ``test_*.py`` children of ``tests_dir``. This
    deliberately excludes ``tests/rom/**`` (owned/edited by a different lane)
    and any nested fixture packages.
    """
    reason_only: set[str] = set()
    meta_only: set[str] = set()
    for path in sorted(tests_dir.glob("test_*.py")):
        src = path.read_text()
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                kind = classify_test(src, node)
                if kind == "reason_only":
                    reason_only.add(f"{path.name}::{node.name}")
                elif kind == "meta_only":
                    meta_only.add(f"{path.name}::{node.name}")
    return reason_only, meta_only


def load_baseline(path: Path) -> tuple[set[str], set[str]]:
    reason_only: set[str] = set()
    meta_only: set[str] = set()
    section: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == REASON_SECTION or line == META_SECTION:
            section = line
            continue
        if line.startswith("#"):
            continue  # a standalone justification/comment line
        entry = line.split("#", 1)[0].strip()  # allow "entry  # justification"
        if not entry:
            continue
        if section == REASON_SECTION:
            reason_only.add(entry)
        elif section == META_SECTION:
            meta_only.add(entry)
    return reason_only, meta_only


_FIX_MESSAGE = """
{count} new {family} test(s) not in the baseline ({baseline_count} entries in
{baseline_path}):
{offenders}

A `.reason` / `.notes` string (or a bare report()-bookkeeping key /
isinstance check) is a log line, not a behaviour assertion. It will not
notice a real regression -- see the module docstring in
test_assertion_quality.py for the measured mutation-campaign numbers.

Two legitimate fixes:
  1. Add a substantive assertion alongside the string check: assert on
     Link's position, a RAM/snapshot field, or the action's byte-level
     effect (buttons pressed, item used, etc).
  2. If the string genuinely IS the contract being tested (e.g. asserting
     precedence/ordering where the reason tag is the only observable), add
     the test to the "{section}" section of {baseline_path} with a one-line
     `# justification` comment explaining why no stronger assertion applies.

Do not silently widen this ratchet by deleting the assertion instead.
""".strip()


def _format_offenders(names: set[str]) -> str:
    return "\n".join(f"  - {n}" for n in sorted(names))


def test_no_new_reason_only_tests():
    baseline_reason_only, _ = load_baseline(BASELINE_PATH)
    current_reason_only, _ = scan_tests_dir(TESTS_DIR)

    new_offenders = current_reason_only - baseline_reason_only
    assert not new_offenders, _FIX_MESSAGE.format(
        count=len(new_offenders),
        family="reason-only",
        baseline_count=len(baseline_reason_only),
        baseline_path=BASELINE_PATH.name,
        offenders=_format_offenders(new_offenders),
        section=REASON_SECTION,
    )


def test_no_new_metadata_only_tests():
    _, baseline_meta_only = load_baseline(BASELINE_PATH)
    _, current_meta_only = scan_tests_dir(TESTS_DIR)

    new_offenders = current_meta_only - baseline_meta_only
    assert not new_offenders, _FIX_MESSAGE.format(
        count=len(new_offenders),
        family="metadata-only",
        baseline_count=len(baseline_meta_only),
        baseline_path=BASELINE_PATH.name,
        offenders=_format_offenders(new_offenders),
        section=META_SECTION,
    )


def test_baseline_file_is_well_formed():
    """The baseline itself must parse, and every entry must look like
    'file.py::test_name' so a typo doesn't silently disable the ratchet."""
    reason_only, meta_only = load_baseline(BASELINE_PATH)
    assert reason_only, "reason_only baseline section is empty or missing"
    assert meta_only, "meta_only baseline section is empty or missing"
    entry_re = re.compile(r"^test_[A-Za-z0-9_]+\.py::test_[A-Za-z0-9_]+$")
    for entry in reason_only | meta_only:
        assert entry_re.match(entry), f"malformed baseline entry: {entry!r}"
