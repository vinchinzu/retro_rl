"""Opening-route catalog: checkpoint data, Z3 validation, and artifacts.

Split modules:
  * ``opening_route_data`` — checkpoint dataclasses and route data
  * ``opening_route_validate`` — validation / boot correlation / emit
  * ``scripts/opening_route_catalog.py`` — argparse CLI

This module re-exports the historical public surface.
"""

from __future__ import annotations

from alttp.opening_route_data import (
    CATALOG_KIND,
    CATALOG_VERSION,
    DEFAULT_ARTIFACT,
    DISCLAIMER,
    OVERWORLD_SCREEN_PATH,
    ExpectedConnection,
    ExpectedNode,
    OpeningCheckpoint,
    opening_checkpoints,
    opening_overworld_route_graph,
    opening_overworld_route_legs,
)
from alttp.opening_route_validate import (
    CatalogValidation,
    CheckResult,
    build_catalog_artifact,
    correlate_boot_report,
    load_and_validate,
    validate_against_z3,
    write_artifact,
)

__all__ = [
    "CATALOG_KIND",
    "CATALOG_VERSION",
    "CatalogValidation",
    "CheckResult",
    "DEFAULT_ARTIFACT",
    "DISCLAIMER",
    "ExpectedConnection",
    "ExpectedNode",
    "OVERWORLD_SCREEN_PATH",
    "OpeningCheckpoint",
    "build_catalog_artifact",
    "correlate_boot_report",
    "load_and_validate",
    "opening_checkpoints",
    "opening_overworld_route_graph",
    "opening_overworld_route_legs",
    "validate_against_z3",
    "write_artifact",
]


def main(argv=None) -> int:
    """Delegate to the scripts CLI entry point."""
    from alttp.scripts.opening_route_catalog import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
