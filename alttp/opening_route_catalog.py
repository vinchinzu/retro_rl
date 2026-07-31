"""Compat shim — prefer ``alttp.opening_route.catalog``."""

from __future__ import annotations

from alttp.opening_route.catalog import *  # noqa: F403
from alttp.opening_route.catalog import main


if __name__ == "__main__":
    raise SystemExit(main())
