"""Controller-only KPDR segments through Hi-Jump and Kraid entry.

Implementation lives in :mod:`super_metroid.routes.kpdr` (segment modules +
registry). This module re-exports the historical public surface.
"""

from __future__ import annotations

from super_metroid.routes.kpdr import *  # noqa: F403
from super_metroid.routes.kpdr import __all__ as __all__  # noqa: F401
