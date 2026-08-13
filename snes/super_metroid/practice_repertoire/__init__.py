"""Practice-hack preset repertoire — shared spine for human + bot work.

The Super Metroid practice ROM (tewtal/sm_practice_hack) category menus are
the **repertoire**: a community-standard ordered list of route sessions
(presets). This package turns that list into the shared index for:

1. **Human practice** — ``./play`` pins, demos, multi-take series
2. **Room policy tune/upgrade** — one session → hop_key + reactive policy paths
3. **Policy graduation** — draft → candidate → verified_live_anchor → product
4. **Route edges** — ordered KPDR seams (pin → hop body → next pin)
5. **Autopilot recovery** — live room+items → nearest repertoire reseed pin

Catalog: ``maps/practice_repertoire.json`` (regenerate with
``scripts/export/practice_repertoire.py``).

```bash
uv run python -m super_metroid.practice_repertoire --route
uv run python -m super_metroid.practice_repertoire --policy-board
uv run python -m super_metroid.practice_repertoire --stitch kpdr25/crateria/morph
uv run python -m super_metroid.practice_repertoire --recovery 0x9E9F --items 0x0004
```
"""

from __future__ import annotations

from super_metroid.practice_repertoire.board import (
    PolicyBoardCard,
    gap_report,
    graduation_status,
    iter_product_sessions,
    mapped_sessions,
    policy_board,
    policy_board_card,
    session_work_card,
)
from super_metroid.practice_repertoire.catalog import (
    GRADES,
    PRODUCT_CATEGORY,
    PRODUCT_ROUTE_ID,
    REACTIVE_PLAN_DIR,
    REACTIVE_POLICY_DIR,
    RepertoireSession,
    categories,
    get_session,
    load_catalog,
    neighbors,
    route_sessions,
    sessions,
)
from super_metroid.practice_repertoire.cli import main
from super_metroid.practice_repertoire.spine import (
    PRODUCT_SESSION_MAP,
    RecoveryHint,
    RouteEdge,
    hop_key_for_session,
    product_route_edges,
    recover_session,
    recovery_hint_for_state,
    route_edge,
)

# Thin back-compat aliases (one release). Prefer RouteEdge / route_edge / product_route_edges.
StitchSeam = RouteEdge
stitch_seam = route_edge
product_stitch_board = product_route_edges

__all__ = [
    "GRADES",
    "PRODUCT_CATEGORY",
    "PRODUCT_ROUTE_ID",
    "PRODUCT_SESSION_MAP",
    "REACTIVE_PLAN_DIR",
    "REACTIVE_POLICY_DIR",
    "PolicyBoardCard",
    "RecoveryHint",
    "RepertoireSession",
    "RouteEdge",
    "StitchSeam",
    "categories",
    "gap_report",
    "get_session",
    "graduation_status",
    "hop_key_for_session",
    "iter_product_sessions",
    "load_catalog",
    "main",
    "mapped_sessions",
    "neighbors",
    "policy_board",
    "policy_board_card",
    "product_route_edges",
    "product_stitch_board",
    "recover_session",
    "recovery_hint_for_state",
    "route_edge",
    "route_sessions",
    "session_work_card",
    "sessions",
    "stitch_seam",
]
