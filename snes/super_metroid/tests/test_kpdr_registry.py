"""Registry: hop ids are present and callable — not module-identity locks."""

from __future__ import annotations

from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS, get_segment
from super_metroid.routes.kpdr.spine import POST_SUPERS_SPINE


def test_registry_and_spine_hops_are_callable() -> None:
    for hop_id, fn in KPDR_SEGMENTS.items():
        assert callable(fn), hop_id
        assert callable(get_segment(hop_id))
    for hop in POST_SUPERS_SPINE:
        assert callable(hop.play), hop.hop_id
