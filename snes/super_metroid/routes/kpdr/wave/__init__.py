"""K4 Wave branch pure controllers — Bubble → Single → Double → Wave.

K4.8 Bubble → Single Chamber (``0xAD5E``): post-Speed return Bubble top-right
→ drop shaft → middle-right blue door into Single left shaft.

K4.9 Single → Double Chamber (``0xADAD``): left-shaft top entry → mid ledge →
floor platform y≈395 → missile red door (Second Top Right) into Double top.

K4.10 Double Chamber → Wave Beam PLM (``0xADDE``): top-left pin → upper path
→ blue gate → right Super/missile door → Wave chozo collect (beam bit 0x0001).

Package layout
--------------
* ``geometry`` — room-prefixed bands + seats + ``WAVE_BEAM_MASK`` / predicates
* ``scripts`` — human gate-open RLE (JSON load is rr-7sn.2)
* ``helpers`` — knockback escape wrappers (stop-room variants)
* ``bubble_to_single`` — K4.8
* ``single_to_double`` — K4.9
* ``double_gate`` — K4.10 Kamer hop + blue gate open
* ``double_to_wave`` — K4.10 Super door + Wave chozo collect

Public API is also re-exported from :mod:`super_metroid.routes.kpdr.k4_wave`
for stable registry / spine_hops / probe imports.
"""

from __future__ import annotations

from super_metroid.routes.kpdr.rooms import (
    ROOM_BUBBLE,
    ROOM_DOUBLE_CHAMBER,
    ROOM_SINGLE_CHAMBER,
    ROOM_WAVE,
)
from super_metroid.routes.kpdr.wave.bubble_to_single import (
    play_bubble_to_single_chamber,
)
from super_metroid.routes.kpdr.wave.double_to_wave import (
    play_double_chamber_to_wave,
)
from super_metroid.routes.kpdr.wave.geometry import WAVE_BEAM_MASK
from super_metroid.routes.kpdr.wave.single_to_double import (
    play_single_to_double_chamber,
)

__all__ = [
    "WAVE_BEAM_MASK",
    "play_bubble_to_single_chamber",
    "play_single_to_double_chamber",
    "play_double_chamber_to_wave",
    "ROOM_BUBBLE",
    "ROOM_SINGLE_CHAMBER",
    "ROOM_DOUBLE_CHAMBER",
    "ROOM_WAVE",
]
