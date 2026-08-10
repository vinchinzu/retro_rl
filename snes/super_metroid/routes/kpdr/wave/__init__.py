"""K4 Wave branch pure controllers — Bubble → Single → Double → Wave (+ return).

K4.8 Bubble → Single Chamber (``0xAD5E``): post-Speed return Bubble top-right
→ drop shaft → middle-right blue door into Single left shaft.

K4.9 Single → Double Chamber (``0xADAD``): left-shaft top entry → mid ledge →
floor platform y≈395 → missile red door (Second Top Right) into Double top.

K4.10 Double Chamber → Wave Beam PLM (``0xADDE``): top-left pin → upper path
→ blue gate → right Super/missile door → Wave chozo collect (beam bit 0x0001).

Wave return (rr-vqv3 stack, Phase B reverse):

* ``wave_to_double`` — Wave → Double (rr-pd0i)
* ``double_to_single`` — Double → Single return (rr-qpkd)
* ``single_to_bubble`` — Single → Bubble return (rr-u0y8)
* ``bubble_to_farm`` — Bubble → Upper Norfair Farm (rr-czg9)
* ``farm_to_speedway`` — Farm → Frog Speedway (rr-z13h; needs Speed)

Package layout
--------------
* ``geometry`` — room-prefixed bands + seats + ``WAVE_BEAM_MASK`` / predicates
* ``scripts`` — human gate-open RLE loaded from ``data/*.json``
* shared ``escape_kb`` — :func:`super_metroid.routes.skills.knockback.escape_kb`
* ``bubble_to_single`` — K4.8
* ``single_to_double`` — K4.9
* ``double_gate`` — K4.10 Kamer hop + blue gate open
* ``double_to_wave`` — K4.10 Super door + Wave chozo collect
* ``wave_to_double`` — Wave return first hop (unblock Ice continuous prefix)
* ``double_to_single`` — Double → Single return (Wave return stack hop 2)
* ``single_to_bubble`` — Single → Bubble return (Wave return stack hop 3)
* ``bubble_to_farm`` — Bubble → Farm return (Wave return stack hop 4)
* ``farm_to_speedway`` — Farm → Frog Speedway return (Wave return stack hop 5)

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
from super_metroid.routes.kpdr.wave.bubble_to_farm import (
    play_bubble_to_farm,
)
from super_metroid.routes.kpdr.wave.bubble_to_single import (
    play_bubble_to_single_chamber,
)
from super_metroid.routes.kpdr.wave.double_to_single import (
    play_double_to_single_chamber,
)
from super_metroid.routes.kpdr.wave.double_to_wave import (
    play_double_chamber_to_wave,
)
from super_metroid.routes.kpdr.wave.farm_to_speedway import (
    play_farm_to_speedway,
)
from super_metroid.routes.kpdr.wave.geometry import WAVE_BEAM_MASK
from super_metroid.routes.kpdr.wave.single_to_bubble import (
    play_single_to_bubble,
)
from super_metroid.routes.kpdr.wave.single_to_double import (
    play_single_to_double_chamber,
)
from super_metroid.routes.kpdr.wave.wave_to_double import (
    play_wave_to_double_chamber,
)

__all__ = [
    "WAVE_BEAM_MASK",
    "play_bubble_to_single_chamber",
    "play_single_to_double_chamber",
    "play_double_chamber_to_wave",
    "play_wave_to_double_chamber",
    "play_double_to_single_chamber",
    "play_single_to_bubble",
    "play_bubble_to_farm",
    "play_farm_to_speedway",
    "ROOM_BUBBLE",
    "ROOM_SINGLE_CHAMBER",
    "ROOM_DOUBLE_CHAMBER",
    "ROOM_WAVE",
]
