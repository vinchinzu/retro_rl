"""Skip markers for tests whose fixtures are built on the legacy nine-ring sites or the south pasture.

The layout is a process-wide choice (``HARVEST_RING_LAYOUT``, ``HARVEST_PASTURE_ZONE``); both layouts are
exercised by running the suite once with each setting.
"""

import pytest

from harvest.maps.farm_pond import ring_layout_name
from harvest.tasks.grass_planter import pasture_zone_name

legacy_rings_only = pytest.mark.skipif(
    ring_layout_name() != "legacy", reason="fixture is built on the legacy nine-ring sites"
)
south_pasture_only = pytest.mark.skipif(
    pasture_zone_name() != "south", reason="fixture uses south-pasture tile coordinates"
)
