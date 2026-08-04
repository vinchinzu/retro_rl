"""Allow running as: python -m donkey_kong_country.optimizer

Thin wrapper that delegates to retro_harness.platformer with DKC defaults.
Old invocations still work: python -m donkey_kong_country.optimizer selftest
"""

import retro_harness.platformer.levels.dkc  # noqa: F401 - register DKC levels
from retro_harness.platformer.runner import main

main(default_level="dkc_winkys_walkway")
