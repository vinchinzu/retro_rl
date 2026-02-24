"""Allow running as: python -m donkey_kong_country.optimizer

Thin wrapper that delegates to platformer_common with DKC defaults.
Old invocations still work: python -m donkey_kong_country.optimizer selftest
"""

import platformer_common.levels.dkc  # noqa: F401 - register DKC levels
from platformer_common.runner import main

main(default_level="dkc_winkys_walkway")
