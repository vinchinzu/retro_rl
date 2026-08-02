#!/usr/bin/env python3
"""Multi-boss natural-entry capture CLI (one shared entry, not per-boss sprawl).

Development infrastructure only — not continuous evidence. Capture records
room + pose + door settle without progression / boss-bit forges.

```bash
# Catalog + requirements
uv run python super_metroid/scripts/probe/natural_entry_cli.py list
uv run python super_metroid/scripts/probe/natural_entry_cli.py describe phantoon

# Bomb Torizo: continuous power-on prefix (slow)
uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural bomb_torizo

# Non-BT bosses: settle capture from a doorway / predecessor save
uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural \\
  kraid --from-state entry --mode room_entry
uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural \\
  phantoon --from-state path/to/phantoon_entry.state --mode room_entry
uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural \\
  botwoon --from-state path/to/botwoon_entry.state --mode room_entry

# Plan only (no emulator)
uv run python super_metroid/scripts/probe/natural_entry_cli.py capture-natural \\
  phantoon --plan-only
```

Bomb Torizo back-compat remains on ``bomb_torizo_combat.py capture-natural``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from super_metroid.combat.natural_entry import cli_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(cli_main())
