# SuperMetroid-Snes integration

- Integration metadata (`data.json`, `scenario.json`, …) lives here.
- **Named anchor save-states** (natural entry, room clears, route fixtures,
  documented working points) live at this directory root.
- **Ephemeral probe states** go under `scratch/` (gitignored with `*.state`).

Do not commit `.state` files. Capture natural-entry states for acceptance;
use door-warp / place only for development.
