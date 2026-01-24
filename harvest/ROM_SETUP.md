# ROM File Organization

## Structure

ROMs are stored in `roms/` directory (gitignored) and symlinked to integration directories:

```
roms/
└── Harvest Moon.sfc          # Actual ROM file (gitignored)

custom_integrations/
└── HarvestMoon-Snes/
    ├── rom.sfc -> /absolute/path/to/roms/Harvest Moon.sfc (symlink)
    ├── rom.sha               # SHA1 hash for verification
    └── ROM_README.md         # Setup instructions
```

## Why This Setup?

1. **Copyright Protection**: ROM files are in gitignored `roms/` directory
2. **Compatibility**: Stable-retro expects ROM at `custom_integrations/HarvestMoon-Snes/rom.sfc`
3. **Single Source**: One ROM file, accessed via symlink

## Verification

Expected SHA1: `a64a5634429a4f5341868a40c220d7be89fda70a`

Verify ROM:
```bash
sha1sum roms/Harvest\ Moon.sfc
```

## Gitignore Settings

Added to `.gitignore`:
- `roms/` directory
- `*.sfc`, `*.smc`, `*.bin` extensions
- `*.md5`, `*.sha` hash files

## Code Requirements

All Python code uses absolute paths for custom integrations:

```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)  # Must be absolute!
```

## Setting Up on New Machine

1. Obtain legal Harvest Moon (SNES) ROM
2. Place as `roms/Harvest Moon.sfc`
3. Symlink should already exist in git
4. If symlink broken, recreate:
   ```bash
   ln -s "$(pwd)/roms/Harvest Moon.sfc" custom_integrations/HarvestMoon-Snes/rom.sfc
   ```
5. Verify: `./run_bot.sh list`
