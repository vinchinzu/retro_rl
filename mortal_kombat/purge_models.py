#!/usr/bin/env python3
"""
Purge obsolete models, keeping only useful ones for experiments.

Keeps:
  - mk1_multichar_ppo_2000000_steps.zip (original base model)
  - mk1_multichar_ppo_final.zip (same, kept for scripts that reference it)
  - mk1_match4_ppo_final.zip (best old model, used for state extraction)
  - mk1_fresh_ppo_* (active training run)

Deletes everything else and cleans up the model registry.

Usage:
    python purge_models.py          # Dry run (show what would be deleted)
    python purge_models.py --apply  # Actually delete
"""

import argparse
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = SCRIPT_DIR / "models"
REGISTRY_FILE = MODEL_DIR / "registry.json"

# Exact filenames to keep
KEEP_EXACT = {
    "mk1_multichar_ppo_2000000_steps.zip",
    "mk1_multichar_ppo_final.zip",
    "mk1_match4_ppo_final.zip",
}

# Prefixes to keep (all checkpoints)
KEEP_PREFIXES = [
    "mk1_fresh_ppo_",
]

# Registry entries to keep
KEEP_REGISTRY = {
    "mk1_multichar_ppo_2000000_steps.zip",
    "mk1_match4_ppo_final.zip",
}


def should_keep(filename):
    if filename in KEEP_EXACT:
        return True
    for prefix in KEEP_PREFIXES:
        if filename.startswith(prefix):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Purge obsolete models")
    parser.add_argument("--apply", action="store_true", help="Actually delete (default: dry run)")
    args = parser.parse_args()

    all_zips = sorted(MODEL_DIR.glob("mk1_*.zip"))

    keep = []
    delete = []
    for f in all_zips:
        if should_keep(f.name):
            keep.append(f)
        else:
            delete.append(f)

    total_delete_bytes = sum(f.stat().st_size for f in delete)
    total_keep_bytes = sum(f.stat().st_size for f in keep)

    print("=" * 60)
    print("MODEL PURGE" + ("  [DRY RUN]" if not args.apply else "  [APPLYING]"))
    print("=" * 60)

    print(f"\nKEEPING ({len(keep)} models, {total_keep_bytes / 1e6:.0f} MB):")
    for f in keep:
        print(f"  + {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")

    print(f"\nDELETING ({len(delete)} models, {total_delete_bytes / 1e6:.0f} MB):")
    # Group by prefix for cleaner output
    prefixes = {}
    for f in delete:
        parts = f.stem.split("_ppo_")
        prefix = parts[0] + "_ppo" if len(parts) > 1 else f.stem
        prefixes.setdefault(prefix, []).append(f)

    for prefix in sorted(prefixes.keys()):
        files = prefixes[prefix]
        size = sum(f.stat().st_size for f in files)
        names = [f.name for f in files]
        if len(names) <= 3:
            print(f"  - {prefix}_* ({len(files)} files, {size / 1e6:.0f} MB): {', '.join(names)}")
        else:
            print(f"  - {prefix}_* ({len(files)} files, {size / 1e6:.0f} MB)")

    print(f"\nSummary: Delete {len(delete)} models, free {total_delete_bytes / 1e6:.0f} MB")
    print(f"         Keep {len(keep)} models, using {total_keep_bytes / 1e6:.0f} MB")

    if not args.apply:
        print("\nDry run complete. Use --apply to delete.")
        return

    # Delete files
    for f in delete:
        f.unlink()
        print(f"  Deleted {f.name}")

    # Clean up registry
    if REGISTRY_FILE.exists():
        data = json.loads(REGISTRY_FILE.read_text())
        removed = []
        kept = []
        for model_name in list(data["models"].keys()):
            if model_name in KEEP_REGISTRY:
                kept.append(model_name)
            else:
                del data["models"][model_name]
                removed.append(model_name)

        REGISTRY_FILE.write_text(json.dumps(data, indent=2) + "\n")
        print(f"\nRegistry: kept {len(kept)}, removed {len(removed)} entries")
        for name in removed:
            print(f"  - {name}")

    print(f"\nDone! Freed {total_delete_bytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
