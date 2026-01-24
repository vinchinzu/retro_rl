#!/usr/bin/env python3
"""
Reprocess recovered files to extract transitions and update stats.
This is useful after a batch of recordings got saved to recovery.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recording.session import SessionManager

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    session = SessionManager(base_dir)

    # Find all recover-* files
    demos_dir = os.path.join(base_dir, "demos")
    recovered_files = [f for f in os.listdir(demos_dir)
                      if f.startswith("recover-") and f.endswith(".bk2")]

    if not recovered_files:
        print("No recovered files to process.")
        return

    print(f"Found {len(recovered_files)} recovered recording(s)")
    print("=" * 70)

    for filename in recovered_files:
        filepath = os.path.join(demos_dir, filename)
        print(f"\nProcessing: {filename}")

        try:
            # Extract transitions
            extracted = session.extractor.extract_all_transitions(filepath)

            if extracted:
                # Update manifest
                route = [e[0] for e in extracted]
                session.manifest.update_analysis(filepath, route)
                print(f"  ✓ Extracted {len(extracted)} transition(s)")
            else:
                print(f"  ℹ No transitions found (single room run)")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 70)
    print("✓ Processing complete!")
    print(f"\nView results with: ./retro_env/bin/python scripts/analysis/view_stats.py")

if __name__ == "__main__":
    main()
