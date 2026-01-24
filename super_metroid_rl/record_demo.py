#!/usr/bin/env python3
"""
Record human demonstrations for imitation learning.
Refactored to use the `recording` package.

Usage:
    ../retro_env/bin/python record_demo.py --state Room11
"""

import os
import argparse
from recording.session import SessionManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="Landing Site", help="Starting State")
    parser.add_argument("--loop", action="store_true", default=True, help="Continuously load the next room after extraction.")
    parser.add_argument("--no-loop", action="store_false", dest="loop", help="Exit after one recording session.")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    session = SessionManager(base_dir)
    try:
        session.run_loop_mode(args.state, loop_forever=args.loop)
    except KeyboardInterrupt:
        print("\nExiting...")
        session.recorder.close()

if __name__ == "__main__":
    main()
