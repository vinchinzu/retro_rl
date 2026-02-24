#!/usr/bin/env python3
"""
Refactored to use the `recording` package logic.
Can be used to extract states from a specific .bk2 file.
"""

import os
import argparse
import re
from recording.session import SessionManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_name", help="Name of the output state (e.g. Room2)")
    parser.add_argument("--demo", "-d", help="Path to .bk2 recording file")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    session = SessionManager(base_dir)

    input_bk2 = args.demo
    
    # Logic to infer demo if not provided (backward compat)
    if not input_bk2:
         print("No input demo specified. Attempting to infer from Room number...")
         match = re.search(r"(Room)(\d+)", args.output_name, re.IGNORECASE)
         if match:
             current_room_num = int(match.group(2))
             # The recording for Room X comes from playing Room X-1
             prev_room_num = current_room_num - 1
             prev_room_name = f"{match.group(1)}{prev_room_num}"
             
             print(f"Looking for recordings of {prev_room_name} to generate {args.output_name}...")
             
             demo_dir = session.recording_dir
             candidates = [f for f in os.listdir(demo_dir) if f.endswith('.bk2') and prev_room_name in f]
             if candidates:
                 latest_candidate = max([os.path.join(demo_dir, f) for f in candidates], key=os.path.getctime)
                 input_bk2 = latest_candidate
                 print(f"Auto-selected recording: {os.path.basename(input_bk2)}")
             else:
                  print(f"ERROR: No recordings found for {prev_room_name}.")
                  exit(1)
         else:
             print("ERROR: Could not infer previous room from output name.")
             exit(1)

    print(f"extracting states from {input_bk2}...")
    # Map raw extraction to the requested name if it's a single transition
    extracted = session.extractor.extract_all_transitions(input_bk2)
    
    # If the user specifically asked for "Room19" but our auto-extractor named it something else (or "Unknown"),
    # we might want to rename it or confirm.
    # For now, let's trust the auto-extractor and just print results.
    for name, rid, path in extracted:
        print(f"Extracted: {name} (ID: {hex(rid)}) -> {path}")

if __name__ == "__main__":
    main()
