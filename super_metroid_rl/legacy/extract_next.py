#!/usr/bin/env python3
import sys
import os
import re
from extract_state import extract_end_state, DEMO_DIR

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_next.py <RoomName> (e.g. Room17)")
        sys.exit(1)
        
    room_name = sys.argv[1]
    
    # 1. Find latest recording for this room
    files = [f for f in os.listdir(DEMO_DIR) if f.endswith('.bk2') and room_name in f]
    if not files:
        print(f"No recordings found for {room_name}")
        sys.exit(1)
        
    # Sort by modification time
    latest = max([os.path.join(DEMO_DIR, f) for f in files], key=os.path.getmtime)
    print(f"Found latest recording: {os.path.basename(latest)}")
    
    # 2. Determine Next Room Number
    match = re.search(r"(Room)(\d+)", room_name, re.IGNORECASE)
    if not match:
        print("Could not parse room number (e.g. Room17)")
        sys.exit(1)
        
    current_num = int(match.group(2))
    next_num = current_num + 1
    next_room = f"{match.group(1)}{next_num}"
    
    print(f"Extracting state for {next_room}...")
    
    # 3. Extract
    try:
        extract_end_state(latest, next_room)
        print(f"SUCCESS: {next_room}.state created!")
    except Exception as e:
        print(f"ERROR: Extraction failed: {e}")

if __name__ == "__main__":
    main()
