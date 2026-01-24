
import json
import os
import time
import math

ROUTE_ORDER = [
    "Landing Site",
    "Parlor and Alcatraz",
    "Climb",
    "Pit Room",
    "Blue Brinstar Elevator Room",
    "Morph Ball Room",
    "Flyway",
    "Bomb Torizo Room",
]

def _base_room_name(name: str) -> str:
    if " [from " in name:
        return name.split(" [from ", 1)[0]
    return name

def _format_room_name(name: str) -> str:
    if " [from " not in name:
        return name
    base, rest = name.split(" [from ", 1)
    origin = rest.rstrip("]")
    if origin == base:
        return base
    return name

def _route_sort_key(name: str):
    base = _base_room_name(name)
    try:
        idx = ROUTE_ORDER.index(base)
    except ValueError:
        idx = len(ROUTE_ORDER) + 1
    return (idx, base, name)

def main():
    # Point to project root (2 levels up from scripts/analysis/)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stats_path = os.path.join(base_dir, "best_times.json")
    
    if not os.path.exists(stats_path):
        print("No stats found.")
        return

    with open(stats_path, 'r') as f:
        data = json.load(f)

    all_history_events = []
    room_stats = {} # Cache best/avg for lookups

    # 1. Gather Data & Stats
    for room, info in data.items():
        best_sec = info.get("best_seconds", 0)
        history = info.get("history", [])
        attempts = len(history)
        
        avg_sec = 0
        std_dev = 0.0
        
        if attempts > 0:
            secs = [h["seconds"] for h in history]
            avg_sec = sum(secs) / attempts
            if attempts > 1:
                variance = sum((x - avg_sec) ** 2 for x in secs) / (attempts - 1)
                std_dev = math.sqrt(variance)
        
        room_stats[room] = {
            "best": best_sec,
            "avg": avg_sec,
            "std": std_dev,
            "count": attempts
        }
        
        for h in history:
            all_history_events.append({
                "room": room,
                "timestamp": h.get("timestamp", 0),
                "seconds": h.get("seconds", 0)
            })

    # 2. Identify Last Run (Route) - Cluster by Time Gap
    if not all_history_events:
        print("No run history found.")
        return
        
    all_history_events.sort(key=lambda x: x["timestamp"])
    
    runs = []
    current_run = []
    prev_ts = 0
    
    for e in all_history_events:
        ts = e["timestamp"]
        if not current_run:
            current_run.append(e)
        else:
            # Check gap. 
            # Note: timestamps are 'file_mtime + index'. 
            # If runs represent different files, mtime diff will be large (minutes).
            # If same file, diff is 1.
            if (ts - prev_ts) > 60: # 60s gap implies new recording file/session
                runs.append(current_run)
                current_run = [e]
            else:
                current_run.append(e)
        prev_ts = ts
    
    if current_run:
        runs.append(current_run)
        
    # Get the very last run
    final_run = runs[-1]
    
    # Format human readable start time
    start_ts = final_run[0]["timestamp"]
    start_date = time.strftime('%H:%M:%S', time.localtime(start_ts))
    
    print(f"\n{'='*110}")
    print(f" LAST RUN ANALYSIS (Route Order) - Started at {start_date}")
    print(f"{'='*110}")
    print(f" {'Room Name':<35} | {'Seg(Last)':<9} | {'Best':<8} | {'Diff':<8} | {'StDev':<6} | {'Total':<8}")
    print(f"{'-'*110}")
    
    cum_time = 0.0
    
    for e in final_run:
        room = e["room"]
        seg_time = e["seconds"]
        stats = room_stats.get(room, {})
        best = stats.get("best", 0)
        std = stats.get("std", 0)
        
        diff = seg_time - best
        cum_time += seg_time
        
        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
        
        print(f" {room:<35} | {seg_time:.2f}s    | {best:.2f}s   | {diff_str:<8} | {std:.2f}   | {cum_time/60:.2f}m")
        
    print(f"{'='*110}")
    print(f" Total Run Time: {cum_time:.2f}s ({cum_time/60:.2f}m)")
    print(f"{'='*110}\n")

    # 3. Aggregate summary (all rooms)
    print(f"{'='*110}")
    print(" ALL ROOMS SUMMARY (Best/Mean/StDev)")
    print(f"{'='*110}")
    print(f" {'Room Name':<35} | {'Best':<8} | {'Mean':<8} | {'StDev':<6} | {'N':<4}")
    print(f"{'-'*110}")

    for room in sorted(room_stats.keys(), key=_route_sort_key):
        stats = room_stats[room]
        display = _format_room_name(room)
        print(
            f" {display:<35} | {stats['best']:.2f}s   | {stats['avg']:.2f}s   | "
            f"{stats['std']:.2f}   | {stats['count']:<4}"
        )

    print(f"{'='*110}\n")


if __name__ == "__main__":
    main()
