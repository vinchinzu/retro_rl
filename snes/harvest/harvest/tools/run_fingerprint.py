#!/usr/bin/env python3
"""Fingerprint a run_to_day2 --out JSON (+ optional --save-end-state file) for speedup equivalence checks.

Usage: run_fingerprint.py out.json [end.state]; equal json/state hashes = bit-identical outcome (wall_seconds ignored)."""
import gzip, hashlib, json, sys
from pathlib import Path
def fp(js, st=None):
    d = json.loads(Path(js).read_text())
    wall = d.pop("wall_seconds", None); d.pop("progress_sidecar", None)
    h = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
    s = None
    if st and Path(st).exists():
        raw = Path(st).read_bytes()
        try: raw = gzip.decompress(raw)
        except Exception: pass
        s = hashlib.sha256(raw).hexdigest()[:16]
    return dict(frames=d["frames"], wall=wall, fps=round(d["frames"]/wall) if wall else None,
                end=d["end"], json=h, state=s)
if __name__ == "__main__":
    a = sys.argv[1:]
    print(fp(*a[:2]))
