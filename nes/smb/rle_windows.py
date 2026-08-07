"""Known RLE polish windows for the SMB continuous seed.

Frame indices measured into ``smb_1_1_to_ending``-style continuous seeds.
Shared harness must not hard-code these; import from here when polishing SMB.
"""

from __future__ import annotations

from retro_harness.platformer.rle_optimize import RleWindow

# Measured 2026-07-30 on smb_1_1_to_ending.json with settle=14:
#   stairs wall-slam xs→0 at f≈1164 (x=2962) and f≈1210 (x=2994);
#   flag grab (player_state=4) at f≈1311. Old 1700–1974 window was castle
#   score-tally idle (player_state=5) — no optimizable control.
SMB_BOTTLENECK_WINDOWS: tuple[RleWindow, ...] = (
    RleWindow(1050, 1311, "1-1-stairs", min_progress=2700, max_progress=3200),
    RleWindow(350, 520, "1-1-first-pipe", min_progress=850, max_progress=950),
    # 4-2 occupies roughly frames 6366..9130 in the continuous seed (STATUS)
    RleWindow(6366, 9130, "4-2-full", min_progress=0, max_progress=None),
    # Tighter natural-entry polish region at the start of 4-2
    RleWindow(6366, 7000, "4-2-entry", min_progress=0, max_progress=None),
    # 8-1 body in natural_82 continuous seed (after 219f lead idle from 4-2 exit).
    # Prefer isolated Level8_1 polish via smb.scripts.polish_8_1 — continuous
    # absolute windows need full-route eval or a natural-control checkpoint.
    RleWindow(9181, 12159, "8-1-body", min_progress=0, max_progress=None),
    RleWindow(9181, 9981, "8-1-early", min_progress=0, max_progress=None),
    RleWindow(11581, 12159, "8-1-late", min_progress=2500, max_progress=None),
)
