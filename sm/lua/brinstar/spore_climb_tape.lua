-- Hash-pinned map-guided climb bodies from kpdr.brinstar.spore_spawn.
-- Each span is held 16 frames (same as the Python controller).
-- SNES-12 names: B Y select start up down left right A X L R.

local LJ = {"LEFT", "A", "B", "X"}
local RJ = {"RIGHT", "A", "B", "X"}
local RR = {"RIGHT", "B", "X"}
local LR = {"LEFT", "B", "X"}
local J = {"A", "B", "X"}
local IDLE = {}

return {
  hold_frames = 16,
  big_pink_climb = {
    LJ, RJ, RJ, LJ, RJ, RJ, LJ, RJ, RJ, RR, RJ, IDLE, IDLE, LJ,
    RJ, RJ, RJ, RJ, RJ, IDLE, RJ, RJ, RJ, RJ, IDLE, J, RJ, RJ,
    RJ, RJ, LJ, RR, J, LJ, LJ, LJ, LJ, LJ, IDLE, LJ, LJ, J,
    LJ, LJ, RJ, LJ, LJ, RJ, LJ, LJ, LJ, RJ, LJ, LJ, RJ,
    LJ, LJ, J, RJ, RR, RR, RJ, RJ, RJ, RJ, LJ,
  },
  spore_exit_climb = {
    LR, LR, RJ, LJ, RR, LJ, LJ, IDLE, RJ, RJ, RJ, RJ, RJ, LJ,
    LJ, RJ, LR, LJ, LJ, LJ, RJ, LJ, LR, LR, RR, RJ, RJ,
    RR, RR, LR, LJ, LJ, LJ, RJ, LJ, J, RJ, LJ, LR, RJ,
    RJ, RJ, RJ, RJ, J, RJ, IDLE, LJ, IDLE, J, LJ,
  },
}
