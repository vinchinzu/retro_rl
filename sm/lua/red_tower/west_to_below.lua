-- West Tunnel → Below Spazer (K5 hop 8). Reverse of below floor→west.

local rooms = require("rooms")
local ctrl = require("red_tower.ctrl")
local geom = require("red_tower.geometry")

local M = {}
local ROOM_WEST_TUNNEL = rooms.ROOM_WEST_TUNNEL or 0xCF54
local ROOM_BELOW_SPAZER = rooms.ROOM_BELOW_SPAZER or 0xA408

function M.play_west_to_below(session)
  return ctrl.play_run_shoot_exit(session, {
    from_room = ROOM_WEST_TUNNEL,
    to_room = ROOM_BELOW_SPAZER,
    direction = "LEFT",
    label = "west_to_below",
    run_frames = geom.WEST_TO_BELOW_RUN,
    shoot_frames = geom.WEST_TO_BELOW_SHOOT,
    spin_frames = geom.WEST_TO_BELOW_SPIN,
    hold_frames = geom.WEST_TO_BELOW_HOLD,
    settle_frames = geom.WEST_TO_BELOW_SETTLE,
  })
end

return M
