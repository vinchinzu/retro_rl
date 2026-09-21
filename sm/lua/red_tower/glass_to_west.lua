-- Glass Tunnel → West Tunnel (K5 hop 7). Reverse of west→glass.

local rooms = require("rooms")
local ctrl = require("red_tower.ctrl")
local geom = require("red_tower.geometry")

local M = {}
local ROOM_GLASS = rooms.ROOM_GLASS or 0xCEFB
local ROOM_WEST_TUNNEL = rooms.ROOM_WEST_TUNNEL or 0xCF54

function M.play_glass_to_west(session)
  return ctrl.play_run_shoot_exit(session, {
    from_room = ROOM_GLASS,
    to_room = ROOM_WEST_TUNNEL,
    direction = "LEFT",
    label = "glass_to_west",
    run_frames = geom.GLASS_TO_WEST_RUN,
    shoot_frames = geom.GLASS_TO_WEST_SHOOT,
    spin_frames = geom.GLASS_TO_WEST_SPIN,
    hold_frames = geom.GLASS_TO_WEST_HOLD,
    settle_frames = geom.GLASS_TO_WEST_SETTLE,
  })
end

return M
