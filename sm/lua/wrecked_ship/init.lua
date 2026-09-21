-- K6 Wrecked Ship package.

local spine = require("wrecked_ship.spine")
local M = {}
M.play_ice_to_phantoon_leave = spine.play_ice_to_phantoon_leave
M.play_phantoon_to_gravity = spine.play_phantoon_to_gravity
M.play_gravity_collect = spine.play_gravity_collect
M.play = spine.play
return M
