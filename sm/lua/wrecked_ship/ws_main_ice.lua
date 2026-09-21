-- Powered Main Shaft Ice overlay (Atomics / Coverns).

local enemies_mod = require("enemies")
local ice = require("wrecked_ship.ws_basement_ice")

local M = {}
M.SHELF_HOLE_FRAMES = 40
M.SLOPE_523_ICE_X = 1084

function M.ice_keepaway_action(samus_x, samus_y, facing, enemies, opts)
  return ice.ice_keepaway_action(samus_x, samus_y, facing, enemies, opts)
end

function M.shelf_covern_ice_action(samus_x, samus_y, facing, enemies, opts)
  return ice.ice_keepaway_action(samus_x, samus_y, facing, enemies, opts)
end

return M
