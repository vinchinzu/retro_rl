-- Power-on → Gravity Suit. One compose function; hop bodies live in packages.

local ceres = require("ceres.spine")
local morph = require("morph.spine")
local bombs = require("bombs.spine")
local brinstar = require("brinstar.spine")
local kraid = require("kraid.spine")
local ice = require("ice.spine")
local ws = require("wrecked_ship.spine")

local M = {}

function M.play_ceres(session)
  ceres.play(session)
end

function M.play_to_morph(session)
  M.play_ceres(session)
  morph.play_ship_to_morph(session)
end

function M.play_to_bombs(session)
  M.play_to_morph(session)
  bombs.play_morph_to_bombs_exit(session)
end

function M.play_to_red_tower(session)
  M.play_to_bombs(session)
  brinstar.play_to_red_tower(session)
end

function M.play_to_business(session)
  M.play_to_red_tower(session)
  kraid.play_to_business(session)
end

function M.play_to_ice(session)
  M.play_to_business(session)
  ice.play_business_to_ice(session)
end

function M.play_to_phantoon(session)
  M.play_to_ice(session)
  ws.play_ice_to_phantoon_leave(session)
end

function M.play_to_gravity(session)
  M.play_to_phantoon(session)
  ws.play_phantoon_to_gravity(session)
  return ws.play_gravity_collect(session)
end

M.play = M.play_to_gravity

return M
