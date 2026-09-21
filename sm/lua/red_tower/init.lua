-- Red Tower: outbound Kraid path (K2) + K5 reverse tunnels / Ice / Alpha PB.

local stack = require("red_tower.red_stack")
local west = require("red_tower.below_spazer_west")
local warehouse = require("red_tower.warehouse_to_east")
local east = require("red_tower.east_to_glass")
local glass = require("red_tower.glass_to_west")
local west_rev = require("red_tower.west_to_below")
local below = require("red_tower.below_to_bat")
local bat = require("red_tower.bat_to_red")
local hellway = require("red_tower.red_to_hellway")
local cat = require("red_tower.hellway_to_caterpillar")
local alpha = require("red_tower.caterpillar_to_alpha_pb")

local M = {}
for k, v in pairs(stack) do
  M[k] = v
end
M.play_below_spazer_floor_to_west = west.play_below_spazer_floor_to_west
M.play_warehouse_to_east = warehouse.play_warehouse_to_east
M.play_east_to_glass = east.play_east_to_glass
M.play_glass_to_west = glass.play_glass_to_west
M.play_west_to_below = west_rev.play_west_to_below
M.play_below_to_bat = below.play_below_to_bat
M.play_bat_to_red = bat.play_bat_to_red
M.play_red_to_hellway = hellway.play_red_to_hellway
M.play_hellway_to_caterpillar = cat.play_hellway_to_caterpillar
M.play_caterpillar_to_alpha_pb = alpha.play_caterpillar_to_alpha_pb
return M
