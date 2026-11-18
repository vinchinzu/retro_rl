-- Maridia Botwoon package. Not a live proof.

local aqueduct = require("botwoon.aqueduct")
local fight = require("botwoon.fight")

local M = {}

M.proven = false
M.play_main_street_to_botwoon = aqueduct.play_main_street_to_botwoon
M.play_botwoon_fight = fight.play_botwoon_fight

return M
