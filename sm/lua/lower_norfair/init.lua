-- Lower Norfair into Ridley. Not a live proof.

local hall = require("lower_norfair.hall")
local ridley = require("lower_norfair.ridley")

local M = {}

M.proven = false
M.play_ridley_fight = ridley.play_fight

function M.play_space_jump_to_ridley(session)
  hall.play_entry(session)
  return ridley.play_fight(session)
end

return M
