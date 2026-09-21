-- Bombs-exit (post-Torizo Parlor) → Red Tower entry.
-- Compose only. Hop bodies live in the sibling brinstar modules.

local rooms = {}
pcall(function()
  rooms = require("rooms")
end)

local M = {}

M.ROOM_PARLOR = rooms.ROOM_PARLOR or 0x92FD
M.ROOM_RED_TOWER = rooms.ROOM_RED_TOWER or 0xA253
M.LEAVE_ROOM_ID = M.ROOM_RED_TOWER

function M.play_to_red_tower(session)
  local spore = require("brinstar.spore_spawn")
  local supers = require("brinstar.super_collect")
  local pink = require("brinstar.pink_shaft")
  local to_ghz = require("brinstar.pink_to_ghz")
  local to_red = require("brinstar.ghz_to_red")

  spore.play_parlor_to_main_shaft(session)
  spore.play_main_shaft_to_spore_spawn(session)
  supers.play_super_room_collect(session)
  supers.play_super_room_to_farming(session)
  supers.play_farming_to_big_pink(session)
  pink.play_big_pink_into_main_shaft(session)
  to_ghz.play_big_pink_to_ghz(session)
  to_red.play_ghz_to_noob(session)
  to_red.play_noob_to_red_tower(session)
  return session.state
end

return M
