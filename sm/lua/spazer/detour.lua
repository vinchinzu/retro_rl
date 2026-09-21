-- Mainline K2.2 fuse: climb → Super door → collect → return → West.
-- Port of snes/super_metroid/routes/kpdr/spazer/detour.py.

local geo = require("red_tower.ctrl")
local g = require("spazer.geometry")
local approach = require("spazer.approach")
local collect = require("spazer.collect")
local drop = require("spazer.drop")

local M = {}

function M.play_spazer_detour(session)
  geo.require_room(session, geo.ROOM_BELOW_SPAZER, "spazer_detour")
  if g.has_spazer(session.state) then
    return drop.play_spazer_top_to_west(session)
  end
  approach.play_below_spazer_to_spazer(session)
  collect.play_spazer_collect(session)
  collect.play_spazer_return_to_below(session)
  return drop.play_spazer_top_to_west(session)
end

return M
