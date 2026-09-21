-- Shared Spazer hop helpers: weapon select, ground wait, RLE player.
-- Port of snes/super_metroid/routes/kpdr/spazer/helpers.py.

local geo = require("red_tower.ctrl")
local g = require("spazer.geometry")

local M = {}

M.play_script = geo.play_script
M.break_rle_lag = geo.break_rle_lag
M.try_select_weapon = geo.try_select_weapon

function M.wait_true_ground(session, opts)
  opts = opts or {}
  local reason = opts.reason or "spazer_ground"
  local budget = opts.budget or 80
  local room_id = opts.room_id
  if room_id == nil then
    room_id = geo.ROOM_BELOW_SPAZER
  end
  for _ = 1, budget do
    if room_id ~= nil and session.state.room_id ~= room_id then
      return session.state
    end
    if g.is_true_ground_pose(session.state) then
      return session.state
    end
    if g.is_lag_pose(session.state) then
      geo.break_rle_lag(session, reason)
    else
      geo.hold(session, 1, {}, reason)
    end
  end
  return session.state
end

return M
