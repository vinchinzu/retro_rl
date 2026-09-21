-- Top return handoff → mid/floor, then West Tunnel exit.
-- Port of snes/super_metroid/routes/kpdr/spazer/drop.py.

local geo = require("red_tower.ctrl")
local west = require("red_tower.below_spazer_west")
local g = require("spazer.geometry")
local scripts = require("spazer.scripts")

local M = {}

local function settle_floor_land(session)
  for _ = 1, 80 do
    if not g.in_below_spazer(session.state) then
      return session.state
    end
    if g.is_true_ground_pose(session.state) then
      geo.hold(session, 8, {}, "spazer_top_mid_floor_stand")
      return session.state
    end
    if g.is_lag_pose(session.state) then
      geo.break_rle_lag(session)
    elseif g.FLOOR_UNMORPH_POSES[session.state.pose] then
      geo.hold(session, 1, {"UP"}, "spazer_top_mid_unmorph")
    else
      geo.hold(session, 1, {}, "spazer_top_mid_land")
    end
  end
  return session.state
end

function M.play_spazer_top_to_mid(session)
  geo.require_room(session, geo.ROOM_BELOW_SPAZER, "spazer_top_to_mid")
  geo.unmorph(session)
  if g.on_mid_or_floor(session.state) then
    return session.state
  end
  geo.break_rle_lag(session)

  local function stop(state)
    if state.samus_y >= g.FLOOR_Y_MIN then
      return true
    end
    return g.on_mid_or_floor(state) and g.is_true_ground_pose(state)
  end

  geo.play_script(session, scripts.TOP_MID_RLE, {
    reason = "spazer_top_mid",
    room_id = geo.ROOM_BELOW_SPAZER,
    stop_when = stop,
    on_lag = "break",
  })

  if not g.in_below_spazer(session.state) then
    error("spazer_top_to_mid: left Below Spazer: " .. geo.fmt_state(session.state))
  end
  if session.state.samus_y >= g.FLOOR_Y_MIN then
    return settle_floor_land(session)
  end
  if not g.on_mid_or_floor(session.state) then
    error("spazer_top_to_mid: missed mid/floor (need y>="
      .. tostring(g.MID_PLATFORM_Y) .. "): " .. geo.fmt_state(session.state))
  end
  return session.state
end

local function walk_mid_platforms_to_floor(session)
  for _ = 1, 250 do
    if not g.in_below_spazer(session.state) then
      return
    end
    if session.state.samus_y >= g.FLOOR_Y_MIN then
      return
    end
    if g.is_lag_pose(session.state) then
      geo.hold(session, 6, {"A"}, "spazer_mid_lag")
      geo.hold(session, 4, {}, "spazer_mid_lag")
    else
      geo.hold(session, 1, {"LEFT", "B"}, "spazer_mid_left")
    end
  end
end

function M.mid_or_floor_to_west(session)
  walk_mid_platforms_to_floor(session)
  return west.play_below_spazer_floor_to_west(session)
end

function M.play_spazer_top_to_west(session)
  geo.require_room(session, geo.ROOM_BELOW_SPAZER, "spazer_top_to_west")
  geo.unmorph(session)
  if not g.on_mid_or_floor(session.state) then
    M.play_spazer_top_to_mid(session)
  end
  if not g.on_mid_or_floor(session.state) then
    error("spazer_top_to_west: top→mid failed (need y>="
      .. tostring(g.MID_PLATFORM_Y) .. "; got y="
      .. tostring(session.state.samus_y) .. "): " .. geo.fmt_state(session.state))
  end
  return M.mid_or_floor_to_west(session)
end

return M
