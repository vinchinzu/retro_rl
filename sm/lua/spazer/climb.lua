-- Below Spazer climb: floor → standing mid → solid top (node 4).
-- Port of snes/super_metroid/routes/kpdr/spazer/climb.py.

local geo = require("red_tower.ctrl")
local g = require("spazer.geometry")
local helpers = require("spazer.helpers")
local scripts = require("spazer.scripts")

local M = {}

local function off_bat_door(session, target_x)
  target_x = target_x or g.DOOR_SAFE_X
  local high_air = session.state.samus_y <= g.HIGH_AIR_Y_MAX
    and not g.is_true_ground_pose(session.state)
  local limit = high_air and 10 or 40
  for _ = 1, limit do
    if not g.in_below_spazer(session.state) then
      return
    end
    if session.state.samus_x >= target_x
        and (session.state.door_transition or 0) == 0 then
      return
    end
    if high_air then
      geo.hold(session, 1, {"RIGHT"}, "spazer_off_bat_door_air")
    else
      geo.hold(session, 1, {"RIGHT", "B"}, "spazer_off_bat_door")
    end
  end
end

local function settle_over_lip(session)
  for _ = 1, 45 do
    if not g.in_below_spazer(session.state) then
      return session.state
    end
    if g.on_solid_top(session.state) then
      return session.state
    end
    local sx = session.state.samus_x
    if sx < g.CREST_LAND_X[1] then
      geo.hold(session, 1, {"RIGHT"}, "spazer_crest_land")
    elseif sx > g.CREST_LAND_X[2] then
      geo.hold(session, 1, {"LEFT"}, "spazer_crest_land")
    else
      geo.hold(session, 1, {}, "spazer_crest_land")
    end
    if g.on_solid_top(session.state) then
      return session.state
    end
  end
  return session.state
end

local function crest_period_over_lip(session, frames)
  frames = frames or 100
  for i = 0, frames - 1 do
    if not g.in_below_spazer(session.state) then
      return session.state
    end
    if g.on_solid_top(session.state) then
      return session.state
    end
    local x = session.state.samus_x
    local y = session.state.samus_y
    if x >= g.SOLID_TOP_X_MIN and y <= g.OVER_LIP_Y_MAX then
      return settle_over_lip(session)
    end
    if y > g.CREST_FAIL_Y then
      return session.state
    end
    if x < 40 then
      geo.hold(session, 2, {"RIGHT", "B"}, "spazer_crest_door")
    else
      local ph = i % g.CREST_PERIOD
      if ph < g.CREST_PERIOD_INTO then
        geo.hold(session, 1, {"LEFT", "A"}, "spazer_crest_period")
      elseif ph < g.CREST_PERIOD_INTO + g.CREST_PERIOD_FLIP then
        geo.hold(session, 1, {"RIGHT", "A"}, "spazer_crest_period")
      else
        geo.hold(session, 1, {"RIGHT", "B", "A"}, "spazer_crest_period")
      end
    end
  end
  return session.state
end

function M.play_below_spazer_mid_to_top(session, max_attempts)
  max_attempts = max_attempts or 5
  geo.require_room(session, geo.ROOM_BELOW_SPAZER, "below_spazer_mid_to_top")
  if session.state.samus_y > g.HIGH_AIR_Y_MAX or g.is_true_ground_pose(session.state) then
    geo.unmorph(session)
    off_bat_door(session)
  elseif session.state.samus_x < 40 then
    off_bat_door(session)
  end

  local function stop_high(state)
    if not g.in_below_spazer(state) then
      return true
    end
    if g.on_solid_top(state) then
      return true
    end
    if (state.door_transition or 0) ~= 0 and state.samus_x < g.DOOR_SAFE_X then
      return true
    end
    return state.samus_y <= 160
  end

  local pair2 = {g.WJ_LEFT, g.WJ_RIGHT, g.WJ_LEFT, g.WJ_RIGHT}
  for attempt = 0, max_attempts - 1 do
    if not g.in_below_spazer(session.state) then
      break
    end
    if g.on_solid_top(session.state) then
      break
    end
    if session.state.samus_x < g.DOOR_SAFE_X
        and session.state.samus_y > g.HIGH_AIR_Y_MAX then
      off_bat_door(session)
    end
    geo.consecutive_walljumps(session, pair2, "spazer_wj" .. attempt, 0, stop_high)
    if g.on_solid_top(session.state) then
      break
    end
    if session.state.samus_y <= 200 then
      crest_period_over_lip(session)
    end
    if g.on_solid_top(session.state) then
      break
    end
    geo.hold(session, 3, {}, "spazer_wj_gap")
  end

  if not g.in_below_spazer(session.state) then
    error("below_spazer_mid_to_top: left Below Spazer: " .. geo.fmt_state(session.state))
  end
  if not g.on_solid_top(session.state) then
    error("below_spazer_mid_to_top: missed solid top: " .. geo.fmt_state(session.state))
  end
  geo.hold(session, 8, {}, "spazer_top_settle")
  return session.state
end

local function clear_floor_cacatac(session)
  helpers.try_select_weapon(session, 0)
  geo.hold(session, 4, {}, "spazer_cacatac_weapon")
  if session.state.samus_x < g.CACATAC_OFF_DOOR_X then
    geo.hold(session, 6, {"RIGHT"}, "spazer_cacatac_off_door")
    geo.hold(session, 4, {}, "spazer_cacatac_off_door")
  end
  for _ = 1, 6 do
    if not g.in_below_spazer(session.state) then
      return
    end
    if g.mid_band(session.state) or g.on_solid_top(session.state) then
      return
    end
    geo.hold(session, 3, {"UP"}, "spazer_cacatac_aim")
    geo.hold(session, 2, {"UP", "X"}, "spazer_cacatac_shot")
    geo.hold(session, 16, {"UP"}, "spazer_cacatac_wait")
  end
  geo.hold(session, 8, {}, "spazer_cacatac_settle")
  for _ = 1, 20 do
    if not g.in_below_spazer(session.state) then
      return
    end
    if g.is_lag_pose(session.state) then
      geo.break_rle_lag(session, "spazer_cacatac_lag")
    else
      local x = session.state.samus_x
      if x < g.FLOOR_LIP_X[1] then
        geo.hold(session, 1, {"RIGHT"}, "spazer_cacatac_reseal")
      elseif x > g.FLOOR_LIP_X[2] then
        geo.hold(session, 1, {"LEFT"}, "spazer_cacatac_reseal")
      elseif not g.is_true_ground_pose(session.state) then
        geo.hold(session, 1, {}, "spazer_cacatac_reseal")
      else
        break
      end
    end
  end
  geo.hold(session, 6, {}, "spazer_cacatac_ready")
end

function M.play_below_spazer_floor_to_mid(session)
  geo.require_room(session, geo.ROOM_BELOW_SPAZER, "below_spazer_floor_to_mid")
  geo.unmorph(session)
  if g.mid_band(session.state) or g.on_solid_top(session.state) then
    return session.state
  end
  clear_floor_cacatac(session)
  if g.mid_band(session.state) or g.on_solid_top(session.state) then
    return session.state
  end

  local function stop(state)
    return g.standing_mid_seat(state) or g.on_solid_top(state)
  end

  geo.play_script(session, scripts.FLOOR_MID_RLE, {
    reason = "spazer_floor_mid",
    room_id = geo.ROOM_BELOW_SPAZER,
    stop_when = stop,
    on_lag = "ignore",
  })
  for _ = 1, 60 do
    if not g.in_below_spazer(session.state) then
      return session.state
    end
    if g.mid_band(session.state) or g.on_solid_top(session.state) then
      return session.state
    end
    if g.is_lag_pose(session.state) then
      geo.break_rle_lag(session, "spazer_floor_mid_lag")
    else
      geo.hold(session, 1, {}, "spazer_floor_mid_settle")
    end
  end
  if not g.in_below_spazer(session.state) then
    error("below_spazer_floor_to_mid: left Below Spazer: " .. geo.fmt_state(session.state))
  end
  if not g.mid_band(session.state) and not g.on_solid_top(session.state) then
    error("below_spazer_floor_to_mid: missed mid band: " .. geo.fmt_state(session.state))
  end
  return session.state
end

function M.play_below_spazer_climb(session)
  geo.require_room(session, geo.ROOM_BELOW_SPAZER, "below_spazer_climb")
  geo.unmorph(session)
  if g.on_solid_top(session.state) then
    return session.state
  end
  if not g.standing_mid_seat(session.state) and session.state.samus_y > 280 then
    M.play_below_spazer_floor_to_mid(session)
  end
  if g.on_solid_top(session.state) then
    return session.state
  end
  if session.state.samus_y <= g.MID_Y then
    M.play_below_spazer_mid_to_top(session)
  end
  if not g.on_solid_top(session.state) then
    error("below_spazer_climb: missed solid top: " .. geo.fmt_state(session.state))
  end
  return session.state
end

return M
