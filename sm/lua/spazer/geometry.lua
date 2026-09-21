-- Below Spazer / Spazer Room geometry: named bands and predicates.
-- Port of snes/super_metroid/routes/kpdr/spazer/geometry.py.

local geo = require("red_tower.ctrl")

local M = {}

M.SPAZER_BEAM_MASK = 0x0004
M.POSE_ITEM_GRAB = geo.set(164)
M.LAG_POSES = geo.set(137, 138, 164)
M.FLOOR_UNMORPH_POSES = geo.set(31, 39, 40, 41, 42, 65)
M.TRUE_GROUND = geo.TRUE_GROUND

M.TOP_Y_MAX = 160
M.CLIMB_TOP_Y = 190
M.SOLID_TOP_Y = {88, 150}
M.SOLID_TOP_X_MIN = 70
M.OVER_LIP_Y_MAX = 110
M.CREST_LAND_X = {75, 120}
M.MID_Y = 300
M.MID_BAND_Y = {210, 300}
M.MID_BAND_X = {40, 80}
M.MID_PLATFORM_Y = 220
M.FLOOR_Y_MIN = 360
M.HIGH_AIR_Y_MAX = 320
M.CREST_FAIL_Y = 420
M.DOOR_SAFE_X = 48
M.DOOR_X_MIN = 420
M.HANDOFF_X_MAX = 400
M.DOOR_TRAP_X_MAX = 430
M.TOP_LAND_X = {40, 400}
M.TOP_LEDGE_X_MAX = 480
M.FLOOR_LIP_X = {46, 58}
M.CACATAC_OFF_DOOR_X = 48

M.WJ_LEFT = {into = "LEFT", flip = "RIGHT", into_frames = 12, amid_frames = 2, flip_frames = 14}
M.WJ_RIGHT = {into = "RIGHT", flip = "LEFT", into_frames = 12, amid_frames = 2, flip_frames = 12}
M.WJ_PAIR = {M.WJ_LEFT, M.WJ_RIGHT}
M.CREST_PERIOD = 14
M.CREST_PERIOD_INTO = 4
M.CREST_PERIOD_FLIP = 3

function M.in_below_spazer(state)
  return state.room_id == geo.ROOM_BELOW_SPAZER
end

function M.is_true_ground_pose(state)
  return geo.TRUE_GROUND[state.pose] == true
end

function M.is_lag_pose(state)
  return M.LAG_POSES[state.pose] == true
end

function M.on_top_ledge(state, y_max)
  local ym = y_max or M.CLIMB_TOP_Y
  return M.in_below_spazer(state)
    and state.samus_y <= ym
    and state.samus_x >= M.TOP_LAND_X[1]
    and state.samus_x <= M.TOP_LEDGE_X_MAX
end

function M.on_solid_top(state)
  return M.in_below_spazer(state)
    and state.samus_y >= M.SOLID_TOP_Y[1]
    and state.samus_y <= M.SOLID_TOP_Y[2]
    and state.samus_x >= M.SOLID_TOP_X_MIN
    and M.is_true_ground_pose(state)
end

function M.solid_ish_top(state)
  if M.on_solid_top(state) then
    return true
  end
  return M.in_below_spazer(state)
    and state.samus_y <= M.TOP_Y_MAX
    and state.samus_x >= M.SOLID_TOP_X_MIN
    and M.is_true_ground_pose(state)
end

function M.on_super_door_approach(state)
  return M.in_below_spazer(state)
    and state.samus_x >= M.DOOR_X_MIN
    and state.samus_y <= M.TOP_Y_MAX
    and M.is_true_ground_pose(state)
end

function M.mid_band(state)
  return M.in_below_spazer(state)
    and state.samus_y >= M.MID_BAND_Y[1]
    and state.samus_y <= M.MID_BAND_Y[2]
    and state.samus_x >= M.MID_BAND_X[1]
    and state.samus_x <= M.MID_BAND_X[2]
end

function M.standing_mid_seat(state)
  return M.mid_band(state) and M.is_true_ground_pose(state)
end

function M.on_mid_or_floor(state)
  return M.in_below_spazer(state) and state.samus_y >= M.MID_PLATFORM_Y
end

function M.has_spazer(state)
  return geo.has_spazer(state)
end

return M
