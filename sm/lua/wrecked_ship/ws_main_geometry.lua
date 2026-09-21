-- Powered Main Shaft geometry: bands, hops, region / phase classifier.

local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")

local M = {}
local ROOM_WS_MAIN = rooms.ROOM_WS_MAIN or 0xCAF6
local ROOM_WS_ATTIC = rooms.ROOM_WS_ATTIC or 0xCA52
local ROOM_WS_SAVE = rooms.ROOM_WS_SAVE or 0xCE8A
local ROOM_WS_WEST_SUPER = rooms.ROOM_WS_WEST_SUPER or 0xCDA8

M.WS_MAIN_PHASES = {"pit_shot", "grate_seat", "west_super", "mid_climb", "attic_seat", "attic_door"}
M.PIT, M.GRATE_SEAT, M.SHELF, M.SAVE_ALCOVE, M.SAVE_COLUMN, M.SHAFT, M.ATTIC_SEAT, M.ATTIC =
  "pit", "grate_seat", "shelf", "save_alcove", "save_column", "shaft", "attic_seat", "attic"

M.WS_MAIN_SAVE_X = 1240
M.WS_MAIN_STAIR_Y = 1920
M.WS_MAIN_FLOOR_Y = 1960
M.WS_MAIN_PIT_Y = 1850
M.WS_MAIN_SHAFT_CENTER = 1152
M.WS_MAIN_ATTIC_DOOR_X = 1135
M.TUNNEL_CLEAR_X = 1088
M.FIRST_JUMP_LAND_X = {1188, 1232}
M.FIRST_JUMP_LAND_Y = {1852, 1888}
M.FIRST_JUMP_LAND_TARGET_X = 1223
M.SHORT_HOP_X = {1163, 1171}
M.FIRST_JUMP_TAKEOFF_X = {1138, 1162}
M.FIRST_JUMP_TAKEOFF_TARGET_X = 1156
M.PIT_EXIT_RIGHT_X = 1104
M.LIP_SHOT_X, M.LIP_SHOT_Y = {1164, 1227}, {1852, 1896}
M.LIP_FIRE_X = {1188, 1227}
M.POCKET_RELEASE_CHARGE = 8
M.GROUNDED = {[1]=true,[2]=true,[3]=true,[4]=true,[9]=true,[10]=true}
M.CROUCH = {[39]=true,[40]=true}
M.AIR = {[19]=true,[20]=true,[21]=true,[25]=true,[26]=true,[47]=true,[48]=true,[75]=true,[76]=true,[77]=true,[78]=true,[81]=true,[82]=true,[83]=true,[84]=true}
M.WJ_POSES = {[19]=true,[20]=true,[132]=true}
M.TURNING_MOVEMENT = 14
M.THREE_SHOT_X_MIN, M.THREE_SHOT_X_MAX, M.THREE_SHOT_FRAMES = 1168, 1210, 240
M.GRATE_SEAT_PIN = {1216, 1232, 1852, 1868}
M.WEST_SUPER_Y = {1650, 1700}
M.MID_CLIMB_Y = {630, 710}
M.SHAFT_X = {1080, 1220}
M.SAVE_COLUMN_LATCH_X = 1216
M.SAVE_LEDGE_Y = {1836, 1876}
M.UPPER_WALL_SHOT_X, M.UPPER_WALL_CLEAR = 1228, 3
M.SLOPE_523_SEAT_Y = {508, 540}
M.SLOPE_651_SEAT_Y, M.SLOPE_651_WALL_X = {576, 660}, 1240
M.SLOPE_827_SEAT_Y, M.SLOPE_827_WALL_X = {756, 840}, 1064
M.SLOPE_1019_SEAT_Y, M.SLOPE_1019_WALL_X = {900, 1036}, 1240

M.UPPER_SHAFT_HOPS = {
  {y = 443, x0 = 1061, x1 = 1170, side = "RIGHT", take0 = 1110, take1 = 1140},
  {y = 363, x0 = 1180, x1 = 1243, side = "RIGHT", take0 = 1210, take1 = 1234},
  {y = 267, x0 = 1140, x1 = 1170, side = "LEFT", take0 = 1150, take1 = 1164},
  {y = 171, x0 = 1077, x1 = 1120, side = "LEFT", take0 = 1077, take1 = 1084},
  {y = 91, x0 = 1111, x1 = 1159, side = "RIGHT", take0 = 1120, take1 = 1140},
}

function M.ws_main_phase_index(name)
  local key = string.lower(tostring(name)):gsub("-", "_")
  local i
  for i = 1, #M.WS_MAIN_PHASES do
    if M.WS_MAIN_PHASES[i] == key then
      return i - 1
    end
  end
  error("unknown Main Shaft phase " .. tostring(name))
end

function M.ws_main_attic_settled(state)
  return ctrl.num(state.room_id) == ROOM_WS_ATTIC
    and ctrl.num(state.game_state) == 8
    and ctrl.num(state.door_transition) == 0
end

function M.at_ws_main_attic_door_seat(state)
  local pose = ctrl.num(state.pose)
  return ctrl.num(state.room_id) == ROOM_WS_MAIN
    and math.abs(ctrl.x(state) - M.WS_MAIN_ATTIC_DOOR_X) <= 24
    and ctrl.y(state) <= 160
    and (pose == 1 or pose == 2 or pose == 9 or pose == 10)
    and math.abs(ctrl.num(state.velocity_y)) <= 1
end

function M.at_ws_main_first_jump_land(x, y, pose, vy)
  return M.FIRST_JUMP_LAND_X[1] <= x and x <= M.FIRST_JUMP_LAND_X[2]
    and M.FIRST_JUMP_LAND_Y[1] <= y and y <= M.FIRST_JUMP_LAND_Y[2]
    and (M.GROUNDED[pose] or M.CROUCH[pose])
    and math.abs(vy) <= 1
end

function M.at_ws_main_grate_seat(state)
  return ctrl.num(state.room_id) == ROOM_WS_MAIN and ctrl.num(state.game_state) == 8
    and M.at_ws_main_first_jump_land(ctrl.x(state), ctrl.y(state), ctrl.num(state.pose), ctrl.num(state.velocity_y))
end

function M.at_ws_main_usable_grate_seat(state)
  if not (ctrl.num(state.room_id) == ROOM_WS_MAIN and ctrl.num(state.game_state) == 8) then
    return false
  end
  local x, y = ctrl.x(state), ctrl.y(state)
  return M.GRATE_SEAT_PIN[1] <= x and x <= M.GRATE_SEAT_PIN[2]
    and M.GRATE_SEAT_PIN[3] <= y and y <= M.GRATE_SEAT_PIN[4]
    and math.abs(ctrl.num(state.velocity_y)) <= 1
end

function M.at_ws_main_pit(state)
  if ctrl.num(state.room_id) ~= ROOM_WS_MAIN or M.at_ws_main_grate_seat(state) then
    return false
  end
  return ctrl.y(state) >= M.WS_MAIN_PIT_Y
end

function M.at_ws_main_west_super_band(state)
  local room = ctrl.num(state.room_id)
  if room == ROOM_WS_WEST_SUPER or room == ROOM_WS_SAVE or room == ROOM_WS_ATTIC then
    return false
  end
  local x, y = ctrl.x(state), ctrl.y(state)
  return ctrl.num(state.room_id) == ROOM_WS_MAIN and ctrl.num(state.game_state) == 8
    and M.SHAFT_X[1] <= x and x <= M.SHAFT_X[2]
    and M.WEST_SUPER_Y[1] <= y and y <= M.WEST_SUPER_Y[2]
end

function M.at_ws_main_mid_climb(state)
  local room = ctrl.num(state.room_id)
  if room == ROOM_WS_WEST_SUPER or room == ROOM_WS_SAVE or room == ROOM_WS_ATTIC then
    return false
  end
  local x, y, pose = ctrl.x(state), ctrl.y(state), ctrl.num(state.pose)
  return ctrl.num(state.room_id) == ROOM_WS_MAIN and ctrl.num(state.game_state) == 8
    and M.SHAFT_X[1] <= x and x <= M.SHAFT_X[2]
    and M.MID_CLIMB_Y[1] <= y and y <= M.MID_CLIMB_Y[2]
    and M.GROUNDED[pose]
    and math.abs(ctrl.num(state.velocity_y)) <= 1
end

function M.classify_region_xy(x, y, pose, vy)
  if math.abs(x - M.WS_MAIN_ATTIC_DOOR_X) <= 24 and y <= 160
      and (pose == 1 or pose == 2 or pose == 9 or pose == 10) and math.abs(vy) <= 1 then
    return M.ATTIC_SEAT
  end
  if M.at_ws_main_first_jump_land(x, y, pose, vy) then
    return M.GRATE_SEAT
  end
  if y >= M.WS_MAIN_PIT_Y then
    return M.PIT
  end
  return M.SHAFT
end

function M.classify_region(state)
  if M.ws_main_attic_settled(state) or ctrl.num(state.room_id) == ROOM_WS_ATTIC then
    return M.ATTIC
  end
  if M.at_ws_main_attic_door_seat(state) then
    return M.ATTIC_SEAT
  end
  return M.classify_region_xy(ctrl.x(state), ctrl.y(state), ctrl.num(state.pose), ctrl.num(state.velocity_y))
end

return M
