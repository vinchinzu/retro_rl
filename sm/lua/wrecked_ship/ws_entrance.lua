-- Wrecked Ship Entrance → Main Shaft. Walk/run right. Beam the blue door.

local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")

local M = {}
local ROOM_WS_ENTRANCE = rooms.ROOM_WS_ENTRANCE or 0xCA08
local ROOM_WS_MAIN = rooms.ROOM_WS_MAIN or 0xCAF6
local DOOR_X_MIN, DOOR_X_MAX = 900, 1024
local RUN_TIMEOUT, SETTLE = 400, 200

function M.at_ws_entrance_door_seat(state)
  local x = ctrl.x(state)
  return ctrl.num(state.room_id) == ROOM_WS_ENTRANCE and DOOR_X_MIN <= x and x <= DOOR_X_MAX
end

function M.play_ws_entrance_to_main(session)
  local label = "ws_entrance_to_main"
  ctrl.require_room(session, ROOM_WS_ENTRANCE, label)
  pcall(ctrl.select_weapon, session, 0)
  local i
  for i = 1, RUN_TIMEOUT do
    local st = session.state
    if ctrl.num(st.room_id) == ROOM_WS_MAIN or M.at_ws_entrance_door_seat(st) then
      break
    end
    ctrl.hold(session, 1, {"RIGHT", "B"}, label .. "_run")
  end
  if ctrl.num(session.state.room_id) == ROOM_WS_MAIN then
    return ctrl.wait_ordinary_room(session, ROOM_WS_MAIN, {settle_frames = SETTLE, label = label})
  end
  return ctrl.play_run_shoot_exit(session, {
    from_room = ROOM_WS_ENTRANCE,
    to_room = ROOM_WS_MAIN,
    direction = "RIGHT",
    label = label,
    run_frames = 0,
    shoot_frames = 10,
    spin_frames = 0,
    hold_frames = 200,
    settle_frames = SETTLE,
  })
end

return M
