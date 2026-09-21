-- Wrecked Ship Main Shaft → Basement. Human RLE body, ordinary gs=8.

local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")

local M = {}
local ROOM_WS_MAIN = rooms.ROOM_WS_MAIN or 0xCAF6
local ROOM_WS_BASEMENT = rooms.ROOM_WS_BASEMENT or 0xCC6F
local SETTLE = 200

function M.ws_main_basement_settled(state)
  return ctrl.num(state.room_id) == ROOM_WS_BASEMENT
    and ctrl.num(state.game_state) == 8
    and ctrl.num(state.door_transition) == 0
end

function M.play_ws_main_to_basement(session)
  local label = "ws_main_to_basement"
  ctrl.require_room(session, ROOM_WS_MAIN, label)
  if M.ws_main_basement_settled(session.state) then
    return session.state
  end
  ctrl.play_script(session, require("wrecked_ship.tapes.ws_main_to_basement"), {
    reason = label .. "_body",
    room_id = ROOM_WS_MAIN,
    stop_when = function(state)
      return ctrl.num(state.room_id) ~= ROOM_WS_MAIN
    end,
  })
  return ctrl.wait_ordinary_room(session, ROOM_WS_BASEMENT, {settle_frames = SETTLE, label = label})
end

return M
