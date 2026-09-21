-- Hellway → Caterpillar reactive return (K5 hop 13).

local rooms = require("rooms")
local ctrl = require("red_tower.ctrl")

local M = {}
local ROOM_HELLWAY = rooms.ROOM_HELLWAY or 0xA2F7
local ROOM_CATERPILLAR = rooms.ROOM_CATERPILLAR or 0xA322
local DOOR_SEAT_X = 690
local DOOR_SEAT_X_MAX = 800
local TRAVERSE_BUDGET = 2200
local JUMP_PERIOD = 36
local JUMP_HOLD = 24

function M.play_hellway_to_caterpillar(session)
  ctrl.require_room(session, ROOM_HELLWAY, "hellway_to_caterpillar")
  local frame
  for frame = 0, TRAVERSE_BUDGET - 1 do
    local state = session.state
    if ctrl.num(state.room_id) ~= ROOM_HELLWAY then
      break
    end
    local x = ctrl.x(state)
    if x > 20000 then
      ctrl.hold(session, 1, {"RIGHT"}, "hellway_to_caterpillar_door_remap")
    elseif x > DOOR_SEAT_X and x < DOOR_SEAT_X_MAX then
      ctrl.hold(session, 80, {}, "hellway_to_caterpillar_door_land")
      return ctrl.play_run_shoot_exit(session, {
        from_room = ROOM_HELLWAY,
        to_room = ROOM_CATERPILLAR,
        direction = "RIGHT",
        label = "hellway_to_caterpillar",
        run_frames = 0,
        shoot_frames = 24,
        spin_frames = 16,
        hold_frames = 240,
        settle_frames = 200,
      })
    else
      local buttons = {"RIGHT", "B", "X"}
      if (frame % JUMP_PERIOD) < JUMP_HOLD then
        buttons[4] = "A"
      end
      ctrl.hold(session, 1, buttons, "hellway_to_caterpillar_traverse")
    end
  end
  ctrl.timeout("hellway_to_caterpillar: traverse timeout: " .. ctrl.brief(session.state))
end

return M
