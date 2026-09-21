-- Natural Alpha Power Bomb room collection from its right-hand entry.

local rooms = require("rooms")
local ctrl = require("red_tower.ctrl")

local M = {}
M.ROOM_ALPHA_PB = rooms.ROOM_ALPHA_PB or 0xA3AE
M.COLLECT_ATTEMPT_FRAMES = 2200

function M.play_alpha_pb_collect(session, max_frames)
  max_frames = max_frames or M.COLLECT_ATTEMPT_FRAMES
  ctrl.require_room(session, M.ROOM_ALPHA_PB, "alpha_pb_collect")
  local body = require("red_tower.tapes.alpha_pb")
  ctrl.play_script(session, body, {
    reason = "alpha_pb_human_body",
    stop_when = function(state)
      return ctrl.num(state.max_power_bombs) > 0
    end,
  })
  if ctrl.num(session.state.max_power_bombs) > 0 then
    return session.state
  end
  local frame
  for frame = 0, max_frames - 1 do
    local state = session.state
    if ctrl.num(state.max_power_bombs) > 0 then
      return state
    end
    if ctrl.x(state) > 360 then
      local buttons = {"LEFT", "B"}
      if (frame % 44) < 30 then
        buttons[3] = "A"
      end
      ctrl.hold(session, 1, buttons, "alpha_pb_cross_left")
    else
      ctrl.hold(session, 1, {"LEFT"}, "alpha_pb_collect_plm")
    end
  end
  ctrl.timeout(
    "alpha_pb_collect: timeout before PB capacity increased: "
      .. ctrl.brief(session.state)
      .. " max_pb=" .. tostring(session.state.max_power_bombs)
  )
end

return M
