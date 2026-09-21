-- Spazer-safe Caterpillar descent to the Alpha Power Bomb room (K5 hop 14).

local rooms = require("rooms")
local ctrl = require("red_tower.ctrl")
local alpha = require("red_tower.alpha_pb")

local M = {}
M.ROOM_CATERPILLAR = rooms.ROOM_CATERPILLAR or 0xA322
M.ROOM_ALPHA_PB = rooms.ROOM_ALPHA_PB or 0xA3AE
local DESCENT_BUDGET = 1800
local ENTRY_SHAFT_MIN_X = 78
local ENTRY_SHAFT_MAX_X = 100
local ENTRY_FLOOR_Y = 1405

local function downshot(session, frame, horizontal)
  local buttons = {}
  if horizontal then
    buttons[1] = horizontal
    buttons[2] = "DOWN"
    if (frame % 3) == 0 then
      buttons[3] = "X"
    end
  else
    buttons[1] = "DOWN"
    if (frame % 3) == 0 then
      buttons[2] = "X"
    end
  end
  ctrl.hold(session, 1, buttons, "caterpillar_spazer_downshot")
end

function M.play_caterpillar_to_alpha_pb(session)
  ctrl.require_room(session, M.ROOM_CATERPILLAR, "caterpillar_to_alpha_pb")
  pcall(ctrl.select_weapon, session, 0)
  local bottom_frames = 0
  local frame
  for frame = 0, DESCENT_BUDGET - 1 do
    local state = session.state
    if ctrl.num(state.room_id) == M.ROOM_ALPHA_PB then
      ctrl.wait_ordinary_room(session, M.ROOM_ALPHA_PB, {
        settle_frames = 260,
        label = "caterpillar_to_alpha_pb",
      })
      return alpha.play_alpha_pb_collect(session)
    end
    if ctrl.num(state.room_id) ~= M.ROOM_CATERPILLAR then
      ctrl.timeout(string.format(
        "caterpillar_to_alpha_pb: unexpected room 0x%04X",
        ctrl.num(state.room_id)
      ))
    end
    local x, y = ctrl.x(state), ctrl.y(state)
    if y < 1490 then
      local grounded = ctrl.num(state.velocity_y) == 0
        and ctrl.num(state.vertical_direction) == 0
      if grounded and x > ENTRY_SHAFT_MAX_X then
        ctrl.hold(session, 1, {"LEFT"}, "caterpillar_entry_recenter")
      elseif x < ENTRY_SHAFT_MIN_X then
        ctrl.hold(session, 1, {"RIGHT"}, "caterpillar_entry_runup")
      elseif y >= ENTRY_FLOOR_Y and (grounded or ctrl.num(state.vertical_direction) == 1) then
        ctrl.hold(session, 1, {"A"}, "caterpillar_entry_jump")
      else
        downshot(session, frame)
      end
    elseif y < 1585 then
      if x > 70 then
        downshot(session, frame, "LEFT")
      else
        downshot(session, frame)
      end
    elseif y < 1810 then
      if y >= 1660 and ctrl.num(state.velocity_y) == 0 then
        ctrl.hold(session, 1, {"LEFT"}, "caterpillar_third_shelf_left")
      else
        local h
        if x < 78 then
          h = "RIGHT"
        elseif x > 100 then
          h = "LEFT"
        end
        downshot(session, frame, h)
      end
    elseif y < 1910 then
      if x > 72 then
        downshot(session, frame, "LEFT")
      else
        downshot(session, frame)
      end
    elseif y < 1928 or ctrl.num(state.velocity_y) ~= 0 then
      ctrl.hold(session, 1, {}, "caterpillar_bottom_land")
    else
      if bottom_frames == 0 then
        pcall(ctrl.select_weapon, session, 2)
        ctrl.hold(session, 1, {}, "caterpillar_bottom_super_selected")
      elseif bottom_frames < 3 then
        ctrl.hold(session, 1, {"LEFT"}, "caterpillar_bottom_door_face")
      elseif bottom_frames < 8 then
        ctrl.hold(session, 1, {}, "caterpillar_bottom_door_release")
      elseif bottom_frames == 8 then
        ctrl.hold(session, 1, {"X"}, "caterpillar_bottom_door_shot")
      elseif bottom_frames < 78 then
        ctrl.hold(session, 1, {}, "caterpillar_bottom_door_open")
      else
        ctrl.hold(session, 1, {"LEFT"}, "caterpillar_bottom_left_exit")
      end
      bottom_frames = bottom_frames + 1
    end
  end
  ctrl.timeout("caterpillar_to_alpha_pb: descent timeout: " .. ctrl.brief(session.state))
end

return M
