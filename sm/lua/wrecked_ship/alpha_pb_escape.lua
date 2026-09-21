-- Alpha PB escape and K6 approach tapes to Moat.

local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")

local M = {}
local ROOM_ALPHA_PB = rooms.ROOM_ALPHA_PB or 0xA3AE
local ROOM_CATERPILLAR = rooms.ROOM_CATERPILLAR or 0xA322
local ROOM_ELEV = rooms.ROOM_RED_BRINSTAR_ELEVATOR or 0x962A
local ROOM_KIHUNTER = rooms.ROOM_CRATERIA_KIHUNTER or 0x948C
local ROOM_MOAT = rooms.ROOM_MOAT or 0x95FF
local ESCAPE_BUDGET = 2200
local PROGRESS_WINDOW = 42

local function clear_obstacle(session, label)
  local i
  for i = 1, 18 do
    ctrl.hold(session, 1, {"A"}, label .. "_jump")
  end
  for i = 0, 33 do
    local buttons = {"R"}
    if (i % 3) == 0 then
      buttons[2] = "X"
    end
    ctrl.hold(session, 1, buttons, label .. "_aim_shoot")
  end
  ctrl.hold(session, 10, {}, label .. "_land")
end

function M.play_alpha_pb_to_caterpillar(session)
  ctrl.require_room(session, ROOM_ALPHA_PB, "alpha_pb_to_caterpillar")
  pcall(ctrl.select_weapon, session, 0)
  local best_x = ctrl.x(session.state)
  local stale, frame = 0, 0
  for frame = 0, ESCAPE_BUDGET - 1 do
    local state = session.state
    if ctrl.num(state.room_id) == ROOM_CATERPILLAR then
      ctrl.wait_ordinary_room(session, ROOM_CATERPILLAR, {
        settle_frames = 260,
        label = "alpha_pb_to_caterpillar",
        x_range = {20, 80},
        y_range = {1920, 1940},
      })
      local i
      for i = 1, 60 do
        state = session.state
        if ctrl.y(state) >= 1930 and ctrl.num(state.velocity_y) == 0 then
          return state
        end
        ctrl.hold(session, 1, {}, "alpha_pb_to_caterpillar_land")
      end
      ctrl.timeout("alpha_pb_to_caterpillar: Caterpillar entry did not land: " .. ctrl.brief(session.state))
    end
    if ctrl.num(state.room_id) ~= ROOM_ALPHA_PB then
      ctrl.timeout(string.format("alpha_pb_to_caterpillar: unexpected room 0x%04X", ctrl.num(state.room_id)))
    end
    if ctrl.num(state.max_power_bombs) <= 0 then
      ctrl.timeout("alpha_pb_to_caterpillar: Alpha PB is not collected")
    end
    local x = ctrl.x(state)
    if x > best_x + 2 then
      best_x, stale = x, 0
    else
      stale = stale + 1
    end
    if stale >= PROGRESS_WINDOW then
      clear_obstacle(session, "alpha_pb_escape_stall")
      stale = 0
      best_x = ctrl.x(session.state)
    else
      local buttons = {"RIGHT", "B", "X"}
      if (frame % 52) < 30 then
        buttons[4] = "A"
      end
      ctrl.hold(session, 1, buttons, "alpha_pb_escape_advance")
    end
  end
  ctrl.timeout("alpha_pb_to_caterpillar: escape timeout: " .. ctrl.brief(session.state))
end

function M.play_caterpillar_to_elevator(session)
  return ctrl.play_rle_room_exit(session, {
    from_room = ROOM_CATERPILLAR,
    to_room = ROOM_ELEV,
    script = require("wrecked_ship.tapes.caterpillar_to_elevator"),
    label = "caterpillar_to_elevator",
    settle_frames = 260,
  })
end

function M.play_elevator_to_kihunter(session)
  return ctrl.play_rle_room_exit(session, {
    from_room = ROOM_ELEV,
    to_room = ROOM_KIHUNTER,
    script = require("wrecked_ship.tapes.elevator_to_kihunter"),
    label = "elevator_to_kihunter",
  })
end

function M.play_kihunter_to_moat(session)
  return ctrl.play_rle_room_exit(session, {
    from_room = ROOM_KIHUNTER,
    to_room = ROOM_MOAT,
    script = require("wrecked_ship.tapes.kihunter_to_moat"),
    label = "kihunter_to_moat",
    settle_frames = 260,
  })
end

return M
