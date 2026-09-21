-- Below Spazer → Bat Room (K5 hop 9). Pin right high sill for bat_to_red.

local rooms = require("rooms")
local ctrl = require("red_tower.ctrl")
local geom = require("red_tower.geometry")

local M = {}
local ROOM_BAT = rooms.ROOM_BAT or 0xA3DD
local ROOM_BELOW_SPAZER = rooms.ROOM_BELOW_SPAZER or 0xA408
local MORPH = {
  [27]=true,[28]=true,[29]=true,[30]=true,[31]=true,[37]=true,[38]=true,
  [39]=true,[40]=true,[41]=true,[42]=true,[43]=true,[45]=true,[49]=true,
  [50]=true,[55]=true,[65]=true,[137]=true,[138]=true,
}

local function on_sill(state)
  local x, y = ctrl.x(state), ctrl.y(state)
  return geom.BAT_SILL_X_MIN <= x and x <= geom.BAT_SILL_X_MAX
    and geom.BAT_SILL_Y_MIN <= y and y <= geom.BAT_SILL_Y_MAX
    and ctrl.num(state.velocity_y) == 0
    and not MORPH[ctrl.num(state.pose)]
end

local function pin_sill(session)
  ctrl.unmorph(session)
  local i
  for i = 1, geom.BAT_SILL_PIN_FRAMES do
    local state = session.state
    if ctrl.num(state.room_id) ~= ROOM_BAT then
      ctrl.timeout("below_to_bat: left Bat while pinning sill: " .. ctrl.brief(state))
    end
    if on_sill(state) then
      return ctrl.wait_ordinary_room(session, ROOM_BAT, {
        settle_frames = geom.BELOW_TO_BAT_SETTLE,
        label = "below_to_bat",
        x_range = {geom.BAT_SILL_X_MIN, geom.BAT_SILL_X_MAX},
        y_range = {geom.BAT_SILL_Y_MIN, geom.BAT_SILL_Y_MAX},
      })
    end
    if MORPH[ctrl.num(state.pose)] then
      ctrl.unmorph(session)
    else
      local x, y = ctrl.x(state), ctrl.y(state)
      if y > geom.BAT_SILL_Y_MAX then
        ctrl.hold(session, 1, {"RIGHT", "B", "A"}, "below_to_bat_sill_hop")
      elseif x < geom.BAT_SILL_X_MIN + 30 then
        ctrl.hold(session, 1, {"RIGHT", "B"}, "below_to_bat_sill_right")
      elseif x > geom.BAT_SILL_X_MAX then
        ctrl.hold(session, 1, {"LEFT"}, "below_to_bat_sill_left")
      else
        ctrl.hold(session, 1, {}, "below_to_bat_sill_wait")
      end
    end
  end
  ctrl.timeout("below_to_bat: failed to pin right high sill: " .. ctrl.brief(session.state))
end

function M.play_below_to_bat(session)
  ctrl.require_room(session, ROOM_BELOW_SPAZER, "below_to_bat")
  ctrl.hold(session, 6, {}, "below_to_bat_entry_glide")
  ctrl.unmorph(session)
  pcall(ctrl.select_weapon, session, 0)
  local frame
  for frame = 0, geom.BELOW_TO_BAT_FRAMES - 1 do
    local buttons
    if (frame % 35) < 10 then
      buttons = {"LEFT", "B", "X"}
    else
      buttons = {"LEFT", "B", "A"}
    end
    local state = ctrl.hold(session, 1, buttons, "below_to_bat_left")
    if ctrl.num(state.room_id) == ROOM_BAT then
      return pin_sill(session)
    end
  end
  ctrl.timeout("below_to_bat: Bat Room not reached: " .. ctrl.brief(session.state))
end

return M
