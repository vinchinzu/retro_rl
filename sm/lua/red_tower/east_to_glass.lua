-- East Tunnel → Glass Tunnel (K5 hop 6). Reverse of glass→east.

local rooms = require("rooms")
local ctrl = require("red_tower.ctrl")
local geom = require("red_tower.geometry")

local M = {}
local ROOM_EAST_TUNNEL = rooms.ROOM_EAST_TUNNEL or 0xCF80
local ROOM_GLASS = rooms.ROOM_GLASS or 0xCEFB
local MORPH = {
  [27]=true,[28]=true,[29]=true,[30]=true,[31]=true,[37]=true,[38]=true,
  [39]=true,[40]=true,[41]=true,[42]=true,[43]=true,[45]=true,[49]=true,
  [50]=true,[55]=true,[65]=true,[137]=true,[138]=true,
}
local CROUCH = {[26]=true,[39]=true,[40]=true,[43]=true}
local STAND = {
  [1]=true,[2]=true,[9]=true,[10]=true,[12]=true,[25]=true,[75]=true,[77]=true,[81]=true,
}

function M.play_east_to_glass(session)
  local label = "east_to_glass"
  ctrl.require_room(session, ROOM_EAST_TUNNEL, label)
  ctrl.unmorph(session)
  pcall(ctrl.select_weapon, session, 0)
  local i
  for i = 1, 48 do
    local state = ctrl.hold(session, 1, {}, label .. "_stand")
    local pose = ctrl.num(state.pose)
    if MORPH[pose] or CROUCH[pose] then
      ctrl.hold(session, 1, {"UP"}, label .. "_uncrouch")
    elseif ctrl.num(state.velocity_y) == 0 and STAND[pose] then
      break
    end
  end
  local frame
  local reached = false
  for frame = 0, geom.EAST_TO_GLASS_FRAMES - 1 do
    local state = session.state
    if ctrl.num(state.room_id) == ROOM_GLASS then
      reached = true
      break
    end
    if ctrl.num(state.room_id) ~= ROOM_EAST_TUNNEL then
      break
    end
    if ctrl.is_knockback(state) then
      ctrl.escape_kb(session, {prefer_dir = "LEFT", label = label, stop_room_id = ROOM_GLASS})
    else
      local pose = ctrl.num(state.pose)
      if MORPH[pose] or CROUCH[pose] then
        ctrl.hold(session, 6, {"UP"}, label .. "_uncrouch")
      else
        local x = ctrl.x(state)
        if x <= geom.EAST_GLASS_DOOR_X and ctrl.num(state.velocity_y) == 0 then
          local phase = frame % 16
          if phase < 4 then
            ctrl.hold(session, 1, {"LEFT", "X"}, label .. "_door_shot")
          elseif phase < 12 then
            ctrl.hold(session, 1, {"LEFT", "B"}, label .. "_door_push")
          else
            ctrl.hold(session, 1, {"LEFT", "B", "A"}, label .. "_door_spin")
          end
        else
          local phase = frame % 20
          if phase < 12 then
            ctrl.hold(session, 1, {"LEFT", "B"}, label .. "_run")
          elseif phase < 16 then
            ctrl.hold(session, 1, {"LEFT", "B", "X"}, label .. "_shot")
          elseif phase < 18 then
            ctrl.hold(session, 1, {"LEFT", "B", "A"}, label .. "_hop")
          else
            ctrl.hold(session, 1, {"LEFT"}, label .. "_walk")
          end
        end
      end
    end
    if ctrl.num(session.state.room_id) == ROOM_GLASS then
      reached = true
      break
    end
  end
  if not reached and ctrl.num(session.state.room_id) ~= ROOM_GLASS then
    ctrl.timeout(label .. ": left East door missed; " .. ctrl.brief(session.state))
  end
  return ctrl.wait_ordinary_room(session, ROOM_GLASS, {
    settle_frames = geom.EAST_TO_GLASS_SETTLE,
    label = label,
  })
end

return M
