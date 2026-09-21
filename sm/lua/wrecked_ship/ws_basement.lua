-- Wrecked Ship Basement → Phantoon's Room. Morph-bomb tunnel (X), Super Gadora.

local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")

local M = {}
local ROOM_PHANTOON = rooms.ROOM_PHANTOON or 0xCD13
local ROOM_WS_BASEMENT = rooms.ROOM_WS_BASEMENT or 0xCC6F
local ROOM_WS_MAIN = rooms.ROOM_WS_MAIN or 0xCAF6
local ROOM_WS_MAP = rooms.ROOM_WS_MAP or 0xCCCB
local WEAPON_SUPER = 2
local FLOOR_Y, MORPH_X_MIN, BOMB_X_MIN = 170, 930, 1000
local ALCOVE_X, MAP_X = 1160, 80
local FALLING = {[23]=true,[24]=true,[25]=true,[26]=true}
local RUN_BUDGET, BOMB_CYCLES, EYE_BUDGET, SETTLE = 420, 6, 900, 200

function M.ws_basement_phantoon_settled(state)
  return ctrl.num(state.room_id) == ROOM_PHANTOON
    and ctrl.num(state.game_state) == 8
    and ctrl.num(state.door_transition) == 0
end

local function guard(session, label)
  local room = ctrl.num(session.state.room_id)
  if room == ROOM_PHANTOON then
    return
  end
  if room == ROOM_WS_MAP then
    ctrl.timeout(label .. ": entered map 0xCCCB: " .. ctrl.brief(session.state))
  end
  if room == ROOM_WS_MAIN then
    ctrl.timeout(label .. ": back to Main Shaft 0xCAF6: " .. ctrl.brief(session.state))
  end
  if room ~= ROOM_WS_BASEMENT then
    ctrl.timeout(label .. ": left Basement into 0x" .. string.format("%04X", room))
  end
  if ctrl.x(session.state) < MAP_X then
    ctrl.timeout(label .. ": walked into left map door: " .. ctrl.brief(session.state))
  end
end

local function land(session, label)
  local i
  for i = 1, 80 do
    local st = session.state
    guard(session, label)
    local pose = ctrl.num(st.pose)
    if not FALLING[pose] and ctrl.num(st.velocity_y) == 0 and not ctrl.is_knockback(st) then
      return
    end
    ctrl.hold(session, 1, {}, label .. "_land")
  end
end

local function kb(session, label)
  ctrl.escape_kb(session, {
    prefer_dir = "RIGHT", run_frames = 6, spin_frames = 24,
    label = label, stop_room_id = ROOM_PHANTOON,
  })
end

local function run_to_morph(session, label)
  ctrl.hold(session, 12, {"RIGHT"}, label .. "_turn")
  local i
  for i = 1, RUN_BUDGET do
    local st = session.state
    guard(session, label)
    if ctrl.num(st.room_id) == ROOM_PHANTOON then
      return
    end
    if ctrl.is_knockback(st) then
      kb(session, label .. "_run_kb")
    else
      local x, y = ctrl.x(st), ctrl.y(st)
      if y >= FLOOR_Y and x >= MORPH_X_MIN and ctrl.num(st.velocity_y) == 0 and not ctrl.is_morph(st.pose) then
        return
      end
      ctrl.hold(session, 1, {"RIGHT", "B"}, label .. "_run")
    end
  end
  ctrl.timeout(label .. ": did not reach morph-tunnel floor: " .. ctrl.brief(session.state))
end

local function bomb_tunnel(session, label)
  if ctrl.num(session.state.room_id) == ROOM_PHANTOON then
    return
  end
  ctrl.ensure_morph(session)
  local cycle
  for cycle = 1, BOMB_CYCLES do
    local st = session.state
    guard(session, label)
    if ctrl.num(st.room_id) == ROOM_PHANTOON or ctrl.x(st) >= ALCOVE_X then
      return
    end
    local prev, stall, rolled = ctrl.x(st), 0, false
    local i
    local broke = false
    for i = 1, 160 do
      st = session.state
      guard(session, label)
      if ctrl.num(st.room_id) == ROOM_PHANTOON or ctrl.x(st) >= ALCOVE_X then
        return
      end
      if ctrl.is_knockback(st) then
        kb(session, label .. "_roll_kb")
        stall = 0
      else
        if not ctrl.is_morph(st.pose) then
          ctrl.ensure_morph(session)
        end
        ctrl.hold(session, 1, {"RIGHT"}, label .. "_roll")
        rolled = true
        local x = ctrl.x(session.state)
        if math.abs(x - prev) < 2 then
          stall = stall + 1
        else
          stall = 0
        end
        prev = x
        if stall >= 16 then
          -- Morph bombs are X, not A.
          ctrl.hold(session, 3, {"X"}, label .. "_bomb")
          ctrl.hold(session, 80, {}, label .. "_boom")
          broke = true
          break
        end
      end
    end
    if not broke then
      if rolled and ctrl.x(session.state) >= ALCOVE_X then
        return
      end
      break
    end
  end
  if ctrl.num(session.state.room_id) == ROOM_PHANTOON then
    return
  end
  if ctrl.x(session.state) < ALCOVE_X then
    ctrl.timeout(label .. ": morph tunnel did not reach alcove: " .. ctrl.brief(session.state))
  end
end

local function super_eye(session, label)
  if ctrl.num(session.state.room_id) == ROOM_PHANTOON then
    return
  end
  ctrl.unmorph(session)
  pcall(ctrl.select_weapon, session, WEAPON_SUPER)
  local frame
  for frame = 0, EYE_BUDGET - 1 do
    local st = session.state
    guard(session, label)
    if ctrl.num(st.room_id) == ROOM_PHANTOON then
      return
    end
    if ctrl.is_knockback(st) then
      kb(session, label .. "_eye_kb")
    elseif ctrl.is_morph(st.pose) then
      ctrl.unmorph(session)
    else
      local phase = frame % 28
      if phase < 4 then
        ctrl.hold(session, 1, {"RIGHT", "X"}, label .. "_super")
      elseif phase >= 16 then
        ctrl.hold(session, 1, {"RIGHT", "B", "A"}, label .. "_spin")
      else
        ctrl.hold(session, 1, {"RIGHT", "B"}, label .. "_run")
      end
    end
  end
  if ctrl.num(session.state.room_id) ~= ROOM_PHANTOON then
    ctrl.timeout(label .. ": Gadora / blue door missed: " .. ctrl.brief(session.state))
  end
end

function M.play_ws_basement_to_phantoon(session)
  local label = "ws_basement_to_phantoon"
  if M.ws_basement_phantoon_settled(session.state) then
    return session.state
  end
  ctrl.require_room(session, ROOM_WS_BASEMENT, label)
  land(session, label)
  run_to_morph(session, label)
  bomb_tunnel(session, label)
  super_eye(session, label)
  return ctrl.wait_ordinary_room(session, ROOM_PHANTOON, {settle_frames = SETTLE, label = label})
end

return M
