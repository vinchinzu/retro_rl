-- Powered Wrecked Ship Basement → Main Shaft. Morph-roll LEFT, jump UP hatch.

local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")
local fight = require("wrecked_ship.phantoon_fight")
local ice = require("wrecked_ship.ws_basement_ice")
local ceiling = require("wrecked_ship.ws_ceiling_door")
local enemies_mod = require("enemies")
local takeoff = require("takeoff")
local charge_shot = require("skills.charge_shot")

local M = {}
local ROOM_PHANTOON = rooms.ROOM_PHANTOON or 0xCD13
local ROOM_WS_BASEMENT = rooms.ROOM_WS_BASEMENT or 0xCC6F
local ROOM_WS_MAIN = rooms.ROOM_WS_MAIN or 0xCAF6
local ROOM_WS_MAP = rooms.ROOM_WS_MAP or 0xCCCB
local FLOOR_Y, TUNNEL_LIP_X, TUNNEL_CLEAR_X = 170, 1140, 900
local HATCH_X_MIN, HATCH_X_MAX, PLATFORM_X = 630, 690, 657
local PLATFORM_Y, SEAT_Y = 175, 168
local TAKEOFF_X_MIN, TAKEOFF_X_MAX, LATCH_SLACK = 750, 780, 16
local MAP_X, BOMB_CYCLES, SETTLE = 80, 8, 200
local DROP_BUDGET, TUNNEL_ROLL, RUN_BUDGET = 240, 160, 2400
local AIR = {[19]=true,[20]=true,[21]=true,[25]=true,[26]=true,[47]=true,[48]=true,[81]=true,[82]=true,[83]=true,[84]=true}
local FACING_LEFT = ctrl.FACING_LEFT

function M.ws_basement_main_settled(state)
  return ctrl.num(state.room_id) == ROOM_WS_MAIN
    and ctrl.num(state.game_state) == 8
    and ctrl.num(state.door_transition) == 0
end

function M.at_ws_basement_hatch_seat(state)
  local pose = ctrl.num(state.pose)
  return ctrl.num(state.room_id) == ROOM_WS_BASEMENT
    and math.abs(ctrl.x(state) - PLATFORM_X) <= 16
    and ctrl.y(state) <= PLATFORM_Y
    and (pose == 1 or pose == 2 or pose == 9 or pose == 10)
    and math.abs(ctrl.num(state.velocity_y)) <= 1
end

local function spin_jump(dir)
  if takeoff.spin_jump then
    return takeoff.spin_jump(dir)
  end
  return {dir, "B", "A"}
end

local function walk_toward_x(x, target, slack)
  if takeoff.walk_toward_x then
    return takeoff.walk_toward_x(x, target, slack)
  end
  if x < target - slack then return {"RIGHT"} end
  if x > target + slack then return {"LEFT"} end
  return {}
end

function M.hatch_jump_action(samus_x, samus_y, pose, frame)
  local names = ceiling.ceiling_door_action(samus_x, samus_y, pose, frame, {
    seat_x = PLATFORM_X, lip_y = 140, shaft_y = 80, slack = 8, hold_charge = false,
  })
  if names ~= nil then
    return names
  end
  if samus_x < PLATFORM_X - 8 then
    return {"RIGHT", "B"}
  end
  if samus_x > PLATFORM_X + 8 then
    return {"LEFT", "B"}
  end
  return ceiling.tap_up_action(frame, true)
end

function M.hatch_mount_action(samus_x, samus_y, pose, velocity_y, facing, movement_type)
  if pose == 137 or pose == 138 then
    return {}
  end
  if samus_y <= SEAT_Y and math.abs(velocity_y) <= 1 then
    return walk_toward_x(samus_x, PLATFORM_X, 8)
  end
  if AIR[pose] then
    if facing == FACING_LEFT then
      return spin_jump("LEFT")
    end
    return {}
  end
  if samus_x > TAKEOFF_X_MAX then
    return {"LEFT", "B"}
  end
  local in_band = samus_x >= TAKEOFF_X_MIN
  local latched = samus_x >= TAKEOFF_X_MIN - LATCH_SLACK and (facing == FACING_LEFT or movement_type == 14)
  if in_band or latched then
    if facing ~= FACING_LEFT or movement_type == 14 then
      return {"LEFT"}
    end
    return spin_jump("LEFT")
  end
  return {"RIGHT", "B"}
end

local function guard(session, label)
  local room = ctrl.num(session.state.room_id)
  if room == ROOM_WS_MAIN then
    return
  end
  if room == ROOM_WS_MAP then
    ctrl.timeout(label .. ": entered map 0xCCCB")
  end
  if room == ROOM_PHANTOON then
    ctrl.timeout(label .. ": back into Phantoon 0xCD13")
  end
  if room ~= ROOM_WS_BASEMENT then
    ctrl.timeout(label .. ": left Basement into 0x" .. string.format("%04X", room))
  end
  if ctrl.x(session.state) < MAP_X then
    ctrl.timeout(label .. ": walked into left map door")
  end
end

local function kb(session, label)
  ctrl.escape_kb(session, {
    prefer_dir = "LEFT", run_frames = 6, spin_frames = 24,
    label = label, stop_room_id = ROOM_WS_MAIN,
  })
end

local function drop_to_floor(session, label)
  if ctrl.num(session.state.room_id) == ROOM_WS_MAIN or ctrl.is_morph(session.state.pose) then
    return
  end
  ctrl.hold(session, 8, {"LEFT"}, label .. "_turn")
  local i
  for i = 1, DROP_BUDGET do
    local st = session.state
    guard(session, label)
    if ctrl.num(st.room_id) == ROOM_WS_MAIN then
      return
    end
    if ctrl.is_knockback(st) then
      kb(session, label .. "_drop_kb")
    elseif ctrl.y(st) >= FLOOR_Y and ctrl.x(st) <= TUNNEL_LIP_X + 20
        and ctrl.num(st.velocity_y) == 0 and not ctrl.is_morph(st.pose) then
      return
    else
      ctrl.hold(session, 1, {"LEFT", "B"}, label .. "_drop")
    end
  end
  if ctrl.y(session.state) < FLOOR_Y - 20 then
    ctrl.timeout(label .. ": did not drop to tunnel floor: " .. ctrl.brief(session.state))
  end
end

local function bomb_tunnel_left(session, label)
  if ctrl.num(session.state.room_id) == ROOM_WS_MAIN or ctrl.x(session.state) <= TUNNEL_CLEAR_X then
    return
  end
  ctrl.ensure_morph(session)
  local cycle
  for cycle = 1, BOMB_CYCLES do
    local st = session.state
    guard(session, label)
    if ctrl.num(st.room_id) == ROOM_WS_MAIN or ctrl.x(st) <= TUNNEL_CLEAR_X then
      return
    end
    local prev, stall, i = ctrl.x(st), 0, 1
    local broke = false
    for i = 1, TUNNEL_ROLL do
      st = session.state
      guard(session, label)
      if ctrl.num(st.room_id) == ROOM_WS_MAIN or ctrl.x(st) <= TUNNEL_CLEAR_X then
        return
      end
      if ctrl.is_knockback(st) then
        kb(session, label .. "_roll_kb")
        stall = 0
      else
        if not ctrl.is_morph(st.pose) then
          ctrl.ensure_morph(session)
        end
        ctrl.hold(session, 1, {"LEFT"}, label .. "_roll")
        local x = ctrl.x(session.state)
        if math.abs(x - prev) < 2 then stall = stall + 1 else stall = 0 end
        prev = x
        if stall >= 16 then
          ctrl.hold(session, 3, {"X"}, label .. "_bomb")
          ctrl.hold(session, 80, {}, label .. "_boom")
          broke = true
          break
        end
      end
    end
    if not broke and ctrl.x(session.state) <= TUNNEL_CLEAR_X then
      return
    end
  end
  if ctrl.num(session.state.room_id) ~= ROOM_WS_MAIN and ctrl.x(session.state) > TUNNEL_CLEAR_X then
    ctrl.timeout(label .. ": morph tunnel did not clear left: " .. ctrl.brief(session.state))
  end
end

local function run_to_hatch(session, label)
  if ctrl.num(session.state.room_id) == ROOM_WS_MAIN then
    return
  end
  if ctrl.is_morph(session.state.pose) then
    ctrl.unmorph(session)
  end
  pcall(ctrl.select_weapon, session, 0)
  local i
  for i = 1, RUN_BUDGET do
    local st = session.state
    guard(session, label)
    if ctrl.num(st.room_id) == ROOM_WS_MAIN or M.at_ws_basement_hatch_seat(st) then
      return
    end
    if ctrl.is_knockback(st) then
      kb(session, label .. "_run_kb")
    elseif ctrl.is_morph(st.pose) then
      ctrl.unmorph(session)
      ctrl.hold(session, 8, {"UP"}, label .. "_unmorph")
    else
      local list = ice.basement_overlay_targets(
        ctrl.x(st), ctrl.y(st),
        (enemies_mod.list and enemies_mod.list(session)) or (enemies_mod.list_enemies and enemies_mod.list_enemies(session)) or {}
      )
      local buttons
      if enemies_mod.choose then
        local choice = enemies_mod.choose(
          ctrl.x(st), ctrl.y(st), ctrl.num(st.facing), list,
          {engage = {[ice.ATOMIC_ID]=true}, absorb = {[ice.COVERN_ID]=true}, avoid = {[ice.WORKROBOT_ID]=true}},
          {
            movement_type = ctrl.num(st.movement_type),
            charge = charge_shot.session_beam_charge and charge_shot.session_beam_charge(session) or 0,
            velocity_y = ctrl.num(st.velocity_y),
            takeoff_x_min = TAKEOFF_X_MIN,
            clamp_solids = true,
          }
        )
        buttons = choice and choice.buttons
      end
      if buttons ~= nil then
        if buttons[1] then
          ctrl.hold(session, 1, buttons, label .. "_engage")
        else
          ctrl.hold(session, 1, {}, label .. "_engage_wait")
        end
      else
        local names = M.hatch_mount_action(
          ctrl.x(st), ctrl.y(st), ctrl.num(st.pose), ctrl.num(st.velocity_y),
          ctrl.num(st.facing), ctrl.num(st.movement_type)
        )
        if names[1] then
          ctrl.hold(session, 1, names, label .. "_mount")
        else
          ctrl.hold(session, 1, {}, label .. "_wait")
        end
      end
    end
  end
  if ctrl.num(session.state.room_id) ~= ROOM_WS_MAIN and not M.at_ws_basement_hatch_seat(session.state) then
    ctrl.timeout(label .. ": did not reach ceiling hatch: " .. ctrl.brief(session.state))
  end
end

function M.play_ws_basement_to_main(session)
  local label = "ws_basement_to_main"
  if M.ws_basement_main_settled(session.state) then
    return session.state
  end
  ctrl.require_room(session, ROOM_WS_BASEMENT, label)
  if not fight.phantoon_boss_bit_set(session) then
    ctrl.timeout(label .. ": Phantoon not defeated: " .. ctrl.brief(session.state))
  end
  drop_to_floor(session, label)
  bomb_tunnel_left(session, label)
  run_to_hatch(session, label)
  if ctrl.num(session.state.room_id) ~= ROOM_WS_MAIN then
    pcall(ctrl.select_weapon, session, 0)
    ceiling.play_ceiling_door(session, {
      label = label,
      dest_room = ROOM_WS_MAIN,
      lip_y = PLATFORM_Y,
      remount = function(st)
        return M.hatch_mount_action(
          ctrl.x(st), ctrl.y(st), ctrl.num(st.pose), ctrl.num(st.velocity_y),
          ctrl.num(st.facing), ctrl.num(st.movement_type)
        )
      end,
      door_action = function(st, i)
        return M.hatch_jump_action(ctrl.x(st), ctrl.y(st), ctrl.num(st.pose), i)
      end,
      guard = guard,
      on_knockback = kb,
    })
  end
  return ceiling.settle_ceiling_dest(session, ROOM_WS_MAIN, {label = label, settle_frames = SETTLE})
end

return M
