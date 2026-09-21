-- Powered Wrecked Ship Main Shaft → Attic.

local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")
local fight = require("wrecked_ship.phantoon_fight")
local geom = require("wrecked_ship.ws_main_geometry")
local actions = require("wrecked_ship.ws_main_actions")
local ceiling = require("wrecked_ship.ws_ceiling_door")
local charge_shot = require("skills.charge_shot")

local M = {}
local ROOM_WS_MAIN = rooms.ROOM_WS_MAIN or 0xCAF6
local ROOM_WS_ATTIC = rooms.ROOM_WS_ATTIC or 0xCA52
local ROOM_WS_BASEMENT = rooms.ROOM_WS_BASEMENT or 0xCC6F
local ROOM_WS_SAVE = rooms.ROOM_WS_SAVE or 0xCE8A
local ROOM_WS_WEST_SUPER = rooms.ROOM_WS_WEST_SUPER or 0xCDA8
local ROOM_WS_SPONGE = rooms.ROOM_WS_SPONGE or 0xCD5C
local CLIMB_BUDGET, SIDE_TRIP_BUDGET = 3600, 400
local THREE_SHOT_FRAMES = geom.THREE_SHOT_FRAMES + 40

function M.ws_main_attic_settled(state)
  return geom.ws_main_attic_settled(state)
end

function M.guard_main_shaft(session, label)
  local room = ctrl.num(session.state.room_id)
  if room == ROOM_WS_MAIN or room == ROOM_WS_ATTIC or room == ROOM_WS_WEST_SUPER or room == ROOM_WS_SPONGE then
    return
  end
  if room == ROOM_WS_SAVE then
    ctrl.timeout(label .. ": entered save 0xCE8A: " .. ctrl.brief(session.state))
  end
  if room == ROOM_WS_BASEMENT then
    ctrl.timeout(label .. ": dropped back to Basement 0xCC6F: " .. ctrl.brief(session.state))
  end
  ctrl.timeout(label .. ": left Main Shaft into 0x" .. string.format("%04X", room))
end

function M.knockback_main_shaft(session, label)
  local prefer = (ctrl.x(session.state) > geom.WS_MAIN_SHAFT_CENTER) and "LEFT" or "RIGHT"
  ctrl.escape_kb(session, {
    prefer_dir = prefer, run_frames = 6, spin_frames = 24,
    label = label, stop_room_id = ROOM_WS_ATTIC,
  })
end

function M.exit_side_room(session, label)
  local i
  for i = 1, SIDE_TRIP_BUDGET do
    local st = session.state
    local room = ctrl.num(st.room_id)
    if room == ROOM_WS_MAIN or room == ROOM_WS_ATTIC then
      return
    end
    M.guard_main_shaft(session, label)
    if ctrl.is_knockback(st) then
      M.knockback_main_shaft(session, label .. "_side_kb")
    elseif ctrl.is_morph(st.pose) then
      ctrl.unmorph(session)
    elseif room == ROOM_WS_WEST_SUPER then
      ctrl.hold(session, 1, {"RIGHT", "B"}, label .. "_west_super")
    else
      ctrl.hold(session, 1, {"LEFT", "B"}, label .. "_sponge")
    end
  end
  if ctrl.num(session.state.room_id) ~= ROOM_WS_MAIN and ctrl.num(session.state.room_id) ~= ROOM_WS_ATTIC then
    ctrl.timeout(label .. ": side room did not return: " .. ctrl.brief(session.state))
  end
end

function M.three_shot_tunnel(session, label)
  if ctrl.num(session.state.room_id) == ROOM_WS_ATTIC or ctrl.num(session.state.room_id) == ROOM_WS_WEST_SUPER then
    return
  end
  if geom.at_ws_main_grate_seat(session.state) or not geom.at_ws_main_pit(session.state) then
    return
  end
  pcall(ctrl.select_weapon, session, 0)
  local shot_i, i = 0, 1
  for i = 1, THREE_SHOT_FRAMES do
    local st = session.state
    M.guard_main_shaft(session, label)
    if ctrl.num(st.room_id) == ROOM_WS_ATTIC or ctrl.num(st.room_id) == ROOM_WS_WEST_SUPER then
      return
    end
    if geom.at_ws_main_grate_seat(st) or not geom.at_ws_main_pit(st) then
      return
    end
    if ctrl.is_knockback(st) then
      M.knockback_main_shaft(session, label .. "_shot_kb")
    elseif ctrl.is_morph(st.pose) then
      ctrl.unmorph(session)
    else
      local names = actions.three_shot_action(
        ctrl.x(st), ctrl.y(st), ctrl.num(st.pose), ctrl.num(st.facing),
        shot_i, charge_shot.session_beam_charge and charge_shot.session_beam_charge(session) or 0,
        ctrl.num(st.movement_type), ctrl.num(st.velocity_y)
      )
      shot_i = shot_i + 1
      if names and names[1] then
        ctrl.hold(session, 1, names, label .. "_3shot")
      else
        ctrl.hold(session, 1, {}, label .. "_3shot_land")
      end
    end
  end
end

function M.at_attic_climb_done(state)
  return ctrl.num(state.room_id) == ROOM_WS_ATTIC or geom.at_ws_main_attic_door_seat(state)
end

function M.climb_until(session, label, done)
  local i
  for i = 1, CLIMB_BUDGET do
    local st = session.state
    M.guard_main_shaft(session, label)
    if done(st) or ctrl.num(st.room_id) == ROOM_WS_ATTIC then
      return
    end
    if ctrl.num(st.room_id) == ROOM_WS_WEST_SUPER or ctrl.num(st.room_id) == ROOM_WS_SPONGE then
      M.exit_side_room(session, label)
    elseif ctrl.is_knockback(st) then
      M.knockback_main_shaft(session, label .. "_kb")
    elseif ctrl.is_morph(st.pose) then
      ctrl.unmorph(session)
    else
      local names = actions.climb_action(
        ctrl.x(st), ctrl.y(st), ctrl.num(st.pose), ctrl.num(st.facing),
        ctrl.num(st.velocity_y), ctrl.num(st.movement_type),
        ctrl.num(session.frame), false,
        charge_shot.session_beam_charge and charge_shot.session_beam_charge(session) or 0
      )
      if names and names[1] then
        ctrl.hold(session, 1, names, label .. "_climb")
      else
        ctrl.hold(session, 1, {}, label .. "_wait")
      end
    end
  end
  ctrl.timeout(label .. ": climb timeout: " .. ctrl.brief(session.state))
end

function M.play_ws_main_to_attic(session, start, stop)
  start = start or "pit_shot"
  stop = stop or "attic_door"
  local label = "ws_main_to_attic"
  local start_i, stop_i = geom.ws_main_phase_index(start), geom.ws_main_phase_index(stop)
  if start_i > stop_i then
    error("start phase " .. tostring(start) .. " is after stop " .. tostring(stop))
  end
  if geom.ws_main_attic_settled(session.state) then
    return session.state
  end
  ctrl.require_room(session, ROOM_WS_MAIN, label)
  if not fight.phantoon_boss_bit_set(session) then
    ctrl.timeout(label .. ": Phantoon not defeated: " .. ctrl.brief(session.state))
  end
  if start_i <= 0 then
    M.three_shot_tunnel(session, label .. "_pit_shot")
  end
  if start_i <= 1 then
    M.climb_until(session, label .. "_grate_seat", geom.at_ws_main_usable_grate_seat)
    ctrl.hold(session, 8, {}, label .. "_grate_seat_settle")
  end
  if start_i <= 2 then
    M.climb_until(session, label .. "_west_super", geom.at_ws_main_west_super_band)
  end
  if start_i <= 3 then
    M.climb_until(session, label .. "_mid_climb", geom.at_ws_main_mid_climb)
  end
  if start_i <= 4 then
    M.climb_until(session, label .. "_attic_seat", M.at_attic_climb_done)
  end
  if ctrl.num(session.state.room_id) ~= ROOM_WS_ATTIC then
    pcall(ctrl.select_weapon, session, 0)
    ceiling.play_ceiling_door(session, {
      label = label,
      dest_room = ROOM_WS_ATTIC,
      lip_y = 160,
      remount = function(st)
        return actions.climb_action(
          ctrl.x(st), ctrl.y(st), ctrl.num(st.pose), ctrl.num(st.facing),
          ctrl.num(st.velocity_y), ctrl.num(st.movement_type),
          ctrl.num(session.frame), false, 0, geom.classify_region(st)
        )
      end,
      door_action = function(st, i)
        return actions.attic_door_action(ctrl.x(st), ctrl.y(st), ctrl.num(st.pose), i)
      end,
      guard = M.guard_main_shaft,
      on_knockback = M.knockback_main_shaft,
      side_rooms = {ROOM_WS_WEST_SUPER, ROOM_WS_SPONGE},
      on_side_room = M.exit_side_room,
    })
  end
  return ceiling.settle_ceiling_dest(session, ROOM_WS_ATTIC, {label = label})
end

return M
