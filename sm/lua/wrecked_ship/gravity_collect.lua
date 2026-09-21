-- Post-Phantoon Attic→Gravity traversal + Gravity Suit collect.
-- s23 / gravity_path_v2 hop bodies are Lua tables under wrecked_ship.tapes.

local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")
local enemies_mod = require("enemies")
local knockback = require("skills.knockback")
local charge_shot = require("skills.charge_shot")

local M = {}
local ROOM_WS_ATTIC = rooms.ROOM_WS_ATTIC or 0xCA52
local ROOM_WEST_OCEAN = rooms.ROOM_WEST_OCEAN or 0x93FE
local ROOM_PANCAKES = rooms.ROOM_PANCAKES or 0x9461
local ROOM_HOMING_GEEMER = rooms.ROOM_HOMING_GEEMER or 0x968F
local ROOM_BOWLING = rooms.ROOM_BOWLING or 0xC98E
local ROOM_GRAVITY = rooms.ROOM_GRAVITY or 0xCE40
local GRAVITY_MASK = ctrl.GRAVITY_MASK
local ATTIC_KIHUNTER_ID, ATTIC_KIHUNTER_WINGS_ID, ATTIC_ATOMIC_ID = 0xEB3F, 0xEB7F, 0xE9FF
local ATTIC_COMBAT_BUDGET = 4200
local ATTIC_POWER_BOMB_FUSE = 130
local ATTIC_WEST_OCEAN_SETTLE = 420
local WEST_OCEAN_ENTRY_RUN_FRAMES = 8
local WEST_OCEAN_PANCAKES_SETTLE = 240
local PANCAKES_HOMING_GEEMER_SETTLE = 240
local HOMING_GEEMER_BOWLING_SETTLE = 240
local BOWLING_GRAVITY_SETTLE = 240
M.BOWLING_FLOOR_Y = 512
M.BOWLING_RIDE_BOT_ID = 0xF0FF
M.LAST_BOWLING_DROP = nil

local function arrived(state, dest_room)
  return ctrl.num(state.room_id) == dest_room
    and ctrl.num(state.game_state) == 8
    and ctrl.num(state.door_transition) == 0
end

local function has_gravity(state)
  return ctrl.band(ctrl.num(state.collected_items), GRAVITY_MASK) ~= 0
end

function M.is_bowling_floor_drop(state, prev_y)
  if ctrl.num(state.room_id) ~= ROOM_BOWLING then
    return false
  end
  if not ctrl.is_morph(state.pose) then
    return false
  end
  return prev_y < M.BOWLING_FLOOR_Y and ctrl.y(state) >= M.BOWLING_FLOOR_Y
end

local function attic_required(list)
  local out, i = {}, 1
  for i = 1, #(list or {}) do
    local e = list[i]
    local eid = ctrl.num(e.enemy_id or e.id)
    if (eid == ATTIC_KIHUNTER_ID or eid == ATTIC_KIHUNTER_WINGS_ID or eid == ATTIC_ATOMIC_ID)
        and 24 < ctrl.num(e.x) and ctrl.num(e.x) < 1700
        and 32 < ctrl.num(e.y) and ctrl.num(e.y) < 260 then
      out[#out + 1] = e
    end
  end
  return out
end

local function play_s23_to_room(session, label, start_room, dest_room, body, settle)
  ctrl.require_room(session, start_room, label)
  if arrived(session.state, dest_room) then
    return session.state
  end
  ctrl.play_script(session, body, {
    reason = label .. "_tape",
    stop_when = function(st)
      return arrived(st, dest_room)
    end,
  })
  local i
  for i = 1, settle or 120 do
    if arrived(session.state, dest_room) then
      return session.state
    end
    ctrl.hold(session, 1, {}, label .. "_settle")
  end
  if not arrived(session.state, dest_room) then
    ctrl.timeout(string.format(
      "%s: expected 0x%04X gs=8, got %s", label, dest_room, ctrl.brief(session.state)
    ))
  end
  return session.state
end

local function settled_west_ocean_entry(state)
  local pose = ctrl.num(state.pose)
  return arrived(state, ROOM_WEST_OCEAN)
    and (pose == 1 or pose == 2 or pose == 9 or pose == 10)
    and math.abs(ctrl.num(state.velocity_y)) <= 1
    and 120 <= ctrl.y(state) and ctrl.y(state) <= 160
end

function M.play_attic_to_west_ocean(session)
  local label = "attic_to_west_ocean"
  ctrl.require_room(session, ROOM_WS_ATTIC, label)
  ctrl.hold(session, 12, {}, label .. "_settle")
  pcall(ctrl.select_weapon, session, 3)
  ctrl.ensure_morph(session)
  ctrl.hold(session, 8, {"X"}, label .. "_power_bomb")
  ctrl.hold(session, ATTIC_POWER_BOMB_FUSE, {}, label .. "_power_bomb_fuse")
  ctrl.unmorph(session)
  local i
  for i = 1, ATTIC_COMBAT_BUDGET do
    local state = session.state
    if settled_west_ocean_entry(state) then
      return state
    end
    if ctrl.num(state.room_id) == ROOM_WEST_OCEAN then
      local j
      for j = 1, ATTIC_WEST_OCEAN_SETTLE do
        if settled_west_ocean_entry(session.state) then
          return session.state
        end
        ctrl.hold(session, 1, {}, label .. "_door_settle")
      end
      ctrl.timeout(label .. ": West Ocean entry did not ground after the door: " .. ctrl.brief(session.state))
    end
    if ctrl.num(state.room_id) ~= ROOM_WS_ATTIC then
      ctrl.timeout(label .. ": unexpected room " .. ctrl.brief(state))
    end
    local list = attic_required(enemies_mod.list and enemies_mod.list(session) or enemies_mod.list_enemies and enemies_mod.list_enemies(session) or {})
    if ctrl.is_knockback(state) then
      local dir, target = "LEFT", nil
      local k, best
      for k = 1, #list do
        local d = math.abs(ctrl.num(list[k].x) - ctrl.x(state))
        if not best or d < best then
          best, target = d, list[k]
        end
      end
      if target and ctrl.num(target.x) >= ctrl.x(state) then
        dir = "RIGHT"
      end
      ctrl.escape_kb(session, {prefer_dir = dir, run_frames = 4, spin_frames = 16, label = label})
    elseif #list == 0 then
      ctrl.hold(session, 1, {"LEFT", "B", "X"}, label .. "_exit")
    else
      local choice
      if enemies_mod.choose then
        choice = enemies_mod.choose(
          ctrl.x(state), ctrl.y(state), ctrl.num(state.facing), list,
          {engage = {[ATTIC_KIHUNTER_ID]=true,[ATTIC_KIHUNTER_WINGS_ID]=true,[ATTIC_ATOMIC_ID]=true}},
          {
            movement_type = ctrl.num(state.movement_type),
            charge = charge_shot.session_beam_charge and charge_shot.session_beam_charge(session) or 0,
            velocity_y = ctrl.num(state.velocity_y),
            fire_range_px = 96,
          }
        )
      end
      local buttons = (choice and choice.buttons) or {"LEFT", "B"}
      if buttons and buttons[1] then
        ctrl.hold(session, 1, buttons, label .. "_engage")
      else
        ctrl.hold(session, 1, {}, label .. "_engage_wait")
      end
    end
  end
  ctrl.timeout(label .. ": kill-all/left-door budget exhausted in " .. ctrl.brief(session.state))
end

function M.play_west_ocean_to_pancakes(session)
  local label = "west_ocean_to_pancakes"
  ctrl.require_room(session, ROOM_WEST_OCEAN, label)
  if arrived(session.state, ROOM_PANCAKES) then
    return session.state
  end
  ctrl.hold(session, WEST_OCEAN_ENTRY_RUN_FRAMES, {"LEFT", "B"}, label .. "_entry_run")
  local body = require("wrecked_ship.tapes.west_ocean_v2")
  -- Skip the 8-frame entry run already played (first runs of the body).
  local skipped, used, sliced, i = 0, 0, {}, 1
  for i = 1, #body do
    if skipped < WEST_OCEAN_ENTRY_RUN_FRAMES then
      local take = math.min(body[i][1], WEST_OCEAN_ENTRY_RUN_FRAMES - skipped)
      skipped = skipped + take
      local rest = body[i][1] - take
      if rest > 0 then
        sliced[#sliced + 1] = {rest, body[i][2]}
      end
    else
      sliced[#sliced + 1] = body[i]
    end
  end
  return play_s23_to_room(session, label, ROOM_WEST_OCEAN, ROOM_PANCAKES, sliced, WEST_OCEAN_PANCAKES_SETTLE)
end

function M.play_pancakes_to_homing_geemer(session)
  return play_s23_to_room(
    session, "pancakes_to_homing_geemer", ROOM_PANCAKES, ROOM_HOMING_GEEMER,
    require("wrecked_ship.tapes.pancakes"), PANCAKES_HOMING_GEEMER_SETTLE
  )
end

function M.play_homing_geemer_to_bowling(session)
  return play_s23_to_room(
    session, "homing_geemer_to_bowling", ROOM_HOMING_GEEMER, ROOM_BOWLING,
    require("wrecked_ship.tapes.homing_geemer"), HOMING_GEEMER_BOWLING_SETTLE
  )
end

function M.play_bowling_to_gravity(session)
  M.LAST_BOWLING_DROP = nil
  local label = "bowling_to_gravity"
  ctrl.require_room(session, ROOM_BOWLING, label)
  if arrived(session.state, ROOM_GRAVITY) then
    return session.state
  end
  local body = require("wrecked_ship.tapes.bowling_v2")
  local prev_y = ctrl.y(session.state)
  local function maybe_drop(state)
    if M.LAST_BOWLING_DROP == nil and M.is_bowling_floor_drop(state, prev_y) then
      M.LAST_BOWLING_DROP = {
        frame = ctrl.num(session.frame),
        room = string.format("0x%04X", ctrl.num(state.room_id)),
        xy = {ctrl.x(state), ctrl.y(state)},
        prev_y = prev_y,
        pose = ctrl.num(state.pose),
        gs = ctrl.num(state.game_state, 8),
      }
    end
  end
  ctrl.play_script(session, body, {
    reason = label .. "_tape",
    stop_when = function(st)
      maybe_drop(st)
      prev_y = ctrl.y(st)
      return arrived(st, ROOM_GRAVITY)
    end,
  })
  local i
  for i = 1, BOWLING_GRAVITY_SETTLE do
    if arrived(session.state, ROOM_GRAVITY) then
      return session.state
    end
    ctrl.hold(session, 1, {}, label .. "_settle")
    maybe_drop(session.state)
    prev_y = ctrl.y(session.state)
  end
  if not arrived(session.state, ROOM_GRAVITY) then
    ctrl.timeout(label .. ": expected 0xCE40 gs=8, got " .. ctrl.brief(session.state))
  end
  return session.state
end

function M.play_gravity_collect(session)
  ctrl.require_room(session, ROOM_GRAVITY, "gravity_collect")
  if has_gravity(session.state) then
    return session.state
  end
  ctrl.play_script(session, require("wrecked_ship.tapes.gravity"), {
    reason = "gravity_tape",
    stop_when = has_gravity,
  })
  if not has_gravity(session.state) then
    ctrl.timeout(string.format(
      "gravity_collect: items still 0x%04X after tape: %s",
      ctrl.num(session.state.collected_items), ctrl.brief(session.state)
    ))
  end
  return session.state
end

function M.require_gravity_collected(session)
  if not has_gravity(session.state) then
    ctrl.timeout(string.format(
      "Gravity not collected: items=0x%04X", ctrl.num(session.state.collected_items)
    ))
  end
end

return M
