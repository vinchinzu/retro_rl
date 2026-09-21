-- Phantoon loot pickup + left-door exit to WS Basement 0xCC6F gs=8.

local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")
local fight = require("wrecked_ship.phantoon_fight")

local M = {}
local ROOM_PHANTOON = rooms.ROOM_PHANTOON or 0xCD13
local ROOM_WS_BASEMENT = rooms.ROOM_WS_BASEMENT or 0xCC6F
M.DOOR_X_MAX = 70
M.SWEEP_X = 160
local WALL_HURT = {[83]=true,[84]=true,[109]=true,[137]=true,[138]=true,[143]=true,[158]=true,[159]=true,[160]=true}
local SETTLE = 240

function M.loot_walk_action(samus_x, target_x, swept)
  if target_x ~= nil then
    if target_x < samus_x - 6 then
      return {"LEFT", "B"}
    end
    if target_x > samus_x + 6 then
      return {"RIGHT", "B"}
    end
    return {}
  end
  if not swept and samus_x < M.SWEEP_X then
    return {"RIGHT", "B"}
  end
  return {}
end

function M.door_jump_action(samus_x, pose, frame)
  if WALL_HURT[pose] then
    return {}
  end
  if samus_x > M.DOOR_X_MAX then
    return {"LEFT", "B"}
  end
  local phase = frame % 36
  if phase < 20 then
    return {"LEFT", "A"}
  end
  if phase < 24 then
    return {"LEFT", "X"}
  end
  return {}
end

function M.require_phantoon_left(session)
  local st = session.state
  if ctrl.num(st.room_id) ~= ROOM_WS_BASEMENT then
    ctrl.timeout("phantoon_loot_exit: expected WS Basement 0xCC6F, got " .. ctrl.brief(st))
  end
  if not fight.phantoon_boss_bit_set(session) then
    ctrl.timeout("phantoon_loot_exit: Wrecked Ship $D82B bit 0 not set: " .. ctrl.brief(st))
  end
end

local function collect_loot(session, label)
  local enemies = require("enemies")
  local swept, i = false, 1
  for i = 1, 360 do
    local st = session.state
    if ctrl.num(st.room_id) ~= ROOM_PHANTOON then
      return
    end
    if WALL_HURT[ctrl.num(st.pose)] then
      ctrl.hold(session, 1, {}, label .. "_loot_hurt")
    elseif ctrl.is_morph(st.pose) then
      pcall(ctrl.unmorph, session)
    else
      local drops = {}
      if enemies.list_pickups then
        drops = enemies.list_pickups(session) or {}
      end
      local target
      if #drops > 0 then
        local best, d = drops[1], math.abs(ctrl.num(drops[1].x) - ctrl.x(st))
        local j
        for j = 2, #drops do
          local dd = math.abs(ctrl.num(drops[j].x) - ctrl.x(st))
          if dd < d then
            best, d = drops[j], dd
          end
        end
        target = ctrl.num(best.x)
      elseif ctrl.x(st) >= M.SWEEP_X then
        swept = true
      end
      local names = M.loot_walk_action(ctrl.x(st), target, swept)
      if not names[1] then
        return
      end
      ctrl.hold(session, 1, names, label .. "_loot")
    end
  end
end

function M.play_phantoon_loot_exit(session)
  local label = "phantoon_loot_exit"
  if ctrl.num(session.state.room_id) == ROOM_WS_BASEMENT
      and ctrl.num(session.state.game_state) == 8
      and ctrl.num(session.state.door_transition) == 0 then
    return session.state
  end
  ctrl.require_room(session, ROOM_PHANTOON, label)
  if not fight.phantoon_boss_bit_set(session) then
    ctrl.timeout(label .. ": Phantoon not defeated: " .. ctrl.brief(session.state))
  end
  if ctrl.is_morph(session.state.pose) then
    pcall(ctrl.unmorph, session)
  end
  collect_loot(session, label)
  local frame
  for frame = 0, 479 do
    local st = session.state
    if ctrl.num(st.room_id) == ROOM_WS_BASEMENT or ctrl.num(st.room_id) ~= ROOM_PHANTOON then
      break
    end
    local names = M.door_jump_action(ctrl.x(st), ctrl.num(st.pose), frame)
    if names[1] then
      ctrl.hold(session, 1, names, label .. "_door")
    else
      ctrl.hold(session, 1, {}, label .. "_hurt")
    end
  end
  if ctrl.num(session.state.room_id) ~= ROOM_WS_BASEMENT then
    ctrl.timeout(label .. ": left door missed: " .. ctrl.brief(session.state))
  end
  return ctrl.wait_ordinary_room(session, ROOM_WS_BASEMENT, {settle_frames = SETTLE, label = label})
end

return M
