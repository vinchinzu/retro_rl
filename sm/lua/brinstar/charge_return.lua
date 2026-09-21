-- Big Pink Charge Beam collect + conventional return (no IBJ).
-- Python: routes/kpdr/brinstar/charge_return.py

local ctrl = require("brinstar.ctrl")

local rooms = {}
pcall(function()
  rooms = require("rooms")
end)

local num = ctrl.num
local hold = ctrl.hold
local brief = ctrl.brief
local band = ctrl.band

local ROOM_BIG_PINK = rooms.ROOM_BIG_PINK or 0x9D19
local CHARGE_BEAM_MASK = 0x1000
local POSE_ITEM_FANFARE = 138
local MASS_Y_MAX = 1685

local M = {}
M.ROOM_BIG_PINK = ROOM_BIG_PINK
M.ROOM_CHARGE = ROOM_BIG_PINK
M.CHARGE_BEAM_MASK = CHARGE_BEAM_MASK

local function has_charge(state)
  return band(num(state.collected_beams), CHARGE_BEAM_MASK) ~= 0
end

local function wait_item_fanfare(session, reason)
  if num(session.state.pose) ~= POSE_ITEM_FANFARE then
    return session.state
  end
  local i
  for i = 1, 240 do
    local state = hold(session, 1, {}, reason)
    if num(state.pose) ~= POSE_ITEM_FANFARE then
      hold(session, 12, {}, reason .. "_settle")
      return session.state
    end
  end
  hold(session, 8, {"RIGHT"}, reason .. "_unstick")
  hold(session, 12, {}, reason .. "_unstick_settle")
  return session.state
end

function M._descend_main_to_mass(session)
  ctrl.require_room(session, ROOM_BIG_PINK, "charge_to_mass")
  local missiles_before = num(session.state.missiles)
  local max_before = num(session.state.max_missiles)
  ctrl.ensure_morph(session)

  local reached = false
  local i
  for i = 1, 500 do
    local state = hold(session, 1, {"LEFT"}, "charge_lower_left")
    if num(state.samus_x) <= 560 and num(state.samus_y) >= 1540 then
      reached = true
      break
    end
  end
  if not reached then
    error("charge_to_mass: missed lower-left shelf: " .. brief(session.state))
  end

  ctrl.unmorph(session)
  reached = false
  for i = 1, 220 do
    local state = hold(session, 1, {"RIGHT", "B", "A"}, "charge_lower_drop")
    if num(state.samus_x) >= 665 and num(state.samus_y) >= 1660 then
      reached = true
      break
    end
  end
  if not reached then
    error("charge_to_mass: missed lower mass: " .. brief(session.state))
  end

  if num(session.state.max_missiles) <= max_before and num(session.state.samus_x) > 560 then
    for i = 1, 50 do
      local state = hold(session, 1, {"LEFT"}, "charge_missile_walk")
      if num(state.missiles) > missiles_before or num(state.max_missiles) > max_before then
        break
      end
      if num(state.samus_x) <= 555 then
        break
      end
    end
    if num(session.state.missiles) > missiles_before
        or num(session.state.max_missiles) > max_before then
      wait_item_fanfare(session, "charge_missile_fanfare")
    end
    for i = 1, 80 do
      local state = hold(session, 1, {"RIGHT", "B"}, "charge_missile_return")
      if num(state.samus_x) >= 680 then
        break
      end
    end
  end

  hold(session, 30, {"RIGHT", "B"}, "charge_mass_run")
  hold(session, 10, {}, "charge_mass_settle")
  hold(session, 12, {"LEFT"}, "charge_mass_brake")
  hold(session, 8, {"A"}, "charge_mass_vertical")
  return session.state
end

local function bomb_drop_to_charge_floor(session)
  ctrl.ensure_morph(session)
  local i
  for i = 0, 399 do
    if (i % 25) == 0 then
      hold(session, 2, {"X"}, "charge_drop_bomb")
      hold(session, 40, {}, "charge_drop_fuse")
    end
    local direction
    if math.floor(i / 50) % 2 == 0 then
      direction = "LEFT"
    else
      direction = "RIGHT"
    end
    local state = hold(session, 1, {direction}, "charge_drop_roll")
    if num(state.samus_y) >= 1850 then
      return state
    end
  end
  error("charge_drop: never reached charge depth: " .. brief(session.state))
end

local GROUND_POSE = {
  [1] = true, [2] = true, [9] = true, [10] = true,
}

local function approach_chozo_platform(session)
  ctrl.unmorph(session)
  hold(session, 30, {}, "charge_drop_land")

  local i
  for i = 1, 50 do
    local state = hold(session, 1, {"RIGHT"}, "charge_runup")
    if num(state.samus_x) >= 690 then
      break
    end
  end
  hold(session, 15, {}, "charge_runup_settle")

  local cycle
  for cycle = 1, 8 do
    hold(session, 6, {"A"}, "charge_platform_hop")
    for i = 1, 30 do
      local state = hold(session, 1, {"LEFT"}, "charge_platform_drift")
      if has_charge(state) then
        return state
      end
      if GROUND_POSE[num(state.pose)]
          and num(state.samus_y) <= 1920
          and num(state.samus_x) <= 640 then
        hold(session, 25, {}, "charge_platform_settle")
        return session.state
      end
    end
    hold(session, 15, {}, "charge_platform_land")
  end
  error("charge_platform: never reached Chozo ledge: " .. brief(session.state))
end

local function shoot_and_collect_charge(session)
  if has_charge(session.state) then
    return wait_item_fanfare(session, "charge_item_fanfare")
  end

  hold(session, 40, {}, "charge_platform_settle")

  local i
  for i = 1, 25 do
    hold(session, 1, {"RIGHT"}, "charge_spin_back")
  end
  hold(session, 8, {}, "charge_spin_settle")
  hold(session, 6, {"LEFT"}, "charge_face")
  hold(session, 4, {}, "charge_face_release")

  hold(session, 8, {"R"}, "charge_angle_hold")
  for i = 1, 8 do
    hold(session, 8, {"X", "R"}, "charge_chozo_angle_shot")
    hold(session, 14, {"R"}, "charge_chozo_angle_wait")
    if has_charge(session.state) then
      return wait_item_fanfare(session, "charge_item_fanfare")
    end
  end

  ctrl.try_select(session, 0)

  for i = 1, 5 do
    hold(session, 2, {"LEFT", "X"}, "charge_chozo_shot")
    hold(session, 18, {}, "charge_chozo_shot_wait")
    if has_charge(session.state) then
      return wait_item_fanfare(session, "charge_item_fanfare")
    end
  end

  local collected = false
  for i = 0, 99 do
    if has_charge(session.state) then
      collected = true
      break
    end
    if (i % 20) < 6 then
      hold(session, 1, {"LEFT", "A"}, "charge_collect_hop")
    else
      hold(session, 1, {"LEFT"}, "charge_collect_walk")
    end
  end
  if not collected and not has_charge(session.state) then
    error("charge_collect: Charge Beam PLM not collected: " .. brief(session.state))
  end
  return wait_item_fanfare(session, "charge_item_fanfare")
end

function M.play_charge_beam_collect(session)
  ctrl.require_room(session, ROOM_BIG_PINK, "charge_beam_collect")
  if has_charge(session.state) then
    return session.state
  end
  M._descend_main_to_mass(session)
  bomb_drop_to_charge_floor(session)
  approach_chozo_platform(session)
  return shoot_and_collect_charge(session)
end

local MASS_LAND_POSE = {
  [1] = true, [2] = true, [9] = true, [10] = true,
  [39] = true, [40] = true, [41] = true, [42] = true,
  [81] = true, [101] = true, [129] = true, [130] = true,
}

local function grounded_mass_land(state, y_max)
  if num(state.samus_y) > y_max or num(state.samus_x) < 690 then
    return false
  end
  if math.abs(num(state.velocity_y)) > 1 then
    return false
  end
  return MASS_LAND_POSE[num(state.pose)] == true
end

function M.play_charge_beam_return(session)
  ctrl.require_room(session, ROOM_BIG_PINK, "charge_beam_return")
  if not has_charge(session.state) then
    error(string.format(
      "charge_beam_return: Charge not collected (beams=0x%04X)",
      num(session.state.collected_beams)
    ))
  end

  if grounded_mass_land(session.state, MASS_Y_MAX) then
    hold(session, 12, {}, "charge_return_mass_settle")
    return session.state
  end

  ctrl.unmorph(session)
  hold(session, 15, {}, "charge_return_stand")
  wait_item_fanfare(session, "charge_return_fanfare_clear")

  local i
  for i = 1, 100 do
    local state = hold(session, 1, {"RIGHT", "B"}, "charge_return_to_shaft")
    if num(state.samus_x) >= 710 then
      break
    end
  end
  hold(session, 12, {}, "charge_return_shaft_settle")

  local best_y = num(session.state.samus_y)
  local soft_mass = 1760
  local cycle
  for cycle = 1, 50 do
    local state = session.state
    if grounded_mass_land(state, MASS_Y_MAX) then
      hold(session, 16, {}, "charge_return_mass_settle")
      return session.state
    end

    hold(session, 8, {"RIGHT", "B"}, "charge_return_runup")
    hold(session, 1, {"RIGHT", "A"}, "charge_return_jump")
    for i = 1, 32 do
      state = hold(session, 1, {"RIGHT", "A"}, "charge_return_air")
      if num(state.samus_y) < best_y - 2 then
        best_y = num(state.samus_y)
      end
    end
    for i = 1, 28 do
      state = hold(session, 1, {}, "charge_return_land")
      if grounded_mass_land(state, soft_mass) then
        hold(session, 16, {}, "charge_return_mass_settle")
        return session.state
      end
    end
  end
  error(string.format(
    "charge_beam_return: stuck best_y=%s: %s",
    tostring(best_y),
    brief(session.state)
  ))
end

function M.play_big_pink_charge_detour(session)
  ctrl.require_room(session, ROOM_BIG_PINK, "charge_detour")
  if has_charge(session.state) then
    if num(session.state.samus_y) > 1600 and num(session.state.samus_x) < 800 then
      return session.state
    end
    M._descend_main_to_mass(session)
    return session.state
  end
  M.play_charge_beam_collect(session)
  return M.play_charge_beam_return(session)
end

M.has_charge = has_charge

return M
