-- Warehouse with Hi-Jump → Zeela → Kihunter → Baby Kraid → Kraid.
-- Port of snes/super_metroid/routes/kpdr/kraid/to_kraid.py.

local geo = require("red_tower.ctrl")
local hj = require("kraid.collect_hijump")
local ret = require("kraid.return_hijump")
local wh = require("kraid.warehouse_stack")

local M = {}

function M.play_warehouse_to_zeela_with_hijump(session)
  wh.play_warehouse_wall_to_lower_lip(session)
  geo.unmorph(session)
  geo.select_weapon(session, 0)
  geo.settle_hold(session, 12, "warehouse_hj_release")
  for _ = 1, 120 do
    if session.state.samus_x <= 445 then
      break
    end
    geo.hold(session, 1, {"LEFT"}, "warehouse_hj_backoff")
  end
  geo.hold(session, 5, {"RIGHT"}, "warehouse_hj_brake")
  geo.settle_hold(session, 8, "warehouse_hj_jump_release")
  for frame = 0, 179 do
    local names = (frame < 25) and {"A"} or {"RIGHT", "B", "A"}
    local state = geo.hold(session, 1, names, "warehouse_hj_climb")
    if state.samus_x >= 720 and state.samus_y <= 160 then
      break
    end
  end
  geo.settle_hold(session, 30, "warehouse_hj_door_settle")
  geo.unmorph(session)
  geo.select_weapon(session, 0)
  geo.hold(session, 3, {"RIGHT"}, "warehouse_face_zeela")
  geo.settle_hold(session, 3, "warehouse_face_zeela_release")
  geo.hold(session, 2, {"RIGHT", "X"}, "warehouse_zeela_door_shot")
  geo.settle_hold(session, 30, "warehouse_zeela_door_open")
  local entered = false
  local state
  for _ = 1, 420 do
    state = geo.hold(session, 1, {"RIGHT", "B", "A"}, "warehouse_enter_zeela")
    if state.room_id == geo.ROOM_ZEELA then
      entered = true
      break
    end
  end
  if not entered then
    error("warehouse_to_zeela: " .. geo.fmt_state(session.state))
  end
  return geo.wait_ordinary_room(session, geo.ROOM_ZEELA, {
    settle_frames = 280,
    label = "warehouse_to_zeela",
  })
end

function M.play_zeela_to_kihunter(session)
  geo.require_room(session, geo.ROOM_ZEELA, "zeela_to_kihunter")
  geo.unmorph(session)
  geo.select_weapon(session, 0)
  geo.settle_hold(session, 10, "zeela_entry_release")
  geo.hold(session, 10, {"A"}, "zeela_first_drop_jump")
  geo.hold(session, 1, {"DOWN"}, "zeela_first_drop_aim")
  geo.hold(session, 2, {"X"}, "zeela_first_drop_shot")
  geo.settle_hold(session, 80, "zeela_first_drop")
  geo.ensure_morph(session)
  for _ = 1, 300 do
    local state = geo.hold(session, 1, {"RIGHT"}, "zeela_middle_roll")
    if state.samus_x >= 105 and state.samus_y >= 325 then
      break
    end
  end
  geo.unmorph(session)
  geo.settle_hold(session, 30, "zeela_middle_land")
  geo.select_weapon(session, 0)
  geo.hold(session, 8, {"A"}, "zeela_second_drop_jump")
  geo.hold(session, 1, {"DOWN"}, "zeela_second_drop_aim")
  geo.hold(session, 2, {"X"}, "zeela_second_drop_shot")
  for _ = 1, 180 do
    local state = geo.hold(session, 1, {"LEFT"}, "zeela_second_drop")
    if state.samus_y >= 395 then
      break
    end
  end
  geo.settle_hold(session, 40, "zeela_bottom_land")
  geo.ensure_morph(session)
  local rolled = false
  local state
  for frame = 0, 699 do
    local names = (frame % 45 < 2) and {"RIGHT", "X"} or {"RIGHT"}
    state = geo.hold(session, 1, names, "zeela_bottom_bomb_roll")
    if state.samus_x >= 400 then
      rolled = true
      break
    end
  end
  if not rolled then
    error("zeela_to_kihunter: tunnel stalled: " .. geo.fmt_state(session.state))
  end
  geo.unmorph(session)
  geo.settle_hold(session, 40, "zeela_up_door_stand")
  geo.select_weapon(session, 0)
  geo.hold(session, 2, {"UP"}, "zeela_up_door_aim")
  geo.hold(session, 2, {"UP", "X"}, "zeela_up_door_shot")
  geo.settle_hold(session, 35, "zeela_up_door_open")
  local entered = false
  for _ = 1, 400 do
    state = geo.hold(session, 1, {"A"}, "zeela_enter_kihunter")
    if state.room_id == geo.ROOM_WAREHOUSE_KIHUNTER then
      entered = true
      break
    end
  end
  if not entered then
    error("zeela_to_kihunter: up door failed: " .. geo.fmt_state(session.state))
  end
  return geo.wait_ordinary_room(session, geo.ROOM_WAREHOUSE_KIHUNTER, {
    settle_frames = 280,
    label = "zeela_to_kihunter",
  })
end

function M.play_kihunter_to_baby_kraid(session)
  geo.require_room(session, geo.ROOM_WAREHOUSE_KIHUNTER, "kihunter_to_baby")
  geo.settle_hold(session, 80, "kihunter_entry_floor")
  geo.unmorph(session)
  geo.select_weapon(session, 0)
  for _ = 1, 300 do
    if session.state.samus_x >= 350 then
      break
    end
    geo.hold(session, 1, {"RIGHT", "B"}, "kihunter_drop_position")
  end
  geo.hold(session, 6, {"LEFT"}, "kihunter_drop_brake")
  geo.settle_hold(session, 10, "kihunter_drop_settle")
  geo.hold(session, 3, {"RIGHT"}, "kihunter_drop_exact")
  geo.hold(session, 2, {"LEFT"}, "kihunter_drop_exact_brake")
  geo.settle_hold(session, 10, "kihunter_drop_exact_settle")
  geo.ensure_morph(session)
  geo.hold(session, 2, {"X"}, "kihunter_floor_bomb")
  geo.settle_hold(session, 55, "kihunter_floor_bomb_wait")
  geo.hold(session, 2, {"X"}, "kihunter_floor_bomb2")
  for _ = 1, 180 do
    local state = geo.hold(session, 1, {}, "kihunter_floor_drop")
    if state.samus_y >= 310 then
      break
    end
  end
  geo.ensure_morph(session)
  for _ = 1, 160 do
    local state = geo.hold(session, 1, {"LEFT"}, "kihunter_shaft_align")
    if state.samus_y >= 350 then
      break
    end
  end
  for _ = 1, 360 do
    local state = geo.hold(session, 1, {"RIGHT"}, "kihunter_lower_roll")
    if state.samus_x >= 470 then
      break
    end
  end
  geo.unmorph(session)
  geo.select_weapon(session, 0)
  local entered = false
  local state
  for frame = 0, 499 do
    local names = (frame % 25 < 5) and {"RIGHT", "B", "X"} or {"RIGHT", "B", "A"}
    state = geo.hold(session, 1, names, "kihunter_enter_baby")
    if state.room_id == geo.ROOM_BABY_KRAID then
      entered = true
      break
    end
  end
  if not entered then
    error("kihunter_to_baby: " .. geo.fmt_state(session.state))
  end
  return geo.wait_ordinary_room(session, geo.ROOM_BABY_KRAID, {
    settle_frames = 280,
    label = "kihunter_to_baby",
  })
end

function M.baby_kraid_sweep(session, direction, target_x, limit, label)
  for frame = 0, limit - 1 do
    local phase = frame % 24
    local names
    if phase < 3 then
      names = {direction, "X"}
    elseif phase >= 14 then
      names = {direction, "B", "A"}
    else
      names = {direction, "B"}
    end
    local state = geo.hold(session, 1, names, label)
    if direction == "RIGHT" and state.samus_x >= target_x then
      return
    end
    if direction == "LEFT" and state.samus_x <= target_x then
      return
    end
  end
  error(label .. ": " .. geo.fmt_state(session.state))
end

function M.play_baby_kraid_to_eye(session)
  geo.require_room(session, geo.ROOM_BABY_KRAID, "baby_kraid_to_eye")
  geo.settle_hold(session, 100, "baby_kraid_entry_floor")
  geo.unmorph(session)
  geo.select_weapon(session, 2)
  M.baby_kraid_sweep(session, "RIGHT", 1490, 1700, "baby_kraid_forward")
  if (session.state.enemies_killed or 0) < (session.state.num_enemies or 0) then
    M.baby_kraid_sweep(session, "LEFT", 50, 1900, "baby_kraid_cleanup")
  end
  M.baby_kraid_sweep(session, "RIGHT", 1490, 1900, "baby_kraid_return")
  local entered = false
  local state
  for _ = 1, 600 do
    state = geo.hold(session, 1, {"RIGHT", "B", "A"}, "baby_kraid_enter_eye")
    if state.room_id == geo.ROOM_KRAID_EYE then
      entered = true
      break
    end
  end
  if not entered then
    error("baby_kraid_to_eye: gray door failed: " .. geo.fmt_state(session.state))
  end
  return geo.wait_ordinary_room(session, geo.ROOM_KRAID_EYE, {
    settle_frames = 300,
    label = "baby_kraid_to_eye",
  })
end

function M.play_eye_to_kraid(session)
  geo.require_room(session, geo.ROOM_KRAID_EYE, "eye_to_kraid")
  geo.settle_hold(session, 100, "kraid_eye_entry_floor")
  geo.unmorph(session)
  geo.select_weapon(session, 2)
  local entered = false
  local state
  for frame = 0, 1799 do
    local phase = frame % 28
    local names
    if phase < 3 then
      names = {"RIGHT", "X"}
    elseif phase >= 16 then
      names = {"RIGHT", "B", "A"}
    else
      names = {"RIGHT", "B"}
    end
    state = geo.hold(session, 1, names, "kraid_eye_run")
    if state.room_id == geo.ROOM_KRAID then
      entered = true
      break
    end
  end
  if not entered then
    error("eye_to_kraid: eye door failed: " .. geo.fmt_state(session.state))
  end
  return geo.wait_ordinary_room(session, geo.ROOM_KRAID, {
    settle_frames = 340,
    label = "eye_to_kraid",
  })
end

function M.play_kraid_entry_to_varia(session)
  local combat = require("combat.kraid")
  geo.require_room(session, geo.ROOM_KRAID, "kraid_entry_to_varia")
  local evidence = combat.play_kraid_fight_to_varia(session)
  if evidence.fight.outcome ~= "kraid_defeated" then
    error("kraid_entry_to_varia: fight failed (" .. tostring(evidence.fight.outcome)
      .. "): " .. geo.fmt_state(session.state))
  end
  if evidence.varia.outcome ~= "varia_collected" then
    error("kraid_entry_to_varia: Varia failed (" .. tostring(evidence.varia.outcome)
      .. "): " .. geo.fmt_state(session.state))
  end
  return session.state
end

function M.play_warehouse_to_kraid_with_hijump(session)
  if not geo.has_hi_jump(session.state) then
    error("warehouse_to_kraid_with_hijump: Hi-Jump not collected")
  end
  M.play_warehouse_to_zeela_with_hijump(session)
  M.play_zeela_to_kihunter(session)
  M.play_kihunter_to_baby_kraid(session)
  M.play_baby_kraid_to_eye(session)
  return M.play_eye_to_kraid(session)
end

function M.play_warehouse_hijump_kraid(session)
  hj.play_warehouse_to_hijump(session)
  ret.play_hijump_to_warehouse(session)
  return M.play_warehouse_to_kraid_with_hijump(session)
end

return M
