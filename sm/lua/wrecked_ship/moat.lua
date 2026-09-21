-- Crateria Kihunter → Moat shinespark → West Ocean (K6).

local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")
local spark = require("skills.shinespark")
local door = require("skills.door")
local west = require("wrecked_ship.west_ocean")

local M = {}
local ROOM_KIHUNTER = rooms.ROOM_CRATERIA_KIHUNTER or 0x948C
local ROOM_MOAT = rooms.ROOM_MOAT or 0x95FF
local ROOM_WEST_OCEAN = rooms.ROOM_WEST_OCEAN or 0x93FE
local ROOM_WS_ENTRANCE = rooms.ROOM_WS_ENTRANCE or 0xCA08
local ENEMY_BASE, ENEMY_STRIDE = 0x0F78, 0x40
local CLEAR_BUDGET = 2400
local RUNWAY_BUDGET = 700
local LEAVE_MOAT_BUDGET = 1500
local CHARGE_BUDGET = 500
local STORE_FRAMES = 18
local STAND_FRAMES, MICRO_RUN, HOP_FRAMES = 4, 2, 14
local UNSPIN_FRAMES, SPARK_ACTIVATE, SPARK_TRAVEL = 3, 16, 700
local SPARK_POSES = {[199]=true,[200]=true,[201]=true,[202]=true}
local PRE_SPARK_PIN_X0, PRE_SPARK_PIN_X1, PRE_SPARK_PIN_Y_MAX = 28, 50, 200
local LEFT_DOOR_LIP_X, WALK_STOP_X, RIGHT_DOOR_LIP_X = 80, 180, 720
local RUNWAY_X0, RUNWAY_X1 = 140, 200

local function wram_u16(session, addr)
  local ram = require("ram")
  if ram.u16 then
    return ram.u16(session, addr)
  end
  return ctrl.u16(ctrl.wram(session), addr)
end

function M.list_air_enemies(session)
  local out, i = {}, 0
  for i = 0, 7 do
    local base = ENEMY_BASE + i * ENEMY_STRIDE
    local eid = wram_u16(session, base)
    if eid ~= 0 then
      local hp = wram_u16(session, base + 0x14)
      if hp > 0 and hp <= 120 then
        local x = wram_u16(session, base + 0x02)
        local y = wram_u16(session, base + 0x06)
        if y <= 350 then
          out[#out + 1] = {i, x, y, hp, eid}
        end
      end
    end
  end
  return out
end

function M.air_enemies_alive(session)
  return #M.list_air_enemies(session) > 0
end

function M.near_left_door(state)
  return ctrl.num(state.room_id) == ROOM_KIHUNTER
    and ctrl.x(state) < LEFT_DOOR_LIP_X
    and ctrl.y(state) < 250
end

function M.near_right_door(state)
  return ctrl.num(state.room_id) == ROOM_KIHUNTER
    and ctrl.x(state) > RIGHT_DOOR_LIP_X
    and ctrl.y(state) < 250
end

function M.avoid_kihunter_doors(session, label, allow_right)
  local st = session.state
  if ctrl.num(st.room_id) ~= ROOM_KIHUNTER then
    return false
  end
  if ctrl.num(st.door_transition) ~= 0 then
    ctrl.hold(session, 1, {}, label .. "_door_transition")
    return true
  end
  if M.near_left_door(st) then
    ctrl.hold(session, 1, {"RIGHT"}, label .. "_avoid_tube")
    return true
  end
  if not allow_right and M.near_right_door(st) then
    ctrl.hold(session, 1, {"LEFT"}, label .. "_avoid_moat")
    return true
  end
  return false
end

function M.play_leave_moat_to_kihunter(session)
  ctrl.require_room(session, ROOM_MOAT, "leave_moat")
  local label = "leave_moat"
  local frame
  for frame = 0, LEAVE_MOAT_BUDGET - 1 do
    local st = session.state
    if ctrl.num(st.room_id) == ROOM_KIHUNTER
        and ctrl.num(st.game_state) == 8
        and ctrl.num(st.door_transition) == 0 then
      return st
    end
    if ctrl.num(st.room_id) == ROOM_MOAT then
      if ctrl.x(st) < 100 then
        if (frame % 8) < 5 then
          ctrl.hold(session, 1, {"LEFT", "X"}, label .. "_door")
        else
          ctrl.hold(session, 1, {"LEFT"}, label .. "_push")
        end
      else
        ctrl.hold(session, 1, {"LEFT", "B"}, label .. "_walk")
      end
    else
      ctrl.hold(session, 1, {}, label .. "_transition")
    end
  end
  ctrl.timeout(label .. ": never reached Kihunter 0x948C: " .. ctrl.brief(session.state))
end

function M.play_open_kihunter_moat_door(session)
  ctrl.require_room(session, ROOM_KIHUNTER, "open_moat_door")
  pcall(ctrl.select_weapon, session, 0)
  local label = "open_moat_door"
  local i
  for i = 1, 50 do
    local st = session.state
    if ctrl.num(st.room_id) ~= ROOM_KIHUNTER then
      ctrl.timeout(label .. ": left Kihunter: " .. ctrl.brief(st))
    end
    if 640 <= ctrl.x(st) and ctrl.x(st) <= 680 then
      break
    end
    local dir = (ctrl.x(st) > 680) and "LEFT" or "RIGHT"
    ctrl.hold(session, 1, {dir}, label .. "_align")
  end
  ctrl.hold(session, 12, {"RIGHT"}, label .. "_face")
  ctrl.hold(session, 8, {}, label .. "_face_release")
  if door.beam_open_door then
    door.beam_open_door(session, {label = label, shots = 24, shot_frames = 2, fuse_frames = 7})
  else
    for i = 1, 24 do
      ctrl.hold(session, 2, {"X"}, label .. "_shot")
      ctrl.hold(session, 7, {}, label .. "_fuse")
    end
  end
  for i = 1, 40 do
    local st = session.state
    if ctrl.num(st.room_id) == ROOM_KIHUNTER and ctrl.num(st.door_transition) == 0 then
      if ctrl.x(st) < 700 then
        break
      end
      ctrl.hold(session, 1, {"LEFT"}, label .. "_safe")
    elseif ctrl.num(st.room_id) == ROOM_MOAT or ctrl.num(st.door_transition) ~= 0 then
      ctrl.hold(session, 1, {"LEFT"}, label .. "_abort_enter")
      if ctrl.num(st.room_id) == ROOM_MOAT and ctrl.num(st.door_transition) == 0 then
        return M.play_leave_moat_to_kihunter(session)
      end
    end
  end
  if ctrl.num(session.state.room_id) ~= ROOM_KIHUNTER then
    ctrl.timeout(label .. ": not in Kihunter after open: " .. ctrl.brief(session.state))
  end
  return session.state
end

function M.play_kihunter_pre_spark_pin(session)
  ctrl.require_room(session, ROOM_KIHUNTER, "kihunter_pre_spark_pin")
  local label = "kihunter_pre_spark_pin"
  local i
  for i = 1, RUNWAY_BUDGET do
    local st = session.state
    if ctrl.num(st.room_id) ~= ROOM_KIHUNTER then
      ctrl.timeout(label .. ": left room: " .. ctrl.brief(st))
    end
    if ctrl.num(st.door_transition) ~= 0 then
      ctrl.hold(session, 1, {"RIGHT"}, label .. "_door_trans")
    elseif ctrl.is_knockback(st) then
      ctrl.escape_kb(session, {prefer_dir = "RIGHT", run_frames = 4, spin_frames = 12, label = label})
    elseif PRE_SPARK_PIN_X0 <= ctrl.x(st) and ctrl.x(st) <= PRE_SPARK_PIN_X1
        and ctrl.y(st) <= PRE_SPARK_PIN_Y_MAX and ctrl.num(st.velocity_y) == 0 then
      ctrl.hold(session, 8, {}, label .. "_settle")
      ctrl.hold(session, 6, {"RIGHT"}, label .. "_face")
      ctrl.hold(session, 6, {}, label .. "_face_settle")
      return session.state
    elseif ctrl.x(st) < PRE_SPARK_PIN_X0 then
      ctrl.hold(session, 1, {"RIGHT"}, label .. "_too_left")
    else
      ctrl.hold(session, 1, {"LEFT"}, label .. "_left")
    end
  end
  ctrl.timeout(label .. ": never reached left pin: " .. ctrl.brief(session.state))
end

function M.play_clear_kihunter_room(session)
  ctrl.require_room(session, ROOM_KIHUNTER, "kihunter_clear")
  pcall(ctrl.select_weapon, session, 0)
  local label, plant_face, zero_streak = "kihunter_clear", nil, 0
  local frame
  for frame = 0, CLEAR_BUDGET - 1 do
    local st = session.state
    if ctrl.num(st.room_id) ~= ROOM_KIHUNTER then
      ctrl.timeout(label .. ": left room during clear")
    end
    if M.avoid_kihunter_doors(session, label, false) then
      plant_face = nil
    elseif ctrl.is_knockback(st) then
      local prefer = (ctrl.x(st) < WALK_STOP_X) and "RIGHT" or "LEFT"
      ctrl.escape_kb(session, {prefer_dir = prefer, run_frames = 4, spin_frames = 12, label = label})
      M.avoid_kihunter_doors(session, label, false)
    else
      local enemies = M.list_air_enemies(session)
      if #enemies == 0 then
        zero_streak = zero_streak + 1
        ctrl.hold(session, 1, {}, label .. "_confirm")
        if zero_streak >= 20 then
          return session.state
        end
      else
        zero_streak = 0
        if ctrl.x(st) <= WALK_STOP_X then
          local sum, e = 0, 1
          for e = 1, #enemies do
            sum = sum + enemies[e][2]
          end
          local want = (sum / #enemies >= ctrl.x(st) - 8) and "RIGHT" or "LEFT"
          if want == "LEFT" and ctrl.x(st) < LEFT_DOOR_LIP_X + 40 then
            want = "RIGHT"
          end
          if plant_face ~= want then
            ctrl.hold(session, 2, {want}, label .. "_face")
            ctrl.hold(session, 3, {}, label .. "_face_release")
            plant_face = want
          end
          if (frame % 32) < 26 then
            ctrl.hold(session, 1, {"X"}, label .. "_plant")
          else
            ctrl.hold(session, 1, {}, label .. "_plant_rel")
          end
        else
          plant_face = nil
          local pulse = frame % 30
          if pulse < 24 or pulse >= 26 then
            ctrl.hold(session, 1, {"LEFT", "X"}, label .. "_walk_shot")
          else
            ctrl.hold(session, 1, {"LEFT"}, label .. "_release")
          end
        end
      end
    end
  end
  ctrl.timeout(label .. ": air enemies still alive after left-walk clear")
end

function M.play_kihunter_charge_store(session, charge_mode)
  charge_mode = charge_mode or "full"
  ctrl.require_room(session, ROOM_KIHUNTER, "kihunter_charge")
  local label = "kihunter_charge"
  if charge_mode == "short" or charge_mode == "stutter" then
    local attempt
    for attempt = 1, 3 do
      if ctrl.is_knockback(session.state) then
        ctrl.hold(session, 20, {}, label .. "_kb_idle")
        if ctrl.is_knockback(session.state) then
          ctrl.escape_kb(session, {prefer_dir = "LEFT", run_frames = 6, spin_frames = 14, label = label})
        end
      end
      local charge = spark.short_charge_until_boost(session, "RIGHT", {
        stutter = charge_mode == "stutter",
        store_on_last = false,
        label = label .. "_" .. charge_mode,
      })
      local st = session.state
      if ctrl.num(st.room_id) ~= ROOM_KIHUNTER then
        return st
      end
      if (type(charge) ~= "table" or charge.ok) and ctrl.num(st.velocity_y) == 0 and ctrl.y(st) >= 170 then
        local i
        for i = 1, STORE_FRAMES do
          ctrl.hold(session, 1, {"DOWN"}, label .. "_store")
        end
        return session.state
      end
      ctrl.hold(session, 12, {"LEFT"}, label .. "_retry_left")
    end
    ctrl.timeout(label .. ": short-charge store failed")
  end
  local frame
  for frame = 0, CHARGE_BUDGET - 1 do
    local st = session.state
    if ctrl.num(st.room_id) ~= ROOM_KIHUNTER then
      return st
    end
    if ctrl.is_knockback(st) then
      ctrl.hold(session, 20, {}, label .. "_kb_idle")
      if ctrl.is_knockback(session.state) then
        ctrl.escape_kb(session, {prefer_dir = "LEFT", run_frames = 6, spin_frames = 14, label = label})
      end
    elseif (st.speed_boosting or ctrl.num(st.speed_counter) >= 4)
        and ctrl.num(st.velocity_y) == 0
        and ctrl.num(st.pose) ~= 137 and ctrl.num(st.pose) ~= 138
        and ctrl.y(st) >= 170 then
      local i
      for i = 1, STORE_FRAMES do
        ctrl.hold(session, 1, {"DOWN"}, label .. "_store")
      end
      return session.state
    else
      local x = ctrl.x(st)
      if 545 <= x and x <= 575 and ctrl.num(st.velocity_y) == 0 then
        ctrl.hold(session, 1, {"RIGHT", "B", "A"}, label .. "_trap_hop")
      else
        ctrl.hold(session, 1, {"RIGHT", "B"}, label .. "_run")
      end
    end
  end
  ctrl.timeout(label .. ": no grounded speed store")
end

function M.play_moat_shinespark(session, opts)
  opts = opts or {}
  local st = session.state
  local need_moat = opts.from_moat
  if need_moat == nil then
    need_moat = ctrl.num(st.room_id) == ROOM_MOAT
  end
  if need_moat then
    if ctrl.num(st.room_id) == ROOM_MOAT then
      M.play_leave_moat_to_kihunter(session)
    end
    ctrl.require_room(session, ROOM_KIHUNTER, "moat_spark")
    M.play_open_kihunter_moat_door(session)
  else
    ctrl.require_room(session, ROOM_KIHUNTER, "moat_spark")
  end
  if not opts.skip_clear and M.air_enemies_alive(session) then
    local ok = pcall(M.play_clear_kihunter_room, session)
    if not ok then
      ctrl.hold(session, 30, {"RIGHT"}, "kihunter_clear_retry_right")
      M.play_clear_kihunter_room(session)
    end
    if M.air_enemies_alive(session) then
      ctrl.hold(session, 40, {"RIGHT", "X"}, "kihunter_clear_retry_shot")
      M.play_clear_kihunter_room(session)
    end
  end
  local st0 = session.state
  if not (ctrl.x(st0) < 100 and ctrl.y(st0) <= PRE_SPARK_PIN_Y_MAX and ctrl.num(st0.velocity_y) == 0) then
    M.play_kihunter_pre_spark_pin(session)
  end
  M.play_kihunter_charge_store(session, opts.charge_mode or "full")
  local label, i = "moat_spark", 1
  for i = 1, STAND_FRAMES do
    ctrl.hold(session, 1, {}, label .. "_stand")
  end
  for i = 1, MICRO_RUN do
    ctrl.hold(session, 1, {"RIGHT", "B"}, label .. "_micro_run")
  end
  for i = 1, HOP_FRAMES do
    ctrl.hold(session, 1, {"RIGHT", "B", "A"}, label .. "_hop")
  end
  for i = 1, UNSPIN_FRAMES do
    ctrl.hold(session, 1, {"UP"}, label .. "_unspin")
  end
  for i = 1, SPARK_ACTIVATE do
    ctrl.hold(session, 1, {"RIGHT", "A"}, label .. "_activate")
    st = session.state
    if SPARK_POSES[ctrl.num(st.pose)] or ctrl.num(st.room_id) == ROOM_MOAT or ctrl.num(st.room_id) == ROOM_WEST_OCEAN then
      break
    end
  end
  local frame
  for frame = 0, SPARK_TRAVEL - 1 do
    st = session.state
    if ctrl.num(st.room_id) == ROOM_WEST_OCEAN and ctrl.num(st.door_transition) == 0 and ctrl.num(st.game_state) == 8 then
      ctrl.hold(session, 8, {}, label .. "_west_settle")
      return session.state
    end
    if ctrl.num(st.room_id) == ROOM_WEST_OCEAN then
      ctrl.hold(session, 1, {"RIGHT"}, label .. "_west_trans")
    elseif ctrl.num(st.room_id) == ROOM_MOAT and not SPARK_POSES[ctrl.num(st.pose)] then
      if (frame % 8) < 5 then
        ctrl.hold(session, 1, {"RIGHT", "X"}, label .. "_door_open")
      else
        ctrl.hold(session, 1, {"RIGHT"}, label .. "_door_walk")
      end
    else
      ctrl.hold(session, 1, {"RIGHT", "A"}, label .. "_travel")
    end
  end
  st = session.state
  if ctrl.num(st.room_id) == ROOM_WEST_OCEAN then
    return st
  end
  ctrl.timeout(label .. ": did not reach West Ocean 0x93FE " .. ctrl.brief(st))
end

function M.play_moat_cross(session)
  local st = session.state
  if ctrl.num(st.room_id) == ROOM_KIHUNTER or ctrl.num(st.room_id) == ROOM_MOAT then
    return M.play_moat_shinespark(session)
  end
  ctrl.timeout(string.format("moat_cross: expected Kihunter 0x948C or Moat 0x95FF, got 0x%04X", ctrl.num(st.room_id)))
end

function M.play_moat_to_west_ocean(session, charge_mode)
  return M.play_moat_shinespark(session, {charge_mode = charge_mode or "full"})
end

function M.play_moat_to_ws(session, opts)
  opts = opts or {}
  local label = opts.label or "moat_to_ws"
  local st = session.state
  if ctrl.num(st.room_id) == ROOM_WS_ENTRANCE then
    if ctrl.num(st.door_transition) == 0 and ctrl.num(st.game_state) == 8 then
      return st
    end
    ctrl.hold(session, 12, {}, label .. "_ws_settle")
    return session.state
  end
  if ctrl.num(st.room_id) == ROOM_KIHUNTER or ctrl.num(st.room_id) == ROOM_MOAT then
    M.play_moat_to_west_ocean(session, opts.moat_charge_mode or "full")
    st = session.state
  end
  if ctrl.num(st.room_id) == ROOM_WEST_OCEAN then
    return west.play_west_ocean_to_ws(session, {
      charge_mode = opts.wo_charge_mode or "stutter",
      label = label .. "_wo",
    })
  end
  ctrl.timeout(label .. ": expected Kihunter/Moat/West Ocean/WS entrance " .. ctrl.brief(st))
end

return M
