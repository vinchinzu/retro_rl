-- Business → Cathedral Entrance → Cathedral → Rising Tide.
-- Lua 5.1. Session: step / hold / wait_until / span.

local iceg = require("ice.geometry")
local knockback = require("skills.knockback")
local door = require("skills.door")
local geom = require("skills.geometry")

local ROOM_BUSINESS = 0xA7DE
local ROOM_CATH_ENT = 0xA7B3
local ROOM_CATHEDRAL = 0xA788
local ROOM_RISING = 0xAFA3
local LEDGE = geom.LEDGE_POSES or iceg.LEDGE_POSES
local STANDING = geom.STANDING_POSES or iceg.STANDING_POSES

local CATH_DOOR_Y_MIN, CATH_DOOR_Y_MAX = 840, 900
local ELEVATOR_Y = 680
local CATH_ENT_BOMB_FRAMES = 12
local CATH_ENT_FLOOR_FRAMES = 800
local CATH_ENT_CLIMB_FRAMES = 1800
local CATH_ENT_DOOR_X, CATH_ENT_DOOR_Y = 680, 170
local CATH_ENT_CLIMB_X_MIN, CATH_ENT_CLIMB_X_MAX = 560, 680
local CATH_ENT_MID_Y = {270, 330}
local CATH_CROSS_FRAMES = 5000
local CATH_DOOR_X = 640
local CATH_DOOR_Y_MIN2, CATH_DOOR_Y_MAX2 = 300, 420
local CATH_FALL_Y = 230

local M = {}

local function play_business_to_cathedral_entrance(session)
  local label = "business_to_cathedral_entrance"
  iceg.require_room(session, ROOM_BUSINESS, label)
  local stable, settled = 0, false
  for _ = 1, 600 do
    local state = session:hold(1, {}, label .. "_elevator_settle")
    if state.samus_y == ELEVATOR_Y then
      stable = stable + 1
      if stable >= 24 then
        settled = true
        break
      end
    else
      stable = 0
    end
  end
  if not settled then
    error(label .. ": elevator did not settle")
  end

  local banded = false
  for frame = 0, 499 do
    local state = session.state
    if CATH_DOOR_Y_MIN <= state.samus_y and state.samus_y <= CATH_DOOR_Y_MAX
        and state.velocity_y == 0 and LEDGE[state.pose] then
      banded = true
      break
    end
    local direction = (math.floor(frame / 50) % 2 == 0) and "RIGHT" or "LEFT"
    if frame % 50 < 12 then
      session:hold(1, {direction, "B", "A"}, label .. "_door_band")
    else
      session:hold(1, {direction, "B"}, label .. "_door_band")
    end
  end
  if not banded then
    error(label .. ": cathedral door band missed")
  end

  iceg.select_weapon(session, 0)
  local hit = false
  for _ = 1, 680 do
    local state = session.state
    if state.room_id == ROOM_CATH_ENT then
      hit = true
      break
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "RIGHT", run_frames = 4, spin_frames = 12, label = label .. "_kb",
      })
    else
      session:hold(1, {"RIGHT", "B", "X"}, label .. "_door")
    end
  end
  if not hit then
    error(label .. ": Cathedral door missed")
  end

  local state = iceg.wait_ordinary_room(session, ROOM_CATH_ENT, 320, label)
  iceg.unmorph(session)
  for _ = 1, 30 do
    local st = session:hold(1, {}, label .. "_lip_settle")
    if st.velocity_y == 0 and LEDGE[st.pose] and st.door_transition == 0 then
      return st
    end
  end
  return state
end

local function cath_entrance_land(session, label)
  for _ = 1, 40 do
    local state = session:hold(1, {}, label .. "_land")
    if state.velocity_y == 0 and STANDING[state.pose] then
      break
    end
  end
end

local function cath_entrance_bomb_drop(session, label)
  iceg.unmorph(session)
  iceg.ensure_morph(session)
  for _ = 1, 40 do
    local state = session:hold(1, {"RIGHT"}, label .. "_morph_edge")
    if state.samus_x >= 82 then
      break
    end
  end
  for _ = 1, CATH_ENT_BOMB_FRAMES do
    session:hold(2, {"X"}, label .. "_bomb")
    local state = session:hold(48, {}, label .. "_bomb_fuse")
    if state.samus_y > 300 then
      break
    end
  end
  for _ = 1, 100 do
    local state = session:hold(1, {}, label .. "_bomb_fall")
    if state.velocity_y == 0 and state.samus_y > 300 then
      break
    end
  end
  iceg.unmorph(session)
end

local function cath_entrance_floor_cross(session, label)
  iceg.select_weapon(session, 0)
  local floor_ok = {}
  for k, v in pairs(LEDGE) do
    floor_ok[k] = v
  end
  floor_ok[11], floor_ok[12], floor_ok[37], floor_ok[38] = true, true, true, true
  local max_x, min_y = session.state.samus_x, session.state.samus_y
  for frame = 0, CATH_ENT_FLOOR_FRAMES - 1 do
    local state = session.state
    if state.room_id == ROOM_CATHEDRAL then
      break
    end
    if knockback.is_knockback(state) then
      local prefer = (state.samus_x >= 640) and "LEFT" or "RIGHT"
      knockback.escape_knockback_spin(session, {
        prefer_dir = prefer, run_frames = 4, spin_frames = 12, label = label .. "_floor",
        run_with = {"B", "X"}, spin_with = {"B", "A"}, ensure_beam = true,
        break_on_motion_clear = true,
      })
      state = session.state
      if state.samus_x > max_x then max_x = state.samus_x end
      if state.samus_y < min_y then min_y = state.samus_y end
    elseif state.samus_x >= 700 and state.samus_y >= 380 and state.velocity_y == 0
        and not knockback.is_knockback(state) then
      session:hold(1, {"LEFT", "B"}, label .. "_floor_deadwall")
      state = session.state
      if state.samus_x > max_x then max_x = state.samus_x end
      if state.samus_y < min_y then min_y = state.samus_y end
      if 620 <= state.samus_x and state.samus_x <= 680 and state.velocity_y == 0
          and floor_ok[state.pose] then
        break
      end
    elseif state.samus_x >= 620 and state.samus_x <= 690 and state.velocity_y == 0
        and state.samus_y >= 380 and floor_ok[state.pose] then
      break
    else
      local phase = frame % 45
      local inputs
      if phase < 20 then
        inputs = {"RIGHT", "B", "A"}
      elseif phase < 28 then
        inputs = {"RIGHT", "B", "X"}
      else
        inputs = {"RIGHT", "B"}
      end
      state = session:hold(1, inputs, label .. "_floor")
      if state.samus_x > max_x then max_x = state.samus_x end
      if state.samus_y < min_y then min_y = state.samus_y end
    end
  end
  return max_x, min_y
end

local function cath_entrance_climb_and_super_door(session, label, max_x, min_y)
  local stand_mid = {[1] = true, [2] = true, [9] = true, [10] = true}
  local spin_air = {[25] = true, [26] = true, [27] = true, [28] = true, [81] = true, [82] = true}
  local mid_landed, door_reached = false, false
  local settle_budget, near_mid_spin = 0, 0
  local y_lo, y_hi = CATH_ENT_MID_Y[1], CATH_ENT_MID_Y[2]
  local hit = false
  for frame = 0, CATH_ENT_CLIMB_FRAMES - 1 do
    local state = session.state
    if state.room_id == ROOM_CATHEDRAL then
      hit = true
      break
    end
    if state.samus_x > max_x then max_x = state.samus_x end
    if state.samus_y < min_y then min_y = state.samus_y end
    if knockback.is_knockback(state) then
      local prefer = (state.samus_x >= 640) and "LEFT" or "RIGHT"
      knockback.escape_knockback_spin(session, {
        prefer_dir = prefer, run_frames = 4, spin_frames = 14, label = label,
        run_with = {"B", "X"}, spin_with = {"B", "A"}, ensure_beam = true,
        break_on_motion_clear = true,
      })
    elseif state.samus_y <= CATH_ENT_DOOR_Y and state.samus_x >= CATH_ENT_DOOR_X then
      door_reached = true
      state = door.super_door_pressure_frame(session, frame, {
        label = label, period = 28, shoot_end = 5, face_end = 12, run_end = 20,
      })
      if state.room_id == ROOM_CATHEDRAL then
        hit = true
        break
      end
    elseif state.samus_y <= 220 and state.samus_x >= 600 then
      if state.selected_item ~= 2 then
        iceg.select_weapon(session, 2)
      end
      local phase = frame % 24
      local inputs
      if phase < 8 then
        inputs = {"RIGHT", "B", "A"}
      elseif phase < 12 then
        inputs = {"RIGHT", "X"}
      else
        inputs = {"RIGHT", "B"}
      end
      state = session:hold(1, inputs, label .. "_high")
      if state.room_id == ROOM_CATHEDRAL then
        hit = true
        break
      end
    elseif y_lo <= state.samus_y and state.samus_y <= y_hi and state.samus_x >= 580
        and state.velocity_y == 0 and stand_mid[state.pose] then
      mid_landed, settle_budget, near_mid_spin = true, 0, 0
      if state.selected_item ~= 0 then
        iceg.select_weapon(session, 0)
      end
      if state.samus_x > 650 then
        for _ = 1, 8 do
          session:hold(1, {"LEFT", "B"}, label .. "_mid_runway")
        end
      end
      session:hold(6, {}, label .. "_mid_plant")
      for _ = 1, 30 do
        session:hold(1, {"A"}, label .. "_mid_charge")
      end
      for _ = 1, 110 do
        state = session:hold(1, {"RIGHT", "B", "A"}, label .. "_mid_jump")
        if state.samus_y < min_y then min_y = state.samus_y end
        if state.samus_x > max_x then max_x = state.samus_x end
        if state.room_id == ROOM_CATHEDRAL then
          hit = true
          break
        end
        if state.samus_y <= CATH_ENT_DOOR_Y and state.samus_x >= CATH_ENT_DOOR_X then
          door_reached = true
          break
        end
      end
      if hit then
        break
      end
    else
      if not mid_landed and y_lo - 20 <= state.samus_y and state.samus_y <= y_hi + 30
          and state.samus_x >= 560 and spin_air[state.pose] then
        near_mid_spin = near_mid_spin + 1
        if near_mid_spin >= 90 and settle_budget <= 0 then
          settle_budget = 28
          near_mid_spin = 0
        end
      end
      if settle_budget > 0 and not mid_landed and state.samus_y < 380 and state.samus_x >= 560 then
        settle_budget = settle_budget - 1
        local dir_h = (state.samus_x > 640) and "LEFT" or "RIGHT"
        session:hold(1, {dir_h, "B"}, label .. "_mid_settle")
      elseif state.samus_y >= 390 and state.samus_x >= 700 and state.velocity_y == 0
          and stand_mid[state.pose] then
        session:hold(1, {"LEFT", "B"}, label .. "_deadwall_left")
      else
        if state.selected_item ~= 0 and not door_reached then
          iceg.select_weapon(session, 0)
        end
        local x = state.samus_x
        local dir_h
        if x > CATH_ENT_CLIMB_X_MAX then
          dir_h = "LEFT"
        elseif x < CATH_ENT_CLIMB_X_MIN then
          dir_h = "RIGHT"
        else
          dir_h = (math.floor(frame / 40) % 2 == 0) and "RIGHT" or "LEFT"
        end
        local phase = frame % 60
        local inputs
        if phase < 30 then
          inputs = {dir_h, "B", "A"}
        elseif phase < 40 then
          inputs = {"A"}
        elseif phase < 50 then
          inputs = {dir_h, "B"}
        else
          inputs = {dir_h, "B", "X"}
        end
        state = session:hold(1, inputs, label .. "_climb")
        if state.room_id == ROOM_CATHEDRAL then
          hit = true
          break
        end
      end
    end
  end
  if session.state.room_id ~= ROOM_CATHEDRAL then
    local state = session.state
    error(string.format(
      "%s: right Super door missed; room=0x%04X pose=%d xy=(%d,%d) mid_landed=%s door_reached=%s",
      label, state.room_id, state.pose, state.samus_x, state.samus_y,
      tostring(mid_landed), tostring(door_reached)
    ))
  end
end

function M.play_cathedral_entrance_to_cathedral(session)
  local label = "cathedral_entrance_to_cathedral"
  iceg.require_room(session, ROOM_CATH_ENT, label)
  cath_entrance_land(session, label)
  cath_entrance_bomb_drop(session, label)
  local max_x, min_y = cath_entrance_floor_cross(session, label)
  if session.state.room_id ~= ROOM_CATHEDRAL then
    cath_entrance_climb_and_super_door(session, label, max_x, min_y)
  end
  return iceg.wait_ordinary_room(session, ROOM_CATHEDRAL, 320, label)
end

function M.play_cathedral_to_rising_tide(session)
  local label = "cathedral_to_rising_tide"
  iceg.require_room(session, ROOM_CATHEDRAL, label)
  for _ = 1, 90 do
    local state = session:hold(1, {}, label .. "_land")
    if state.velocity_y == 0 and LEDGE[state.pose] and state.door_transition == 0 then
      break
    end
  end
  iceg.unmorph(session)
  iceg.select_weapon(session, 0)
  for _ = 1, 40 do
    local state = session.state
    if state.samus_x >= 55 and state.velocity_y == 0 then
      break
    end
    session:hold(1, {"RIGHT"}, label .. "_lip_walk")
  end
  for _ = 1, 12 do session:hold(1, {"RIGHT", "B"}, label .. "_open_run") end
  for _ = 1, 18 do session:hold(1, {"A"}, label .. "_open_charge") end
  for _ = 1, 36 do session:hold(1, {"RIGHT", "B", "A"}, label .. "_open_jump") end
  for _ = 1, 50 do
    local state = session:hold(1, {"RIGHT", "B", "X"}, label .. "_open_fall")
    if state.velocity_y == 0 or knockback.is_knockback(state) then
      break
    end
  end

  local max_x, min_y = session.state.samus_x, session.state.samus_y
  local door_reached, high_reached = false, false
  local stuck_frames, last_x = 0, session.state.samus_x
  local last_progress_x, no_progress = session.state.samus_x, 0
  local hit = false
  for frame = 0, CATH_CROSS_FRAMES - 1 do
    local state = session.state
    if state.room_id == ROOM_RISING then
      hit = true
      break
    end
    if state.samus_x > max_x then max_x = state.samus_x end
    if state.samus_y < min_y then min_y = state.samus_y end
    if state.samus_y <= 160 and state.samus_x >= 180 then
      high_reached = true
    end
    if math.abs(state.samus_x - last_x) <= 2 then
      stuck_frames = stuck_frames + 1
    else
      stuck_frames = 0
      last_x = state.samus_x
    end
    if state.samus_x > last_progress_x + 4 then
      last_progress_x = state.samus_x
      no_progress = 0
    else
      no_progress = no_progress + 1
    end

    if knockback.is_knockback(state) then
      local prefer = (state.samus_x >= 700) and "LEFT" or "RIGHT"
      knockback.escape_knockback_spin(session, {
        prefer_dir = prefer, run_frames = 6, spin_frames = 24, label = label,
        run_with = {"B", "X"}, spin_with = {"B", "A"}, run_reason = "kb_clear",
        spin_reason = "kb_spin", stop_room_id = ROOM_RISING,
        break_on_motion_clear = true, ensure_beam = true,
      })
      stuck_frames = 0
      last_x = session.state.samus_x
    elseif state.samus_x >= CATH_DOOR_X and state.samus_y >= 280 then
      door_reached = true
      if state.selected_item ~= 2 then
        iceg.select_weapon(session, 2)
      end
      if state.samus_y < 340 then
        local phase = frame % 24
        local inputs
        if phase < 10 then inputs = {"LEFT"}
        elseif phase < 16 then inputs = {}
        else inputs = {"RIGHT"} end
        session:hold(1, inputs, label .. "_door_drop")
      elseif state.samus_y > 400 then
        local dir_h = (state.samus_x > 740) and "LEFT" or "RIGHT"
        session:hold(1, {dir_h, "B", "A"}, label .. "_lava_hop")
      else
        state = door.super_door_pressure_frame(session, frame, {
          label = label, period = 40, shoot_end = 4, idle_end = 20,
          face_end = 28, run_end = 34, ensure_weapon = false,
        })
        if state.room_id == ROOM_RISING then
          hit = true
          break
        end
      end
    elseif state.samus_x >= 620 and state.samus_y < 280 then
      door_reached = true
      local phase = frame % 30
      local inputs
      if phase < 12 then inputs = {"RIGHT", "B"}
      elseif phase < 20 then inputs = {"RIGHT"}
      else inputs = (state.samus_x > 720) and {"LEFT"} or {"RIGHT", "B"} end
      session:hold(1, inputs, label .. "_drop_to_door")
    elseif state.samus_y > CATH_FALL_Y and state.samus_x < CATH_DOOR_X then
      if state.selected_item ~= 0 then
        iceg.select_weapon(session, 0)
      end
      if no_progress > 50 and 250 <= state.samus_x and state.samus_x <= 480 then
        for _ = 1, 14 do session:hold(1, {"LEFT", "B"}, label .. "_runway") end
        for _ = 1, 16 do session:hold(1, {"A"}, label .. "_recrest_charge") end
        for _ = 1, 40 do
          local st = session:hold(1, {"RIGHT", "B", "A"}, label .. "_recrest_jump")
          if st.samus_y < min_y then min_y = st.samus_y end
          if st.samus_x > max_x then max_x = st.samus_x end
          if st.room_id == ROOM_RISING or st.samus_y <= 160 then
            break
          end
        end
        for _ = 1, 10 do session:hold(1, {"RIGHT", "B", "X"}, label .. "_recrest_shot") end
        no_progress, stuck_frames = 0, 0
        last_x = session.state.samus_x
      else
        local grounded = state.velocity_y == 0 and LEDGE[state.pose]
        if grounded or stuck_frames > 20 then
          for _ = 1, 14 do session:hold(1, {"A"}, label .. "_recover_charge") end
          for _ = 1, 32 do
            local st = session:hold(1, {"RIGHT", "B", "A"}, label .. "_recover_jump")
            if st.samus_y < min_y then min_y = st.samus_y end
            if st.samus_x > max_x then max_x = st.samus_x end
            if st.room_id == ROOM_RISING or st.samus_y <= 150 then
              break
            end
          end
          session:hold(1, {"RIGHT", "B", "X"}, label .. "_recover_shot")
          stuck_frames = 0
          last_x = session.state.samus_x
        else
          local phase = frame % 20
          if phase < 12 then
            session:hold(1, {"RIGHT", "B", "A"}, label .. "_recover_air")
          else
            session:hold(1, {"RIGHT", "B", "X"}, label .. "_recover_air")
          end
        end
      end
    else
      if state.selected_item ~= 0 then
        iceg.select_weapon(session, 0)
      end
      local x = state.samus_x
      local period, jump_end, shoot_end
      if x < 220 then period, jump_end, shoot_end = 38, 20, 6
      elseif x < 360 then period, jump_end, shoot_end = 44, 28, 6
      elseif x < 480 then period, jump_end, shoot_end = 46, 30, 5
      else period, jump_end, shoot_end = 44, 28, 4 end
      local phase = frame % period
      local inputs
      if phase < shoot_end then inputs = {"RIGHT", "B", "X"}
      elseif phase < shoot_end + 3 then inputs = {"RIGHT", "B"}
      elseif phase < shoot_end + 3 + jump_end then inputs = {"RIGHT", "B", "A"}
      else inputs = {"RIGHT", "B"} end
      state = session:hold(1, inputs, label .. "_ridge")
      if state.room_id == ROOM_RISING then
        hit = true
        break
      end
    end
  end
  if session.state.room_id ~= ROOM_RISING then
    local state = session.state
    error(string.format(
      "%s: right green Super door missed; room=0x%04X pose=%d xy=(%d,%d) max_x=%d min_y=%d high=%s door=%s",
      label, state.room_id, state.pose, state.samus_x, state.samus_y,
      max_x, min_y, tostring(high_reached), tostring(door_reached)
    ))
  end
  return iceg.wait_ordinary_room(session, ROOM_RISING, 320, label)
end

M.play_business_to_cathedral_entrance = play_business_to_cathedral_entrance
return M
