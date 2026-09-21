-- Pure Business Center → Ice Beam Gate (Super green LEFT).
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("ice.geometry")
local climb = require("norfair.business_climb")
local knockback = require("skills.knockback")
local door = require("skills.door")

local ROOM_BUSINESS = G.ROOM_BUSINESS
local ROOM_HJ = G.ROOM_HJ_SHAFT
local ROOM_GATE = G.ROOM_ICE_GATE
local FLOOR_Y_MIN = 1350
local LEDGE = G.LEDGE_POSES

local M = {}

local function settle_elevator(session, label)
  local elev_lo, elev_hi = G.BUSINESS_ELEVATOR_Y - 5, G.BUSINESS_ELEVATOR_Y + 10
  local stable = 0
  for _ = 1, G.ELEVATOR_SETTLE_FRAMES do
    local state = session:hold(1, {}, label .. "_elevator_settle")
    local y = state.samus_y
    if elev_lo <= y and y <= elev_hi and state.velocity_y == 0 then
      stable = stable + 1
      if stable >= 24 then
        return
      end
    else
      stable = 0
    end
  end
  if session.state.samus_y > G.BUSINESS_ELEVATOR_Y + 40 then
    return
  end
  if session.state.samus_y < elev_lo then
    return
  end
  error(label .. ": elevator did not settle")
end

local function anchor_floor(session, label)
  G.unmorph(session)
  for _ = 1, 200 do
    local st = session.state
    if st.room_id ~= ROOM_BUSINESS then
      error(label .. ": left Business during floor anchor")
    end
    local x, y = st.samus_x, st.samus_y
    if y < FLOOR_Y_MIN - 20 then
      return
    end
    local grounded = st.velocity_y == 0 and LEDGE[st.pose]
    if 200 <= x and x <= 240 and grounded then
      session:hold(12, {}, label .. "_floor_anchor_settle")
      return
    end
    if x < 200 then
      session:hold(1, {"RIGHT", "B"}, label .. "_floor_anchor_r")
    elseif x > 240 and grounded then
      if x > 80 then
        session:hold(1, {"LEFT"}, label .. "_floor_anchor_l")
      else
        session:hold(1, {"RIGHT"}, label .. "_floor_anchor_bounce")
      end
    else
      session:hold(1, {}, label .. "_floor_anchor_idle")
    end
  end
  error(label .. ": floor climb anchor missed")
end

local function drop_to_super_band(session, label)
  local hit = false
  for frame = 0, G.DOOR_BAND_FRAMES - 1 do
    local state = session.state
    if state.room_id ~= ROOM_BUSINESS then
      return
    end
    if G.on_ice_super_lip(state) then
      return
    end
    if G.ICE_SUPER_Y_MIN <= state.samus_y and state.samus_y <= G.ICE_SUPER_Y_MAX
        and state.velocity_y == 0 and LEDGE[state.pose]
        and state.samus_x <= G.ICE_SUPER_LIP_X_MAX + 40 then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT", run_frames = 4, spin_frames = 12, label = label .. "_kb",
      })
    else
      local y = state.samus_y
      local buttons
      if y < G.ICE_SUPER_Y_MIN - 40 then
        local direction = (math.floor(frame / 40) % 2 == 0) and "LEFT" or "RIGHT"
        if frame % 40 < 10 then
          buttons = {direction, "B", "A"}
        else
          buttons = {direction, "B"}
        end
      elseif y < G.ICE_SUPER_Y_MIN then
        if frame % 28 < 8 then
          buttons = {"LEFT", "B", "A"}
        else
          buttons = {"LEFT", "B"}
        end
      elseif state.samus_x > G.ICE_SUPER_LIP_X_MAX then
        if frame % 24 < 6 and state.velocity_y == 0 then
          buttons = {"LEFT", "B", "A"}
        else
          buttons = {"LEFT", "B"}
        end
      else
        buttons = (state.velocity_y == 0) and {"LEFT"} or {"LEFT", "B"}
      end
      session:hold(1, buttons, label .. "_door_band")
    end
  end
  error(string.format("%s: Ice Super door band missed (want y∈[%d,%d] x≤%d)",
    label, G.ICE_SUPER_Y_MIN, G.ICE_SUPER_Y_MAX, G.ICE_SUPER_LIP_X_MAX))
end

local function open_ice_super_and_enter(session, label)
  G.unmorph(session)
  if session.state.selected_item ~= 2 then
    G.select_weapon(session, 2)
  end
  session:hold(4, {}, label .. "_super_ready")
  for frame = 0, G.SUPER_PRESSURE_FRAMES - 1 do
    local state = session.state
    if state.room_id == ROOM_GATE then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT", run_frames = 3, spin_frames = 10, label = label .. "_door_kb",
      })
      if session.state.selected_item ~= 2 then
        G.select_weapon(session, 2)
      end
    elseif state.samus_x > G.ICE_SUPER_LIP_X_MAX + 20
        and G.ICE_SUPER_Y_MIN <= state.samus_y and state.samus_y <= G.ICE_SUPER_Y_MAX then
      session:hold(1, {"LEFT", "B"}, label .. "_reseat")
    elseif state.samus_x <= G.ICE_SUPER_DOOR_X + 30 then
      state = door.super_door_pressure_frame(session, frame, {
        label = label, face = "LEFT", period = 28, shoot_end = 6, face_end = 14, run_end = 22,
      })
      if state.room_id == ROOM_GATE then
        return
      end
    else
      local phase = frame % 24
      if phase < 6 then
        session:hold(1, {"LEFT", "X"}, label .. "_super_plant")
      elseif phase < 12 then
        session:hold(1, {"LEFT"}, label .. "_face")
      else
        session:hold(1, {"LEFT", "B"}, label .. "_approach")
      end
    end
  end
  error(label .. ": Ice Super door did not open")
end

local function climb_floor_to_elevator(session, label)
  G.unmorph(session)
  if session.state.room_id ~= ROOM_BUSINESS then
    error(label .. ": not in Business for floor climb")
  end
  anchor_floor(session, label)
  local attempts = {{18, 90, false}, {20, 90, false}, {14, 84, false}, {8, 84, false}}
  local last_err
  for i, row in ipairs(attempts) do
    local ok, err = pcall(function()
      if i > 1 and session.state.room_id ~= ROOM_BUSINESS then
        error(label .. ": left Business during floor climb")
      end
      climb.business_high_jump_platforms(session, row[1], row[2], row[3])
    end)
    if ok then
      last_err = nil
      break
    else
      last_err = err
      if session.state.room_id ~= ROOM_BUSINESS then
        error(label .. ": left Business during floor climb")
      end
    end
  end
  if last_err then
    error(last_err)
  end
  if session.state.samus_y > G.BUSINESS_ELEVATOR_Y + 40 then
    error(string.format("%s: floor climb missed elevator (y=%d)", label, session.state.samus_y))
  end
end

function M.play_business_to_ice_gate(session)
  local label = "business_to_ice_gate"
  G.require_room(session, ROOM_BUSINESS, label)
  local y0 = session.state.samus_y
  local climbed = false
  if y0 >= FLOOR_Y_MIN or y0 > G.ICE_SUPER_Y_MAX + 40 then
    if not G.on_ice_super_lip(session.state) then
      climb_floor_to_elevator(session, label)
      climbed = true
    end
  end
  local y1 = session.state.samus_y
  local on_elev = G.BUSINESS_ELEVATOR_Y - 5 <= y1 and y1 <= G.BUSINESS_ELEVATOR_Y + 10
    and session.state.velocity_y == 0 and LEDGE[session.state.pose]
  if not (climbed and on_elev) then
    if y1 < G.BUSINESS_ELEVATOR_Y + 80 or session.state.pose == 155 then
      if not on_elev then
        settle_elevator(session, label)
      end
    end
  end
  if not G.on_ice_super_lip(session.state) then
    drop_to_super_band(session, label)
  end
  open_ice_super_and_enter(session, label)
  local state = G.wait_ordinary_room(session, ROOM_GATE, G.ICE_GATE_SETTLE_FRAMES, label)
  G.unmorph(session)
  for _ = 1, 40 do
    local st = session:hold(1, {}, label .. "_stand")
    local p = st.pose
    if st.velocity_y == 0 and (p == 1 or p == 2 or p == 9 or p == 10) and st.door_transition == 0 then
      return st
    end
  end
  return state
end

return M
