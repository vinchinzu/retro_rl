-- Rising Tide → Bubble Mountain. Charged HJ when low; RIGHT+B+X door pressure.
-- Lua 5.1. Session: step / hold / wait_until / span.

local iceg = require("ice.geometry")
local knockback = require("skills.knockback")
local geom = require("skills.geometry")

local ROOM_BUBBLE = 0xACB3
local ROOM_RISING = 0xAFA3
local STANDING = geom.STANDING_POSES or iceg.STANDING_POSES
local CROSS_FRAMES = 5000
local SETTLE = 320

local M = {}

local function land_and_arm(session, label)
  for _ = 1, 40 do
    local state = session:hold(1, {}, label .. "_land")
    if state.velocity_y == 0 and STANDING[state.pose] then
      break
    end
  end
  iceg.unmorph(session)
  iceg.select_weapon(session, 0)
end

local function cross_to_bubble(session, label)
  local max_x = session.state.samus_x
  local min_y = session.state.samus_y
  local door_reached = false
  local stuck_frames = 0
  local last_x = session.state.samus_x
  local hit = false

  for frame = 0, CROSS_FRAMES - 1 do
    local state = session.state
    if state.room_id == ROOM_BUBBLE then
      hit = true
      break
    end
    if state.samus_x > max_x then
      max_x = state.samus_x
    end
    if state.samus_y < min_y then
      min_y = state.samus_y
    end
    if math.abs(state.samus_x - last_x) <= 1 then
      stuck_frames = stuck_frames + 1
    else
      stuck_frames = 0
      last_x = state.samus_x
    end

    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "RIGHT",
        run_frames = 6,
        spin_frames = 20,
        label = label,
        stop_room_id = ROOM_BUBBLE,
      })
      stuck_frames = 0
      last_x = session.state.samus_x
    elseif state.samus_x >= 930 then
      door_reached = true
      if state.selected_item ~= 0 then
        iceg.select_weapon(session, 0)
      end
      if state.samus_y > 170 then
        if state.samus_x > 1040 then
          session:hold(1, {"LEFT", "B"}, label .. "_under_back")
        elseif state.velocity_y == 0 and STANDING[state.pose] then
          for _ = 1, 14 do
            session:hold(1, {"A"}, label .. "_door_charge")
          end
          for _ = 1, 40 do
            local st = session:hold(1, {"RIGHT", "B", "A"}, label .. "_door_up")
            if st.room_id == ROOM_BUBBLE then
              break
            end
          end
        else
          session:hold(1, {"RIGHT", "B", "A"}, label .. "_under_hop")
        end
      else
        local phase = frame % 16
        local inputs
        if phase < 8 then
          inputs = {"RIGHT", "B", "X"}
        elseif phase < 12 then
          inputs = {"RIGHT", "B", "A"}
        else
          inputs = {"RIGHT", "B"}
        end
        state = session:hold(1, inputs, label .. "_door")
        if state.room_id == ROOM_BUBBLE then
          hit = true
          break
        end
      end
    elseif state.velocity_y == 0 and state.samus_y > 150 and STANDING[state.pose] then
      for _ = 1, 12 do
        session:hold(1, {"A"}, label .. "_charge")
      end
      for _ = 1, 32 do
        local st = session:hold(1, {"RIGHT", "B", "A"}, label .. "_hj")
        if st.room_id == ROOM_BUBBLE then
          break
        end
      end
    elseif stuck_frames > 40 then
      for _ = 1, 10 do
        session:hold(1, {"LEFT", "B"}, label .. "_unstick_back")
      end
      for _ = 1, 10 do
        session:hold(1, {"A"}, label .. "_unstick_charge")
      end
      for _ = 1, 35 do
        session:hold(1, {"RIGHT", "B", "A"}, label .. "_unstick_jump")
      end
      stuck_frames = 0
      last_x = session.state.samus_x
    else
      if state.selected_item ~= 0 then
        iceg.select_weapon(session, 0)
      end
      local phase = frame % 32
      local inputs
      if phase < 3 then
        inputs = {"RIGHT", "B", "X"}
      elseif phase < 22 then
        inputs = {"RIGHT", "B", "A"}
      else
        inputs = {"RIGHT", "B"}
      end
      state = session:hold(1, inputs, label .. "_cross")
      if state.room_id == ROOM_BUBBLE then
        hit = true
        break
      end
    end
  end

  if session.state.room_id ~= ROOM_BUBBLE then
    local state = session.state
    error(string.format(
      "%s: right blue door missed before room 0x%04X; room=0x%04X pose=%d xy=(%d,%d) max_x=%d min_y=%d door_reached=%s",
      label, ROOM_BUBBLE, state.room_id, state.pose, state.samus_x, state.samus_y,
      max_x, min_y, tostring(door_reached)
    ))
  end
end

function M.play_rising_tide_to_bubble(session)
  local label = "rising_tide_to_bubble"
  iceg.require_room(session, ROOM_RISING, label)
  land_and_arm(session, label)
  cross_to_bubble(session, label)
  return iceg.wait_ordinary_room(session, ROOM_BUBBLE, SETTLE, label)
end

return M
