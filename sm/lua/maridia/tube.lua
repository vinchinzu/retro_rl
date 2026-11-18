-- Gravity door, then stop. The Maridia tube is not measured from here.
-- bowling_v2 ends LEFT and tapes/gravity.lua begins B+LEFT, so the only
-- justified Gravity exit is the opposite direction (RIGHT). ws_main_departure
-- pins are the Main Shaft grate, not this door. alpha_pb_escape lands
-- Caterpillar and then runs toward the ship, not the tube. red_tower
-- glass-door pixels (x=40) are the K5 westbound tunnel, not Main Street.

local M = {}

local GRAVITY = 0xCE40
local MAIN_STREET = 0xCFC9
local CATERPILLAR = 0xA322
local BUSINESS = 0xA7DE

-- Moat hop shape (stand, micro-run, spin). One try, then stop.
local STAND_FRAMES, MICRO_RUN, HOP_FRAMES = 4, 2, 14
local WALK_BUDGET = 360
local STALL_LIMIT = 45
local UNMORPH_CAP = 8
local Y_BAND = 96

local MORPH = {
  [29] = true, [30] = true, [31] = true, [32] = true,
  [49] = true, [50] = true, [65] = true, [66] = true,
}
local CROUCH = {[39] = true, [40] = true}

local function num(v)
  return tonumber(v) or 0
end

local function ids_of(ids)
  ids = ids or {}
  return {
    GRAVITY = ids.GRAVITY or GRAVITY,
    MAIN_STREET = ids.MAIN_STREET or MAIN_STREET,
    CATERPILLAR = ids.CATERPILLAR or CATERPILLAR,
    BUSINESS = ids.BUSINESS or BUSINESS,
  }
end

local function snap(st)
  st = st or {}
  return {
    room = num(st.room_id),
    gs = num(st.game_state),
    x = num(st.samus_x or st.x),
    y = num(st.samus_y or st.y),
    pose = num(st.pose),
    door = num(st.door_transition),
    vy = num(st.velocity_y),
    kb = num(st.knockback_timer),
  }
end

local function unproven(s, why, frame)
  error(string.format(
    "UNPROVEN: %s; room=0x%04X xy=(%d,%d) pose=%d frame=%s",
    why, s.room, s.x, s.y, s.pose, tostring(frame)
  ), 2)
end

local function dead(session, s)
  if s.gs == 26 or s.gs == 36 then
    error(string.format(
      "TimeoutError: gravity_to_main_street: Samus died in room 0x%04X xy=(%d,%d) pose=%d frame=%s",
      s.room, s.x, s.y, s.pose, tostring(session.frame)
    ), 2)
  end
end

-- Step once. Returns true when the room is no longer Gravity.
local function step_in_gravity(session, ids, names, reason)
  local s = snap(session.state)
  dead(session, s)
  if s.room ~= ids.GRAVITY then
    return true
  end
  session:step(names, reason)
  s = snap(session.state)
  dead(session, s)
  return s.room ~= ids.GRAVITY
end

local function hop_right(session, ids)
  local i
  for i = 1, STAND_FRAMES do
    if step_in_gravity(session, ids, {}, "gravity_hop_stand") then
      return true
    end
  end
  for i = 1, MICRO_RUN do
    if step_in_gravity(session, ids, {"RIGHT", "B"}, "gravity_hop_run") then
      return true
    end
  end
  for i = 1, HOP_FRAMES do
    if step_in_gravity(session, ids, {"RIGHT", "B", "A"}, "gravity_hop") then
      return true
    end
  end
  return false
end

function M.leave_gravity(session, ids)
  ids = ids_of(ids)
  local origin = snap(session.state)
  local best_x = origin.x
  local stale, up_frames, hopped, walked = 0, 0, false, 0
  while walked < WALK_BUDGET do
    local s = snap(session.state)
    -- Read room / x / y / pose every iteration before choosing a button.
    local room, x, y, pose = s.room, s.x, s.y, s.pose
    dead(session, s)
    if room ~= ids.GRAVITY then
      return session.state
    end
    if s.door ~= 0 or s.gs ~= 8 then
      session:wait_until(function(st)
        local now = snap(st)
        return now.room ~= ids.GRAVITY or (now.gs == 8 and now.door == 0)
      end, 120, "gravity_door_settle")
    elseif (MORPH[pose] or CROUCH[pose]) and up_frames < UNMORPH_CAP then
      up_frames = up_frames + 1
      if step_in_gravity(session, ids, {"UP"}, "gravity_unmorph") then
        return session.state
      end
    elseif s.kb > 0 then
      session:idle("gravity_kb")
    else
      up_frames = 0
      if x > best_x + 1 then
        best_x = x
        stale = 0
      else
        stale = stale + 1
      end
      if s.vy == 0 and math.abs(y - origin.y) > Y_BAND then
        unproven(s, "left the Gravity pedestal band without a measured door lip", session.frame)
      end
      if stale >= STALL_LIMIT then
        if hopped then
          unproven(s, "Gravity return walk stalled before a measured door lip", session.frame)
        end
        if hop_right(session, ids) then
          return session.state
        end
        hopped = true
        stale = 0
        best_x = snap(session.state).x
      else
        if step_in_gravity(session, ids, {"RIGHT"}, "gravity_return_walk") then
          return session.state
        end
      end
    end
    walked = walked + 1
  end
  unproven(
    snap(session.state),
    "Gravity return budget ended still in 0xCE40; tube path is not measured",
    session.frame
  )
end

function M.stop_unproven(session, ids)
  ids = ids_of(ids)
  local s = snap(session.state)
  local why
  if s.room == ids.MAIN_STREET then
    why = "Main Street visible but this module does not claim the tube"
  elseif s.room == ids.CATERPILLAR then
    why = "Caterpillar is the alpha_pb_escape return seat; no measured pixels from here into the Maridia tube"
  elseif s.room == ids.BUSINESS then
    why = "Business has no measured Maridia-tube pixels in the wrecked-ship return files"
  elseif s.room == ids.GRAVITY then
    why = "still in Gravity; the right-hand door lip is not measured"
  else
    why = "crossed the Gravity door (inbound tapes move LEFT; this walk moved RIGHT) and stopped; no pixel path through bowling/attic/west ocean/red tower/caterpillar into the tube"
  end
  unproven(s, why, session.frame)
end

return M
