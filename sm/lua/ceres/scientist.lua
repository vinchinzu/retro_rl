-- Dead Scientist Room 0xE021: arm-pump run, never jump.
--
-- https://wiki.supermetroid.run/Dead_Scientist_Room — two raised door ledges,
-- stairs into a pit, stairs out. Sniq 100% lsnes never presses A either way:
-- outbound gs=8 f9577→9680 RIGHT+B+L/R off (39,139) p9; reverse f12198→12299
-- LEFT+B+L/R off (472,139) p18. A on the alcove bonks. The floor hop window
-- was the +20f / pose-137 stall versus TAS.
--
-- Fade already matches (161 vs 161 outbound, 162 vs 162 reverse). Leftover
-- dwell is the door-lip stall (outbound x=467 pose 207; reverse x=45 pose 208).

local ram = require("ram")
local rooms = require("rooms")
local takeoff = require("takeoff")
local knockback = require("skills.knockback")
local geom = require("ceres.geometry")
local arm_pump = require("ceres.arm_pump")

local GS_ORDINARY = ram.GS_ORDINARY or 8

local function timeout(msg)
  local ok, runtime = pcall(require, "runtime")
  if ok and runtime and runtime.timeout then
    runtime.timeout(msg)
  end
  error(msg)
end

local function format_state(st)
  if type(st) ~= "table" then
    return tostring(st)
  end
  return string.format(
    "room=%s gs=%s xy=(%s,%s) pose=%s",
    tostring(st.room_id),
    tostring(st.game_state),
    tostring(st.samus_x),
    tostring(st.samus_y),
    tostring(st.pose)
  )
end

local function step_names(session, names, reason)
  if names and names[1] then
    session:step(names, reason)
  else
    session:idle(reason)
  end
end

-- True on the left door alcove.
local function scientist_on_entry_ledge(state)
  return state.room_id == rooms.ROOM_CERES_SCIENTIST
    and math.abs(state.samus_y - geom._CERES_SCI_DOOR_Y) <= 16
    and state.samus_x <= geom._CERES_SCI_ENTRY_LEDGE_X
end

local CeresScientistCross = {}
CeresScientistCross.__index = CeresScientistCross

-- One-frame Dead Scientist Room policy. Never jump.
-- Outbound (RIGHT): Sniq 100% lsnes gs=8 f9577→door f9680.
-- Reverse (LEFT): gs=8 f12198→door f12299. Stairs + y=187 pit, no A.
function CeresScientistCross.new(direction)
  direction = direction or "RIGHT"
  if direction ~= "LEFT" and direction ~= "RIGHT" then
    error(string.format(
      "scientist direction must be LEFT or RIGHT, got %q",
      tostring(direction)
    ))
  end
  return setmetatable({
    direction = direction,
    pump_i = 0,
  }, CeresScientistCross)
end

function CeresScientistCross:action(state)
  if state.game_state ~= GS_ORDINARY then
    return {self.direction}
  end
  local names = {self.direction, "B", takeoff.shoulder_pump_button(self.pump_i)}
  self.pump_i = self.pump_i + 1
  return names
end

setmetatable(CeresScientistCross, {
  __call = function(_, direction)
    return CeresScientistCross.new(direction)
  end,
})

-- True in Flat/Ridley ordinary — not the scientist→flat door (gs 9/11).
local function scientist_past(state)
  if state.game_state ~= GS_ORDINARY then
    return false
  end
  local room = state.room_id
  return room == rooms.ROOM_CERES_FLAT or room == rooms.ROOM_CERES_RIDLEY
end

-- Scientist ordinary → Flat (or Ridley if the door overshoots).
--
-- No-op when already past the room. Waits out the magnet→scientist door
-- before treating x-stagnation as a ledge.
local function play_scientist_to_flat(session)
  if scientist_past(session.state) then
    return
  end
  local entered = false
  for _ = 1, 160 do
    local st = session.state
    if scientist_past(st) then
      return
    end
    if st.room_id == rooms.ROOM_CERES_SCIENTIST and st.game_state == GS_ORDINARY then
      entered = true
      break
    end
    session:step({"RIGHT"}, "ceres_sci_door")
  end
  if not entered then
    local st = session.state
    if st.room_id ~= rooms.ROOM_CERES_SCIENTIST then
      timeout("ceres scientist ordinary missed: " .. format_state(st))
    end
  end

  local cross = CeresScientistCross()
  for _ = 1, 400 do
    local st = session.state
    if scientist_past(st) then
      return
    end
    if knockback.is_knockback(st) then
      arm_pump.clear_knockback(session, "RIGHT", "ceres_sci")
    else
      local names = cross:action(st)
      local reason
      if scientist_on_entry_ledge(st) then
        reason = "ceres_sci_ledge"
      elseif st.game_state ~= GS_ORDINARY then
        reason = "ceres_sci_fade"
      else
        reason = "ceres_sci"
      end
      step_names(session, names, reason)
    end
  end
  timeout("ceres scientist missed Flat: " .. format_state(session.state))
end

-- True in Magnet/Falling ordinary — not the scientist→magnet door.
local function scientist_escape_past(state)
  if state.game_state ~= GS_ORDINARY then
    return false
  end
  local room = state.room_id
  return room == rooms.ROOM_CERES_MAGNET or room == rooms.ROOM_CERES_FALLING
end

-- Scientist ordinary → Magnet (or Falling if the door overshoots).
--
-- Reverse Ceres 4. TAS dwell (f12198–12299) never presses A: LEFT+B+L/R
-- off (472,139) p18, down the east stairs, across y=187, up the west
-- stairs into the left door. stuck-jump on the stairs is leftover.
local function play_scientist_to_magnet(session)
  if scientist_escape_past(session.state) then
    return
  end
  local entered = false
  for _ = 1, 180 do
    local st = session.state
    if scientist_escape_past(st) then
      return
    end
    if st.room_id == rooms.ROOM_CERES_SCIENTIST and st.game_state == GS_ORDINARY then
      entered = true
      break
    end
    session:step({"LEFT"}, "ceres_sci_rev_door")
  end
  if not entered then
    local st = session.state
    if st.room_id ~= rooms.ROOM_CERES_SCIENTIST then
      timeout("ceres scientist reverse ordinary missed: " .. format_state(st))
    end
  end

  local cross = CeresScientistCross("LEFT")
  for _ = 1, 400 do
    local st = session.state
    if scientist_escape_past(st) then
      return
    end
    if knockback.is_knockback(st) then
      arm_pump.clear_knockback(session, "LEFT", "ceres_sci_rev")
    else
      local names = cross:action(st)
      local reason
      if st.game_state ~= GS_ORDINARY then
        reason = "ceres_sci_rev_fade"
      else
        reason = "ceres_sci_rev"
      end
      step_names(session, names, reason)
    end
  end
  timeout("ceres scientist reverse missed Magnet: " .. format_state(session.state))
end

return {
  scientist_on_entry_ledge = scientist_on_entry_ledge,
  CeresScientistCross = CeresScientistCross,
  play_scientist_to_flat = play_scientist_to_flat,
  play_scientist_to_magnet = play_scientist_to_magnet,
}
