-- Moonwalk + moonfall builders. Callers enable $09E4 first.

local ram = require("ram")

local moonfall = {}

moonfall.MOVEMENT_JUMPING = 0x02
moonfall.MOVEMENT_SPIN = 0x03
moonfall.MOVEMENT_FALLING = 0x06
moonfall.MOVEMENT_MOONWALKING = 0x10
moonfall.AIR_MOVEMENT = {
  [0x02] = true,
  [0x03] = true,
  [0x06] = true,
}
moonfall.MOONWALK_TURN_POSES = {
  [0xC0] = true,
  [0xC1] = true,
  [0xC2] = true,
  [0xC3] = true,
  [0xC4] = true,
}
moonfall.NORMAL_FALL_CAP_PX = 5
moonfall.WIKI_URL = "https://wiki.supermetroid.run/Moonwalk"

function moonfall.moonwalk_direction(facing)
  if facing == ram.FACING_RIGHT then
    return "LEFT"
  end
  return "RIGHT"
end

function moonfall.angle_button(aim)
  if aim == "UP" then
    return "R"
  end
  return "L"
end

function moonfall.moonwalk_buttons(facing, opts)
  opts = opts or {}
  local extra = opts.extra or {}
  local names = {
    moonfall.moonwalk_direction(facing),
    "X",
    moonfall.angle_button(opts.aim or "DOWN"),
  }
  for i = 1, #extra do
    names[#names + 1] = extra[i]
  end
  return names
end

function moonfall.is_moonwalking(state)
  return tonumber(state.movement_type) == moonfall.MOVEMENT_MOONWALKING
end

function moonfall.is_airborne(state)
  return moonfall.AIR_MOVEMENT[tonumber(state.movement_type) or -1] or false
end

function moonfall.is_moonfalling(state)
  return tonumber(state.vertical_direction) == 0 and moonfall.is_airborne(state)
end

function moonfall.uncapped_fall(state)
  return moonfall.is_moonfalling(state)
    and (tonumber(state.velocity_y) or 0) > moonfall.NORMAL_FALL_CAP_PX
end

function moonfall.require_moonwalk_on(state, label)
  label = label or "moonfall"
  local enabled = state.moonwalk_enabled and state:moonwalk_enabled() or (state.moonwalk ~= 0)
  if not enabled then
    error(string.format(
      "%s: moonwalk flag $09E4 is off (need Special Setting Mode or ram.set_moonwalk). %s",
      label,
      moonfall.WIKI_URL
    ))
  end
end

function moonfall.initiate_moonfall(session, opts)
  opts = opts or {}
  local aim = opts.aim or "DOWN"
  local spin = opts.spin
  if spin == nil then
    spin = true
  end
  local walk_frames = opts.walk_frames or 10
  local jump_frames = opts.jump_frames or 2
  local release_frames = opts.release_frames or 2
  local timeout = opts.timeout or 40
  local reason = opts.reason or "moonfall"
  moonfall.require_moonwalk_on(session.state, reason)
  local facing = tonumber(session.state.facing) or ram.FACING_LEFT
  if facing == 0 then
    facing = ram.FACING_LEFT
  end
  local walk = moonfall.moonwalk_direction(facing)
  local angle = moonfall.angle_button(aim)
  session:hold(walk_frames, { walk, "X", angle }, reason .. "_moonwalk")
  session:hold(jump_frames, { walk, "X", angle, "A" }, reason .. "_jump")
  if spin then
    session:hold(release_frames, { walk, "A" }, reason .. "_spin")
  else
    session:hold(release_frames, { walk, angle, "A" }, reason .. "_held_angle")
  end
  local state = session.state
  for _ = 1, timeout do
    if moonfall.is_moonfalling(state) then
      return state
    end
    state = session:hold(1, { walk }, reason .. "_wait")
  end
  return state
end

function moonfall.fall_until(session, done, opts)
  opts = opts or {}
  local steer = opts.steer
  local timeout = opts.timeout or 400
  local reason = opts.reason or "moonfall_fall"
  local state = session.state
  for _ = 1, timeout do
    if done(state) then
      return state
    end
    local names = {}
    if steer then
      local d = steer(state)
      if d then
        names = { d }
      end
    end
    state = session:hold(1, names, reason)
  end
  error(string.format(
    "TimeoutError: %s timed out at frame %s: %s",
    reason,
    tostring(session.frame),
    tostring(session.state)
  ), 2)
end

return moonfall
