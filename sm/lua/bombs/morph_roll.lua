-- Morph helpers + bomb-roll. Bomb-while-morph is X, never A.
--
-- Pose-confirmed morph is a DOWN double-tap (held DOWN only crouches).
-- Port of controller_common.ensure_morph / morph_bomb_roll.bomb_roll_left_safe
-- / routes.skills.morph_bomb.

local ram = require("ram")

local M = {}

M.MORPH_POSES = {
  [29] = true, [30] = true, [31] = true, [32] = true,
  [49] = true, [50] = true, [65] = true, [66] = true,
}

-- Double-tap morph timing (MorphPolicy).
M.MAX_ATTEMPTS = 5
M.BASE_UP = 4
M.UP_STEP = 2
M.MAX_UP_EXTRA_STEPS = 3
M.BASE_IDLE = 3
M.BASE_TAP1 = 5
M.TAP1_STEP = 2
M.BASE_TAP2 = 6
M.TAP2_STEP = 3
M.RELEASE = 3
M.BASE_POLL = 28
M.POLL_STEP = 4

-- Bomb-roll defaults (Pink wall-open cycle).
M.DEFAULT_MAX_Y = 415
M.DEFAULT_PIT_Y = 430
M.DEFAULT_ELEV_Y = 400
M.DEEP_PIT_Y = 445
M.BOMB_CYCLE = 38

local function num(v, default)
  v = tonumber(v)
  if v == nil then
    return default or 0
  end
  return v
end

local function i16(v)
  v = num(v)
  if v >= 0x8000 then
    return v - 0x10000
  end
  return v
end

local function timeout(msg)
  local ok, runtime = pcall(require, "runtime")
  if ok and runtime and runtime.timeout then
    runtime.timeout(msg)
  end
  error("TimeoutError: " .. msg)
end

local function brief(state)
  if type(state) ~= "table" then
    return tostring(state)
  end
  return string.format(
    "room=0x%04X gs=%s xy=%s,%s pose=%s",
    num(state.room_id),
    tostring(state.game_state),
    tostring(state.samus_x or state.x),
    tostring(state.samus_y or state.y),
    tostring(state.pose)
  )
end

-- session:hold(frames, names, reason) when present; else step loop.
local function hold(session, frames, names, reason)
  names = names or {}
  if frames <= 0 then
    return session.state
  end
  if session.hold then
    session:hold(frames, names, reason)
    return session.state
  end
  local i
  for i = 1, frames do
    session:step(names, reason)
  end
  return session.state
end

local function wait_until(session, pred, timeout_frames, reason)
  timeout_frames = timeout_frames or 120
  if session.wait_until then
    return session:wait_until(pred, timeout_frames, reason)
  end
  local waited
  for waited = 0, timeout_frames do
    if pred(session.state) then
      return waited
    end
    session:step({}, reason)
  end
  timeout(reason .. " timed out at frame " .. tostring(session.state.frame or session.frame)
    .. ": " .. brief(session.state))
end

function M.is_morph(pose)
  return M.MORPH_POSES[num(pose)] == true
end

function M.unmorph(session)
  local pose = num(session.state.pose)
  if pose == 39 or pose == 40 or pose == 137 or pose == 138
      or pose == 9 or pose == 10 or M.is_morph(pose) then
    hold(session, 8, {"UP"}, "unmorph")
    if M.is_morph(session.state.pose) then
      hold(session, 8, {"A"}, "unmorph")
    end
    hold(session, 10, {}, "unmorph_settle")
  end
end

-- Lay a bomb. Morph bomb is X (A is jump / unmorph).
function M.lay_bomb(session, frames, reason)
  return hold(session, frames or 2, {"X"}, reason or "morph_bomb")
end

function M.ensure_morph(session, max_attempts)
  local attempts = max_attempts or M.MAX_ATTEMPTS
  local attempt
  for attempt = 0, attempts - 1 do
    if M.is_morph(session.state.pose) then
      return session.state
    end
    local extra = attempt
    if extra > M.MAX_UP_EXTRA_STEPS then
      extra = M.MAX_UP_EXTRA_STEPS
    end
    hold(session, M.BASE_UP + extra * M.UP_STEP, {"UP"}, "morph_pre")
    hold(session, M.BASE_IDLE + attempt, {}, "morph_idle")
    hold(session, M.BASE_TAP1 + attempt * M.TAP1_STEP, {"DOWN"}, "morph_tap1")
    hold(session, M.RELEASE, {}, "morph_release")
    hold(session, M.BASE_TAP2 + attempt * M.TAP2_STEP, {"DOWN"}, "morph_tap2")
    local ok = pcall(wait_until, session, function(s)
      return M.is_morph(s.pose)
    end, M.BASE_POLL + attempt * M.POLL_STEP, "morph_poll")
    if ok and M.is_morph(session.state.pose) then
      return session.state
    end
  end
  timeout("ensure_morph failed, pose=" .. tostring(session.state.pose))
end

function M.align_x(session, x_lo, x_hi, label, max_frames, settle_frames, guard)
  max_frames = max_frames or 50
  settle_frames = settle_frames or 0
  local i
  for i = 1, max_frames do
    local state = session.state
    if guard then
      guard(state)
    end
    local x = num(state.samus_x or state.x)
    if x_lo <= x and x <= x_hi then
      break
    end
    local dir
    if x < x_lo then
      dir = "RIGHT"
    else
      dir = "LEFT"
    end
    hold(session, 1, {dir}, label .. "_align")
  end
  if settle_frames > 0 then
    hold(session, settle_frames, {}, label .. "_align_settle")
  end
  return session.state
end

-- Morph-bomb-roll left toward target_x with y-band / pit recovery.
function M.bomb_roll_left_safe(session, target_x, opts)
  opts = opts or {}
  local max_y = opts.max_y or M.DEFAULT_MAX_Y
  local pit_y = opts.pit_y or M.DEFAULT_PIT_Y
  local max_frames = opts.max_frames or 280
  local cycle_len = opts.cycle_len or M.BOMB_CYCLE
  local elev_y = opts.elev_y or M.DEFAULT_ELEV_Y
  local stall_frames = opts.stall_frames or 0
  local deep_pit_y = M.DEEP_PIT_Y
  local frames = 0
  local last_progress_x = num(session.state.samus_x or session.state.x)
  local frames_since_progress = 0
  local pit_recoveries = 0

  while num(session.state.samus_x or session.state.x) > target_x and frames < max_frames do
    local s = session.state
    if num(s.max_power_bombs) > 0 then
      return s
    end
    local vy = i16(s.velocity_y)
    local y = num(s.samus_y or s.y)
    local falling_hard = vy > 80
    local in_pit = y > pit_y or falling_hard
    local deep_pit = y > deep_pit_y

    if in_pit then
      pit_recoveries = pit_recoveries + 1
      if deep_pit then
        hold(session, 4, {"UP"}, "pit_unmorph")
        local r
        for r = 1, 18 do
          hold(session, 1, {"RIGHT"}, "pit_right")
          if num(session.state.samus_y or session.state.y) <= pit_y then
            break
          end
        end
        if (not M.is_morph(session.state.pose))
            or num(session.state.samus_y or session.state.y) <= pit_y + 5 then
          hold(session, 6, {"A", "RIGHT"}, "pit_jump")
          hold(session, 8, {}, "pit_settle")
        end
        if not M.is_morph(session.state.pose) then
          pcall(M.ensure_morph, session)
        end
        frames = frames + 36
      else
        hold(session, 6, {"UP"}, "pit_unmorph")
        hold(session, 8, {"A", "RIGHT"}, "pit_jump")
        hold(session, 10, {}, "pit_settle")
        pcall(M.ensure_morph, session)
        frames = frames + 24
      end
      if deep_pit and pit_recoveries >= 3
          and num(session.state.samus_x or session.state.x) >= last_progress_x - 2 then
        return session.state
      end
    elseif not M.is_morph(session.state.pose) then
      M.ensure_morph(session)
      frames = frames + 20
    elseif stall_frames > 0 and frames_since_progress >= stall_frames then
      M.lay_bomb(session, 2, "safe_watchdog_bomb")
      hold(session, 10, {}, "safe_watchdog_pause")
      frames = frames + 12
      frames_since_progress = 0
    else
      M.lay_bomb(session, 2, "safe_bomb")
      frames = frames + 2
      local step
      for step = 0, cycle_len - 1 do
        if frames >= max_frames then
          break
        end
        s = session.state
        if num(s.max_power_bombs) > 0 then
          return s
        end
        local sx = num(s.samus_x or s.x)
        local sy = num(s.samus_y or s.y)
        if sx <= target_x and sy <= max_y + 5 then
          return s
        end
        if sy > pit_y or i16(s.velocity_y) > 80 then
          break
        end
        if not M.is_morph(s.pose) then
          break
        end
        if sy < elev_y or (sy <= max_y and step > math.floor(cycle_len / 2)) then
          hold(session, 1, {"LEFT"}, "safe_roll")
        elseif sy <= max_y + 8 then
          if step < 8 then
            hold(session, 1, {}, "safe_bomb_wait")
          else
            hold(session, 1, {"LEFT"}, "safe_roll")
          end
        else
          hold(session, 1, {}, "safe_band_wait")
        end
        frames = frames + 1
        frames_since_progress = frames_since_progress + 1
      end
      if num(session.state.samus_x or session.state.x) < last_progress_x - 2 then
        last_progress_x = num(session.state.samus_x or session.state.x)
        frames_since_progress = 0
      else
        hold(session, 6, {}, "safe_stall_pause")
        frames = frames + 6
        frames_since_progress = frames_since_progress + 6
      end
    end
  end
  return session.state
end

function M.morph_bomb_hole_climb(session, opts)
  opts = opts or {}
  local label = opts.label or "hole"
  local hole_x_lo = opts.hole_x_lo or 372
  local hole_x_hi = opts.hole_x_hi or 382
  local success_y = opts.success_y or 200
  local peak_y = opts.peak_y or 210
  local settle_y = opts.settle_y or 240
  local firm_y = opts.firm_y or 195
  local max_cycles = opts.max_cycles or 90
  local guard = opts.guard
  local min_y = num(session.state.samus_y or session.state.y)
  if opts.best_min_y then
    min_y = opts.best_min_y
  end
  M.ensure_morph(session)
  local climbed = false
  local cycle
  for cycle = 1, max_cycles do
    local state = session.state
    min_y = math.min(min_y, num(state.samus_y or state.y))
    if guard then
      guard(state)
    end
    local x = num(state.samus_x or state.x)
    if x < hole_x_lo then
      hold(session, 2, {"RIGHT"}, label .. "_hole_recenter")
    elseif x > hole_x_hi then
      hold(session, 2, {"LEFT"}, label .. "_hole_recenter")
    end
    M.lay_bomb(session, 2, label .. "_hole_bomb")
    local y = num(session.state.samus_y or session.state.y)
    local wait
    if y < 260 then
      wait = 22
    elseif y < 280 then
      wait = 30
    else
      wait = 50
    end
    local w
    for w = 1, wait do
      state = hold(session, 1, {}, label .. "_hole_bomb_wait")
      min_y = math.min(min_y, num(state.samus_y or state.y))
      if guard then
        guard(state)
      end
    end
    if num(session.state.samus_y or session.state.y) < success_y then
      climbed = true
      break
    end
    if min_y < peak_y and num(session.state.samus_y or session.state.y) < settle_y then
      M.lay_bomb(session, 2, label .. "_hole_top_bomb")
      local tw
      for tw = 1, 20 do
        state = hold(session, 1, {}, label .. "_hole_top_wait")
        min_y = math.min(min_y, num(state.samus_y or state.y))
        if guard then
          guard(state)
        end
        if num(state.samus_y or state.y) < firm_y then
          climbed = true
          break
        end
      end
      if climbed then
        break
      end
    end
  end
  if not climbed then
    timeout(label .. ": bomb-hole climb timed out: " .. brief(session.state)
      .. "; best_min_y=" .. tostring(min_y))
  end
  return min_y
end

function M.morph_upper_plant(session, opts)
  opts = opts or {}
  local label = opts.label or "plant"
  local plant_y = opts.plant_y or 190
  local max_bombs = opts.max_bombs or 8
  local wait_frames = opts.wait_frames or 22
  local settle_frames = opts.settle_frames or 10
  local fail_y = opts.fail_y or 230
  local guard = opts.guard
  local min_y = num(session.state.samus_y or session.state.y)
  M.ensure_morph(session)
  local b
  for b = 1, max_bombs do
    if num(session.state.samus_y or session.state.y) < plant_y then
      break
    end
    M.lay_bomb(session, 2, label .. "_upper_plant_bomb")
    local w
    for w = 1, wait_frames do
      local state = hold(session, 1, {}, label .. "_upper_plant_wait")
      min_y = math.min(min_y, num(state.samus_y or state.y))
      if guard then
        guard(state)
      end
    end
  end
  hold(session, settle_frames, {}, label .. "_upper_morph_settle")
  if num(session.state.samus_y or session.state.y) >= fail_y then
    timeout(label .. ": fell off upper after hole climb: " .. brief(session.state)
      .. "; best_min_y=" .. tostring(min_y))
  end
  return min_y
end

function M.morph_roll_to_window(session, opts)
  opts = opts or {}
  local label = opts.label or "window"
  local x_lo = opts.x_lo
  local x_hi = opts.x_hi
  local y_max = opts.y_max
  local max_frames = opts.max_frames or 500
  local sink_y = opts.sink_y or 210
  local fall_y = opts.fall_y or 300
  local boost_wait = opts.boost_wait or 18
  local source_room = opts.source_room
  local forbidden = opts.forbidden_rooms or {}
  local guard = opts.guard
  local i
  for i = 1, max_frames do
    local state = session.state
    local room = num(state.room_id)
    if forbidden[room] then
      timeout(label .. ": upper traverse crossed wrong door: " .. brief(state))
    end
    if source_room ~= nil and room ~= source_room then
      timeout(label .. ": upper traverse left source room: " .. brief(state))
    end
    if num(state.samus_y or state.y) > fall_y then
      timeout(label .. ": fell during upper traverse: " .. brief(state))
    end
    if guard then
      guard(state)
    end
    if num(state.samus_y or state.y) > sink_y then
      M.lay_bomb(session, 2, label .. "_traverse_boost")
      local w
      for w = 1, boost_wait do
        state = hold(session, 1, {}, label .. "_traverse_boost_wait")
        if guard then
          guard(state)
        end
      end
    end
    local x = num(state.samus_x or state.x)
    if x < x_lo then
      hold(session, 1, {"RIGHT"}, label .. "_window_recover")
    elseif x <= x_hi and num(state.samus_y or state.y) < y_max then
      return session.state
    else
      hold(session, 1, {"LEFT"}, label .. "_upper_roll")
    end
  end
  timeout(label .. ": x-window approach timed out: " .. brief(session.state))
end

M.hold = hold
M.wait_until = wait_until
M.brief = brief
M.MORPH_BALL_MASK = ram.MORPH_BALL_MASK or 0x0004

return M
