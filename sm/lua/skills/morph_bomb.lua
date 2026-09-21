-- Morph bomb-jump and morph-roll micro-skills.

local controller = require("skills.controller")

local morph_bomb = {}

function morph_bomb.align_x(session, opts)
  opts = opts or {}
  local x_lo, x_hi = opts.x_lo, opts.x_hi
  local label = opts.label or "align"
  local max_frames = opts.max_frames or 50
  local settle_frames = opts.settle_frames or 0
  local guard = opts.guard
  local reason = opts.reason or "align"
  for _ = 1, max_frames do
    local state = session.state
    if guard then
      guard(state)
    end
    if x_lo <= state.samus_x and state.samus_x <= x_hi then
      break
    end
    local direction = state.samus_x < x_lo and "RIGHT" or "LEFT"
    session:hold(1, { direction }, label .. "_" .. reason)
  end
  if settle_frames > 0 then
    session:hold(settle_frames, {}, label .. "_" .. reason .. "_settle")
  end
  return session.state
end

function morph_bomb.morph_bomb_hole_climb(session, opts)
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
  local best_min_y = opts.best_min_y
  controller.ensure_morph(session)
  local min_y = (best_min_y and best_min_y[1]) or session.state.samus_y
  local climbed = false

  local function do_guard(state)
    if state.samus_y < min_y then
      min_y = state.samus_y
    end
    if guard then
      guard(state)
    end
  end

  for _ = 1, max_cycles do
    local state = session.state
    do_guard(state)
    if state.samus_x < hole_x_lo then
      session:hold(2, { "RIGHT" }, label .. "_hole_recenter")
    elseif state.samus_x > hole_x_hi then
      session:hold(2, { "LEFT" }, label .. "_hole_recenter")
    end
    session:hold(2, { "X" }, label .. "_hole_bomb")
    local wait
    if state.samus_y < 260 then
      wait = 22
    elseif state.samus_y < 280 then
      wait = 30
    else
      wait = 50
    end
    for _w = 1, wait do
      state = session:hold(1, {}, label .. "_hole_bomb_wait")
      do_guard(state)
    end
    if session.state.samus_y < success_y then
      climbed = true
      break
    end
    if min_y < peak_y and session.state.samus_y < settle_y then
      session:hold(2, { "X" }, label .. "_hole_top_bomb")
      for _w = 1, 20 do
        state = session:hold(1, {}, label .. "_hole_top_wait")
        do_guard(state)
        if state.samus_y < firm_y then
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
    error(string.format(
      "TimeoutError: %s: bomb-hole climb timed out: %s; best_min_y=%s",
      label,
      tostring(session.state),
      tostring(min_y)
    ), 2)
  end
  if best_min_y then
    best_min_y[1] = min_y
  end
  return min_y
end

function morph_bomb.morph_upper_plant(session, opts)
  opts = opts or {}
  local label = opts.label or "upper"
  local plant_y = opts.plant_y or 190
  local max_bombs = opts.max_bombs or 8
  local wait_frames = opts.wait_frames or 22
  local settle_frames = opts.settle_frames or 10
  local fail_y = opts.fail_y or 230
  local guard = opts.guard
  local best_min_y = opts.best_min_y
  controller.ensure_morph(session)
  local min_y = (best_min_y and best_min_y[1]) or session.state.samus_y
  for _ = 1, max_bombs do
    if session.state.samus_y < plant_y then
      break
    end
    session:hold(2, { "X" }, label .. "_upper_plant_bomb")
    for _w = 1, wait_frames do
      local state = session:hold(1, {}, label .. "_upper_plant_wait")
      if state.samus_y < min_y then
        min_y = state.samus_y
      end
      if guard then
        guard(state)
      end
    end
  end
  session:hold(settle_frames, {}, label .. "_upper_morph_settle")
  if session.state.samus_y >= fail_y then
    error(string.format(
      "TimeoutError: %s: fell off upper after hole climb: %s; best_min_y=%s",
      label,
      tostring(session.state),
      tostring(min_y)
    ), 2)
  end
  if best_min_y then
    best_min_y[1] = min_y
  end
  return min_y
end

function morph_bomb.morph_roll_to_window(session, opts)
  opts = opts or {}
  local label = opts.label or "roll"
  local x_lo, x_hi = opts.x_lo, opts.x_hi
  local y_max = opts.y_max
  local max_frames = opts.max_frames or 500
  local sink_y = opts.sink_y or 210
  local fall_y = opts.fall_y or 300
  local boost_wait = opts.boost_wait or 18
  local source_room = opts.source_room
  local forbidden_rooms = opts.forbidden_rooms or {}
  local guard = opts.guard
  for _ = 1, max_frames do
    local state = session.state
    if forbidden_rooms[state.room_id] then
      error("TimeoutError: " .. label .. ": upper traverse crossed wrong door: " .. tostring(state), 2)
    end
    if source_room ~= nil and state.room_id ~= source_room then
      error("TimeoutError: " .. label .. ": upper traverse left source room: " .. tostring(state), 2)
    end
    if state.samus_y > fall_y then
      error("TimeoutError: " .. label .. ": fell during upper traverse: " .. tostring(state), 2)
    end
    if guard then
      guard(state)
    end
    if state.samus_y > sink_y then
      session:hold(2, { "X" }, label .. "_traverse_boost")
      for _w = 1, boost_wait do
        state = session:hold(1, {}, label .. "_traverse_boost_wait")
        if guard then
          guard(state)
        end
      end
    end
    if state.samus_x < x_lo then
      session:hold(1, { "RIGHT" }, label .. "_window_recover")
    elseif state.samus_x <= x_hi and state.samus_y < y_max then
      return session.state
    else
      session:hold(1, { "LEFT" }, label .. "_upper_roll")
    end
  end
  error("TimeoutError: " .. label .. ": x-window approach timed out: " .. tostring(session.state), 2)
end

return morph_bomb
