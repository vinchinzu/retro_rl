-- Named climb phases for Kraid return reverse hops (Kihunter / Zeela).
-- Port of skills/kraid_return.py plus door_exit + morph_bomb helpers used here.

local geo = require("red_tower.ctrl")

local M = {}

M.ZEELA_PHASE_BOTTOM_ROLL = "bottom_roll"
M.ZEELA_PHASE_MID_PLATFORM = "mid_platform"
M.ZEELA_PHASE_BELOW_LIP = "below_platform_lip"
M.ZEELA_PHASE_WALL_PLANT = "wall_plant"
M.ZEELA_PHASE_SHOTBLOCK_CLEAR = "shotblock_clear"
M.ZEELA_PHASE_WALL_REPLANT = "wall_replant"
M.ZEELA_PHASE_WALL_SPIN_CLIMB = "wall_spin_climb"
M.ZEELA_PHASE_SHOTBLOCK_CLIMB = "shotblock_wall_climb"
M.ZEELA_PHASE_WAREHOUSE_DOOR = "warehouse_door_exit"
M.EYE_PHASE_MID_ROOM = "eye_mid_room_approach"

-- ---------------------------------------------------------------------------
-- door_exit helpers
-- ---------------------------------------------------------------------------

function M.lip_stage(session, opts)
  opts = opts or {}
  local label = opts.label or "lip"
  if (opts.settle_frames or 0) > 0 then
    geo.hold(session, opts.settle_frames, {}, label .. "_approach_settle")
  end
  geo.hold(session, opts.backoff_frames or 8, {opts.backoff or "RIGHT"}, label .. "_lip_backoff")
  geo.hold(session, opts.face_frames or 8, {opts.face or "LEFT"}, label .. "_face")
  geo.hold(session, opts.release_frames or 6, {}, label .. "_face_release")
end

function M.beam_open_door(session, opts)
  opts = opts or {}
  local label = opts.label or "door"
  local shots = opts.shots or 6
  local shot_frames = opts.shot_frames or 4
  local fuse_frames = opts.fuse_frames or 14
  local shot_buttons = opts.shot_buttons or {"X"}
  for _ = 1, shots do
    geo.hold(session, shot_frames, shot_buttons, label .. "_door_shot")
    geo.hold(session, fuse_frames, {}, label .. "_door_fuse")
  end
end

function M.drain_door_transition(session, target_room, max_frames, reason)
  max_frames = max_frames or 80
  reason = reason or "transition"
  local state = session.state
  for _ = 1, max_frames do
    state = geo.hold(session, 1, {}, reason)
    if state.room_id == target_room and (state.door_transition or 0) == 0 then
      break
    end
  end
  return state
end

function M.period_exit_push(session, target_room, opts)
  opts = opts or {}
  local label = opts.label or "exit"
  local max_frames = opts.max_frames or 700
  local period = opts.period or 30
  local windows = opts.windows
  if not windows or #windows == 0 then
    error(label .. ": period_exit_push requires at least one window")
  end
  local drain = opts.transition_drain or 0
  local drain_reason = opts.transition_reason or (label .. "_transition")
  for index = 0, max_frames - 1 do
    local phase = index % period
    local buttons = windows[#windows][2]
    local suffix = windows[#windows][3]
    for w = 1, #windows do
      if phase < windows[w][1] then
        buttons = windows[w][2]
        suffix = windows[w][3]
        break
      end
    end
    local state = geo.hold(session, 1, buttons, label .. "_" .. suffix)
    if opts.guard then
      opts.guard(state)
    end
    if opts.on_state then
      opts.on_state(state)
    end
    if state.room_id == target_room then
      return state
    end
    if opts.on_wrong_room then
      opts.on_wrong_room(state)
    end
    if drain > 0 and state.door_transition and state.door_transition ~= 0 then
      state = M.drain_door_transition(session, target_room, drain, drain_reason)
      if state.room_id == target_room then
        return state
      end
    end
  end
  error(label .. ": exit timed out: " .. geo.fmt_state(session.state))
end

function M.jump_enter_exit(session, target_room, opts)
  opts = opts or {}
  local direction = opts.direction or "LEFT"
  return M.period_exit_push(session, target_room, {
    label = opts.label or "jump_enter",
    max_frames = opts.max_frames or 700,
    period = opts.period or 30,
    windows = {
      {opts.jump_end or 4, {direction, "A"}, "jump"},
      {opts.spin_end or 10, {direction, "A", "B"}, "jump_spin"},
      {opts.reshot_end or 14, {"X"}, "reshot"},
      {opts.period or 30, {direction, "B"}, "exit"},
    },
    transition_drain = opts.transition_drain or 80,
    transition_reason = (opts.label or "jump_enter") .. "_transition",
    guard = opts.guard,
    on_wrong_room = opts.on_wrong_room,
  })
end

-- ---------------------------------------------------------------------------
-- morph_bomb helpers
-- ---------------------------------------------------------------------------

function M.align_x(session, opts)
  opts = opts or {}
  local x_lo, x_hi = opts.x_lo, opts.x_hi
  local label = opts.label or "align"
  local max_frames = opts.max_frames or 50
  local settle_frames = opts.settle_frames or 0
  local reason = opts.reason or "align"
  for _ = 1, max_frames do
    local state = session.state
    if opts.guard then
      opts.guard(state)
    end
    if state.samus_x >= x_lo and state.samus_x <= x_hi then
      break
    end
    local dir = (state.samus_x < x_lo) and "RIGHT" or "LEFT"
    geo.hold(session, 1, {dir}, label .. "_" .. reason)
  end
  if settle_frames > 0 then
    geo.hold(session, settle_frames, {}, label .. "_" .. reason .. "_settle")
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
  geo.ensure_morph(session)
  local min_y = (opts.best_min_y and opts.best_min_y[1]) or session.state.samus_y
  local climbed = false
  local function guard(state)
    if state.samus_y < min_y then
      min_y = state.samus_y
    end
    if opts.guard then
      opts.guard(state)
    end
  end
  for _ = 1, max_cycles do
    local state = session.state
    guard(state)
    if state.samus_x < hole_x_lo then
      geo.hold(session, 2, {"RIGHT"}, label .. "_hole_recenter")
    elseif state.samus_x > hole_x_hi then
      geo.hold(session, 2, {"LEFT"}, label .. "_hole_recenter")
    end
    geo.hold(session, 2, {"X"}, label .. "_hole_bomb")
    local wait
    if state.samus_y < 260 then
      wait = 22
    elseif state.samus_y < 280 then
      wait = 30
    else
      wait = 50
    end
    for _w = 1, wait do
      state = geo.hold(session, 1, {}, label .. "_hole_bomb_wait")
      guard(state)
    end
    if session.state.samus_y < success_y then
      climbed = true
      break
    end
    if min_y < peak_y and session.state.samus_y < settle_y then
      geo.hold(session, 2, {"X"}, label .. "_hole_top_bomb")
      for _w = 1, 20 do
        state = geo.hold(session, 1, {}, label .. "_hole_top_wait")
        guard(state)
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
    error(label .. ": bomb-hole climb timed out: " .. geo.fmt_state(session.state)
      .. "; best_min_y=" .. tostring(min_y))
  end
  if opts.best_min_y then
    opts.best_min_y[1] = min_y
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
  geo.ensure_morph(session)
  local min_y = (opts.best_min_y and opts.best_min_y[1]) or session.state.samus_y
  for _ = 1, max_bombs do
    if session.state.samus_y < plant_y then
      break
    end
    geo.hold(session, 2, {"X"}, label .. "_upper_plant_bomb")
    for _w = 1, wait_frames do
      local state = geo.hold(session, 1, {}, label .. "_upper_plant_wait")
      if state.samus_y < min_y then
        min_y = state.samus_y
      end
      if opts.guard then
        opts.guard(state)
      end
    end
  end
  geo.hold(session, settle_frames, {}, label .. "_upper_morph_settle")
  if session.state.samus_y >= fail_y then
    error(label .. ": fell off upper after hole climb: " .. geo.fmt_state(session.state)
      .. "; best_min_y=" .. tostring(min_y))
  end
  if opts.best_min_y then
    opts.best_min_y[1] = min_y
  end
  return min_y
end

function M.morph_roll_to_window(session, opts)
  opts = opts or {}
  local label = opts.label or "roll"
  local x_lo, x_hi, y_max = opts.x_lo, opts.x_hi, opts.y_max
  local max_frames = opts.max_frames or 500
  local sink_y = opts.sink_y or 210
  local fall_y = opts.fall_y or 300
  local boost_wait = opts.boost_wait or 18
  local forbidden = opts.forbidden_rooms or {}
  for _ = 1, max_frames do
    local state = session.state
    if forbidden[state.room_id] then
      error(label .. ": upper traverse crossed wrong door: " .. geo.fmt_state(state))
    end
    if opts.source_room and state.room_id ~= opts.source_room then
      error(label .. ": upper traverse left source room: " .. geo.fmt_state(state))
    end
    if state.samus_y > fall_y then
      error(label .. ": fell during upper traverse: " .. geo.fmt_state(state))
    end
    if opts.guard then
      opts.guard(state)
    end
    if state.samus_y > sink_y then
      geo.hold(session, 2, {"X"}, label .. "_traverse_boost")
      for _w = 1, boost_wait do
        state = geo.hold(session, 1, {}, label .. "_traverse_boost_wait")
        if opts.guard then
          opts.guard(state)
        end
      end
    end
    if state.samus_x < x_lo then
      geo.hold(session, 1, {"RIGHT"}, label .. "_window_recover")
    elseif state.samus_x <= x_hi and state.samus_y < y_max then
      return session.state
    else
      geo.hold(session, 1, {"LEFT"}, label .. "_upper_roll")
    end
  end
  error(label .. ": x-window approach timed out: " .. geo.fmt_state(session.state))
end

-- ---------------------------------------------------------------------------
-- Kihunter phases
-- ---------------------------------------------------------------------------

function M.kihunter_guard_rooms(session, label, allow_zeela)
  local state = session.state
  if state.room_id == geo.ROOM_BABY_KRAID then
    error(label .. ": crossed wrong door into Baby Kraid: " .. geo.fmt_state(state))
  end
  if state.room_id == geo.ROOM_ZEELA then
    if allow_zeela then
      return state
    end
    error(label .. ": reached Zeela before true upper Kihunter land: " .. geo.fmt_state(state))
  end
  if state.room_id ~= geo.ROOM_WAREHOUSE_KIHUNTER then
    error(label .. ": left source room during climb: " .. geo.fmt_state(state))
  end
  return state
end

function M.kihunter_wall_plant(session, label)
  geo.hold(session, 10, {}, "kihunter_zeela_entry_release")
  for _ = 1, 220 do
    local state = M.kihunter_guard_rooms(session, label)
    if state.samus_x <= 358 then
      break
    end
    geo.hold(session, 1, {"LEFT", "B"}, "kihunter_zeela_wall_run")
  end
  for _ = 1, 40 do
    local state = M.kihunter_guard_rooms(session, label)
    if state.samus_x <= 120 then
      break
    end
    geo.hold(session, 1, {"LEFT"}, "kihunter_zeela_wall_plant")
  end
  geo.hold(session, 4, {}, "kihunter_zeela_wall_settle")
end

function M.kihunter_mid_ledge(session, label, best_min_y)
  geo.hold(session, 3, {"RIGHT"}, "kihunter_zeela_face_ledge")
  geo.hold(session, 2, {}, "kihunter_zeela_face_settle")
  geo.hold(session, 8, {"DOWN"}, "kihunter_zeela_crouch_load")
  local mid = false
  local has_hj = geo.has_hi_jump(session.state)
  for frame = 0, 109 do
    local state = session.state
    local names
    if has_hj and state.samus_y <= 300 then
      names = (state.samus_x < 367) and {"RIGHT", "A", "B"} or {}
    elseif frame < 30 then
      names = {"A"}
    elseif frame < 45 then
      names = {"A", "UP", "X"}
    elseif state.samus_y <= 300 then
      names = {"RIGHT", "A", "B"}
    else
      names = {"RIGHT", "B"}
    end
    state = geo.hold(session, 1, names, "kihunter_zeela_mid_ledge")
    if state.samus_y < best_min_y[1] then
      best_min_y[1] = state.samus_y
    end
    M.kihunter_guard_rooms(session, label)
    if state.samus_y <= 299 and (state.velocity_y or 0) == 0
        and state.samus_x >= 365 and frame > 40 then
      mid = true
      break
    end
  end
  geo.hold(session, 16, {}, "kihunter_zeela_mid_settle")
  if not mid and not (session.state.samus_y <= 305 and session.state.samus_x >= 360) then
    error(label .. ": mid ledge missed: " .. geo.fmt_state(session.state)
      .. "; best_min_y=" .. tostring(best_min_y[1]))
  end
end

function M.kihunter_bomb_hole(session, label, best_min_y)
  M.align_x(session, {
    x_lo = 374, x_hi = 380, label = "kihunter_zeela_hole",
    max_frames = 50, settle_frames = 6,
    guard = function() M.kihunter_guard_rooms(session, label) end,
    reason = "align",
  })
  M.morph_bomb_hole_climb(session, {
    label = "kihunter_zeela",
    hole_x_lo = 372, hole_x_hi = 382,
    success_y = 200, peak_y = 210, settle_y = 240, firm_y = 195,
    max_cycles = 90,
    guard = function() M.kihunter_guard_rooms(session, label) end,
    best_min_y = best_min_y,
  })
  M.morph_upper_plant(session, {
    label = "kihunter_zeela",
    plant_y = 190, max_bombs = 8, wait_frames = 22, settle_frames = 10, fail_y = 230,
    guard = function() M.kihunter_guard_rooms(session, label) end,
    best_min_y = best_min_y,
  })
end

function M.kihunter_upper_to_zeela_window(session, label)
  M.morph_roll_to_window(session, {
    label = "kihunter_zeela",
    x_lo = 96, x_hi = 160, y_max = 230, max_frames = 500,
    sink_y = 210, fall_y = 300, boost_wait = 18,
    source_room = geo.ROOM_WAREHOUSE_KIHUNTER,
    forbidden_rooms = {[geo.ROOM_BABY_KRAID] = true},
  })
  geo.unmorph(session)
  geo.select_weapon(session, 0)
  geo.hold(session, 10, {}, "kihunter_zeela_window_stand")
  local state = session.state
  if not (state.samus_x >= 90 and state.samus_x <= 170) or state.samus_y >= 250 then
    error(label .. ": invalid Zeela door window: " .. geo.fmt_state(state))
  end
  geo.hold(session, 8, {"LEFT"}, "kihunter_zeela_door_face")
  geo.hold(session, 6, {}, "kihunter_zeela_door_release")
  M.beam_open_door(session, {
    label = "kihunter_zeela", shots = 6, shot_frames = 4, fuse_frames = 14,
    shot_buttons = {"DOWN", "X"},
  })
  M.period_exit_push(session, geo.ROOM_ZEELA, {
    label = "kihunter_zeela",
    max_frames = 520,
    period = 30,
    windows = {
      {8, {"DOWN", "A"}, "drop"},
      {14, {"DOWN", "A", "B"}, "drop_spin"},
      {18, {"DOWN", "X"}, "reshot"},
      {30, {"DOWN"}, "drop"},
    },
    on_wrong_room = function(st)
      if st.room_id == geo.ROOM_BABY_KRAID then
        error(label .. ": blue down-door entered Baby Kraid: " .. geo.fmt_state(session.state))
      end
    end,
  })
end

-- ---------------------------------------------------------------------------
-- Zeela phases
-- ---------------------------------------------------------------------------

local function zeela_guard(session, label, state, phase)
  if state.door_transition and state.door_transition ~= 0 and state.samus_y > 250 then
    error(label .. ": floor door transition during " .. phase .. ": " .. geo.fmt_state(state))
  end
  if state.room_id ~= geo.ROOM_ZEELA then
    error(label .. ": left Zeela during " .. phase .. ": " .. geo.fmt_state(state))
  end
end

function M.zeela_bottom_roll(session, label)
  geo.ensure_morph(session)
  local rolled = false
  for _ = 1, 900 do
    local state = geo.hold(session, 1, {"LEFT"}, "zeela_warehouse_bottom_roll")
    zeela_guard(session, label, state, M.ZEELA_PHASE_BOTTOM_ROLL)
    if state.samus_x <= 160 and state.samus_y >= 350 then
      rolled = true
      break
    end
  end
  if not rolled then
    error(label .. ": " .. M.ZEELA_PHASE_BOTTOM_ROLL .. " stalled: " .. geo.fmt_state(session.state))
  end
  geo.unmorph(session)
  M.align_x(session, {
    x_lo = 110, x_hi = 140, label = "zeela_warehouse_second",
    max_frames = 160, settle_frames = 8,
    guard = function(s) zeela_guard(session, label, s, M.ZEELA_PHASE_BOTTOM_ROLL) end,
    reason = "align",
  })
end

function M.zeela_mid_platform(session, label)
  geo.select_weapon(session, 0)
  local has_hj = geo.has_hi_jump(session.state)
  local mid = false
  for frame = 0, 699 do
    local state = session.state
    zeela_guard(session, label, state, M.ZEELA_PHASE_MID_PLATFORM)
    local mid_x_min = has_hj and 96 or 90
    if state.samus_y >= 300 and state.samus_y <= 350
        and (state.velocity_y or 0) == 0
        and state.samus_x >= mid_x_min and frame > 50 then
      mid = true
      break
    end
    local cadence = frame % 28
    local names
    if cadence < 5 then
      names = {"UP", "X"}
    elseif state.samus_y <= 360 then
      names = (cadence >= 16) and {"RIGHT", "A", "B"} or {"RIGHT", "A"}
    elseif cadence < 14 then
      names = {"A"}
    elseif state.samus_x <= 70 then
      names = {"RIGHT", "A"}
    else
      names = {"LEFT", "A"}
    end
    geo.hold(session, 1, names, "zeela_warehouse_second_reverse_shot")
  end
  geo.hold(session, 12, {}, "zeela_warehouse_mid_settle")
  if not mid and not (session.state.samus_y >= 300 and session.state.samus_y <= 355
      and session.state.samus_x >= 85) then
    error(label .. ": " .. M.ZEELA_PHASE_MID_PLATFORM .. " missed: " .. geo.fmt_state(session.state))
  end
end

function M.zeela_below_platform_lip(session, label)
  for _ = 1, 80 do
    local state = session.state
    zeela_guard(session, label, state, M.ZEELA_PHASE_BELOW_LIP)
    if state.samus_y > 360 then
      error(label .. ": fell off mid platform: " .. geo.fmt_state(state))
    end
    if state.samus_x >= 104 then
      break
    end
    geo.hold(session, 1, {"RIGHT"}, "zeela_warehouse_mid_right_edge")
  end
  geo.hold(session, 8, {}, "zeela_warehouse_mid_edge_settle")
  geo.unmorph(session)
  geo.select_weapon(session, 0)
  geo.hold(session, 10, {"DOWN"}, "zeela_warehouse_first_crouch_load")
  geo.hold(session, 2, {}, "zeela_warehouse_first_crouch_release")
  local lip = false
  for frame = 0, 599 do
    local state = session.state
    zeela_guard(session, label, state, M.ZEELA_PHASE_BELOW_LIP)
    if state.samus_y <= 240 and (state.velocity_y or 0) == 0
        and state.samus_x <= 80 and frame > 40 then
      lip = true
      break
    end
    local names
    if frame < 18 then
      names = {"A"}
    elseif frame < 30 then
      names = {"A", "UP", "X"}
    elseif state.samus_y < 280 then
      names = {"LEFT", "A", "B"}
    else
      local cadence = frame % 24
      if cadence < 6 then
        names = {"UP", "X"}
      elseif cadence < 16 then
        names = {"A"}
      else
        names = {}
      end
    end
    geo.hold(session, 1, names, "zeela_warehouse_first_crouch_climb")
  end
  geo.hold(session, 15, {}, "zeela_warehouse_lip_settle")
  if not lip and not (session.state.samus_y <= 250 and session.state.samus_x <= 90) then
    error(label .. ": " .. M.ZEELA_PHASE_BELOW_LIP .. " missed: " .. geo.fmt_state(session.state))
  end
end

function M.zeela_wall_plant(session, label)
  geo.unmorph(session)
  for frame = 0, 49 do
    local state = session.state
    zeela_guard(session, label, state, M.ZEELA_PHASE_WALL_PLANT)
    if state.samus_x <= 45 and (state.velocity_y or 0) == 0
        and state.samus_y >= 210 and state.samus_y <= 230 and frame > 10 then
      break
    end
    local names
    if frame < 8 then
      names = {"A"}
    elseif frame < 28 then
      names = {"LEFT", "A"}
    else
      names = {"LEFT"}
    end
    geo.hold(session, 1, names, "zeela_warehouse_lip_hop_left")
  end
  geo.hold(session, 20, {}, "zeela_warehouse_bp_settle")
  if not (session.state.samus_y <= 230 and session.state.samus_x <= 55) then
    error(label .. ": " .. M.ZEELA_PHASE_WALL_PLANT .. " missed: " .. geo.fmt_state(session.state))
  end
end

function M.zeela_shotblock_clear(session, label)
  geo.unmorph(session)
  geo.select_weapon(session, 0)
  for _ = 1, 20 do
    if session.state.pose == 1 or session.state.pose == 2 then
      break
    end
    geo.hold(session, 1, {"UP"}, "zeela_warehouse_clear_stand")
  end
  geo.hold(session, 8, {}, "zeela_warehouse_clear_stand_settle")
  if session.state.samus_y > 250 then
    error(label .. ": fell off lip before clear: " .. geo.fmt_state(session.state))
  end
  for _ = 1, 40 do
    geo.hold(session, 2, {"UP", "X"}, "zeela_warehouse_shotblock_clear")
    geo.hold(session, 4, {}, "zeela_warehouse_shotblock_fuse")
  end
  for frame = 0, 59 do
    local state = session.state
    zeela_guard(session, label, state, M.ZEELA_PHASE_SHOTBLOCK_CLEAR)
    local cadence = frame % 8
    local names
    if cadence < 3 then
      names = {"UP", "X"}
    elseif cadence < 6 then
      names = {"A"}
    else
      names = {}
    end
    geo.hold(session, 1, names, "zeela_warehouse_clear_jump")
  end
end

function M.zeela_wall_replant(session, label)
  for _ = 1, 30 do
    local state = session.state
    zeela_guard(session, label, state, M.ZEELA_PHASE_WALL_REPLANT)
    if state.samus_x <= 35 then
      break
    end
    if state.samus_y > 250 then
      break
    end
    geo.hold(session, 1, {"LEFT", "B"}, "zeela_warehouse_wall_plant")
  end
  geo.hold(session, 4, {}, "zeela_warehouse_wall_plant_settle")
end

function M.zeela_wall_spin_climb(session, label)
  local top_poses = geo.set(1, 2, 39, 40, 137, 138)
  for frame = 0, 899 do
    local state = session.state
    zeela_guard(session, label, state, M.ZEELA_PHASE_WALL_SPIN_CLIMB)
    if state.samus_y <= 150 and (state.velocity_y or 0) == 0
        and top_poses[state.pose] and frame > 20 then
      geo.hold(session, 8, {}, "zeela_warehouse_top_confirm")
      if session.state.samus_y <= 155 and (session.state.velocity_y or 0) == 0
          and session.state.room_id == geo.ROOM_ZEELA then
        return {reached_top = true}
      end
    end
    local cadence = frame % 14
    local names
    if cadence < 7 then
      names = {"LEFT", "A", "B"}
    elseif cadence < 10 then
      names = {"RIGHT", "A"}
    elseif cadence < 12 then
      names = {"LEFT", "A"}
    else
      names = {"LEFT"}
    end
    state = geo.hold(session, 1, names, "zeela_warehouse_wall_climb")
    if state.room_id == geo.ROOM_WAREHOUSE then
      if state.samus_y > 250 then
        error(label .. ": floor door transition during wall climb: " .. geo.fmt_state(state))
      end
      return {
        warehouse_settle = geo.wait_ordinary_room(session, geo.ROOM_WAREHOUSE, {
          settle_frames = 320,
          label = label,
        }),
      }
    end
    if session.state.samus_y > 280 then
      break
    end
  end
  return {}
end

function M.zeela_shotblock_wall_climb(session, label)
  for _ = 1, 3 do
    M.zeela_shotblock_clear(session, label)
    M.zeela_wall_replant(session, label)
    local spin = M.zeela_wall_spin_climb(session, label)
    if spin.warehouse_settle then
      return spin.warehouse_settle
    end
    if spin.reached_top then
      return nil
    end
    if not (session.state.samus_y <= 230 and session.state.samus_x <= 55) then
      break
    end
  end
  error(label .. ": " .. M.ZEELA_PHASE_SHOTBLOCK_CLIMB .. " top band missed: "
    .. geo.fmt_state(session.state))
end

function M.eye_mid_room_approach(session, label)
  local reached = false
  for index = 0, 499 do
    local state = session.state
    if state.samus_x <= 140 then
      reached = true
      break
    end
    local phase = index % 30
    if phase < 10 then
      geo.hold(session, 1, {"LEFT", "A"}, "eye_to_baby_hop")
    elseif phase < 18 then
      geo.hold(session, 1, {"LEFT", "A", "B"}, "eye_to_baby_spin")
    elseif phase < 22 then
      geo.hold(session, 1, {"X"}, "eye_to_baby_clear_shot")
    else
      geo.hold(session, 1, {"LEFT", "B"}, "eye_to_baby_run")
    end
  end
  if not reached then
    error(label .. ": " .. M.EYE_PHASE_MID_ROOM .. " timed out: " .. geo.fmt_state(session.state))
  end
end

function M.zeela_warehouse_door_exit(session, label)
  geo.unmorph(session)
  geo.select_weapon(session, 0)
  for _ = 1, 15 do
    if session.state.pose == 1 or session.state.pose == 2 then
      break
    end
    geo.hold(session, 1, {"UP"}, "zeela_warehouse_door_stand")
  end
  geo.hold(session, 10, {}, "zeela_warehouse_door_stand_settle")
  if session.state.samus_y > 170 then
    error(label .. ": fell before Warehouse door: " .. geo.fmt_state(session.state))
  end
  geo.hold(session, 6, {"LEFT"}, "zeela_warehouse_door_face")
  geo.hold(session, 4, {}, "zeela_warehouse_door_face_release")
  M.beam_open_door(session, {
    label = "zeela_warehouse", shots = 10, shot_frames = 3, fuse_frames = 12,
    shot_buttons = {"LEFT", "X"},
  })
  local ok, err = pcall(function()
    M.period_exit_push(session, geo.ROOM_WAREHOUSE, {
      label = "zeela_warehouse",
      max_frames = 400,
      period = 20,
      windows = {
        {10, {"LEFT", "A"}, "exit"},
        {14, {"LEFT", "A", "B"}, "exit_spin"},
        {20, {"LEFT"}, "exit"},
      },
      guard = function(state)
        if state.door_transition and state.door_transition ~= 0 and state.samus_y > 250 then
          error(label .. ": floor door transition at Warehouse exit: " .. geo.fmt_state(state))
        end
        if state.room_id ~= geo.ROOM_ZEELA and state.room_id ~= geo.ROOM_WAREHOUSE then
          error(label .. ": unexpected exit room: " .. geo.fmt_state(state))
        end
        if state.room_id == geo.ROOM_ZEELA and state.samus_y > 200 then
          error(label .. ": fell during Warehouse exit: " .. geo.fmt_state(state))
        end
      end,
    })
  end)
  if not ok then
    error(label .. ": " .. M.ZEELA_PHASE_WAREHOUSE_DOOR .. " timed out: "
      .. geo.fmt_state(session.state) .. " (" .. tostring(err) .. ")")
  end
end

return M
