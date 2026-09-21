-- Runway / fire-seat micro-skills and full recipes.

local takeoff = require("takeoff")
local geometry = require("skills.geometry")
local walljump = require("skills.walljump")
local ram = require("ram")

local runway = {}

local _pol
local function default_policy()
  if not _pol then
    _pol = require("skills.policies.bubble_to_bat")
  end
  return _pol
end

function runway.stationary_missile_clear(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local face_n = opts.face_frames or pol.SAVE_STATIONARY_FACE
  local shoot_n = opts.shoot_frames or pol.SAVE_STATIONARY_X
  local angle_l = opts.angle_l
  if angle_l == nil then
    angle_l = true
  end
  local pose = tonumber(session.state.pose)
  if not geometry.POSE_STAND_LEFT[pose] and tonumber(session.state.samus_x) >= 35 then
    local n = face_n
    if n > 4 then
      n = 4
    end
    for _ = 1, n do
      session:hold(1, { "LEFT" }, label .. "_stat_x_face")
    end
    for _ = 1, 3 do
      session:hold(1, {}, label .. "_stat_x_face_settle")
    end
  end
  for _ = 1, shoot_n do
    local state
    if angle_l then
      state = session:hold(1, { "X", "L" }, label .. "_stat_x_shoot")
    else
      state = session:hold(1, { "X" }, label .. "_stat_x_shoot")
    end
    geometry.track_state(session, track, state, pol)
    if state.room_id ~= pol.ROOM_ID then
      return
    end
    if tonumber(state.samus_x) < 24 then
      session:hold(1, { "RIGHT" }, label .. "_stat_x_abort")
      return
    end
  end
end

function runway.walk_brake_to_x(session, track, target_x, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local max_frames = opts.max_frames or 80
  local band = opts.band or 2
  for _ = 1, max_frames do
    local state = session.state
    if state.room_id ~= pol.ROOM_ID then
      return false
    end
    local x = tonumber(state.samus_x) or 0
    if geometry.POSE_KNOCKBACK[tonumber(state.pose)] then
      session:hold(1, {}, label .. "_brake_kb")
    else
      local dx = x - target_x
      if dx < 0 then
        dx = -dx
      end
      if dx <= band and geometry.is_true_ground(state, pol.TRUE_GROUND) then
        session:hold(1, { x >= target_x and "RIGHT" or "LEFT" }, label .. "_brake_stop")
        for _s = 1, 10 do
          session:hold(1, {}, label .. "_brake_settle")
        end
        local d2 = (tonumber(session.state.samus_x) or 0) - target_x
        if d2 < 0 then
          d2 = -d2
        end
        return d2 <= band + 2
      end
      if x > target_x then
        session:hold(1, { "LEFT" }, label .. "_brake_l")
        session:hold(1, { "RIGHT" }, label .. "_brake_r")
      else
        session:hold(1, { "RIGHT" }, label .. "_brake_r")
        session:hold(1, { "LEFT" }, label .. "_brake_l")
      end
      session:hold(1, {}, label .. "_brake_w")
    end
  end
  local d = (tonumber(session.state.samus_x) or 0) - target_x
  if d < 0 then
    d = -d
  end
  return d <= band + 2
end

function runway.seat_max_left_fire(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local human_lo, human_hi = pol.SAVE_HUMAN_SEAT_X[1], pol.SAVE_HUMAN_SEAT_X[2]
  local aim = opts.target_x or 27
  if aim < human_lo then
    aim = human_lo
  elseif aim > human_hi then
    aim = human_hi
  end
  local attempts = opts.attempts or 3
  if session.state.room_id ~= pol.ROOM_ID then
    return false
  end
  local y = tonumber(session.state.samus_y) or 0
  local x = tonumber(session.state.samus_x) or 0
  local on_runway = geometry.on_save_runway(session.state, pol)
    or (
      pol.SAVE_RUNWAY_Y[1] <= y
      and y <= pol.SAVE_RUNWAY_Y[2]
      and pol.SAVE_RUNWAY_X[1] - 5 <= x
      and x <= pol.SAVE_RUNWAY_X[2]
    )
  if not on_runway then
    return false
  end
  if attempts < 1 then
    attempts = 1
  end
  for attempt = 0, attempts - 1 do
    if session.state.room_id ~= pol.ROOM_ID then
      return false
    end
    local x0 = tonumber(session.state.samus_x) or 0
    if x0 < human_lo - 2 then
      session:hold(1, { "RIGHT", "B" }, label .. "_seat_door")
      return false
    end
    runway.stationary_missile_clear(session, track, {
      policy = pol,
      face_frames = pol.SAVE_STATIONARY_FACE + attempt * 2,
      shoot_frames = pol.SAVE_STATIONARY_X + attempt * 12,
      angle_l = true,
    })
    if session.state.room_id ~= pol.ROOM_ID then
      return false
    end
    local stalled = 0
    local last_x = tonumber(session.state.samus_x) or 0
    for step = 0, 69 do
      local state = session.state
      if state.room_id ~= pol.ROOM_ID then
        return false
      end
      x = tonumber(state.samus_x) or 0
      if geometry.POSE_KNOCKBACK[tonumber(state.pose)] then
        session:hold(1, { "RIGHT", "B", "A" }, label .. "_seat_kb")
      elseif x < human_lo then
        session:hold(1, { "RIGHT" }, label .. "_seat_door")
        session:hold(1, {}, label .. "_seat_door_settle")
        break
      elseif human_lo <= x and x <= human_hi and geometry.is_true_ground(state, pol.TRUE_GROUND) then
        break
      else
        if x > aim then
          session:hold(1, { "LEFT" }, label .. "_seat_walk_l")
          if step % 2 == 1 and tonumber(session.state.samus_x) > human_hi then
            session:hold(1, { "RIGHT" }, label .. "_seat_brake")
          end
        elseif x < aim then
          session:hold(1, { "RIGHT" }, label .. "_seat_walk_r")
        end
        local nx = tonumber(session.state.samus_x) or 0
        local dx = nx - last_x
        if dx < 0 then
          dx = -dx
        end
        if dx <= 0 then
          stalled = stalled + 1
        else
          stalled = 0
          last_x = nx
        end
        if stalled >= 12 then
          break
        end
      end
    end
    runway.walk_brake_to_x(session, track, aim, { policy = pol, max_frames = 40, band = 1 })
    local pose = tonumber(session.state.pose)
    if geometry.POSE_STAND_RIGHT[pose] then
      session:hold(1, { "LEFT" }, label .. "_seat_face_l")
      for _ = 1, 6 do
        session:hold(1, {}, label .. "_seat_face_settle")
      end
    elseif not geometry.POSE_STAND_LEFT[pose] and not geometry.POSE_KNOCKBACK[pose] then
      for _ = 1, 4 do
        session:hold(1, {}, label .. "_seat_settle")
      end
    end
    x = tonumber(session.state.samus_x) or 0
    local st = session.state
    if st.room_id == pol.ROOM_ID
        and human_lo <= x and x <= human_hi
        and geometry.is_true_ground(st, pol.TRUE_GROUND)
        and not geometry.POSE_KNOCKBACK[tonumber(st.pose)] then
      return true
    end
  end
  x = tonumber(session.state.samus_x) or 0
  local st = session.state
  return st.room_id == pol.ROOM_ID
    and human_lo <= x and x <= human_hi
    and geometry.is_true_ground(st, pol.TRUE_GROUND)
    and not geometry.POSE_KNOCKBACK[tonumber(st.pose)]
end

function runway.prepare_fire_run(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local y_clear = opts.y_clear
  if y_clear == nil then
    y_clear = true
  end
  local y_frames = opts.y_frames or 8
  local pose = tonumber(session.state.pose)
  if geometry.POSE_KNOCKBACK[pose] then
    for _ = 1, 8 do
      session:hold(1, {}, label .. "_fire_kb_settle")
      if not geometry.POSE_KNOCKBACK[tonumber(session.state.pose)] then
        break
      end
    end
  end
  local x = tonumber(session.state.samus_x) or 0
  local human_lo, human_hi = pol.SAVE_HUMAN_SEAT_X[1], pol.SAVE_HUMAN_SEAT_X[2]
  pose = tonumber(session.state.pose)
  if geometry.POSE_STAND_LEFT[pose] and not (human_lo <= x and x <= human_hi) and x > human_hi then
    session:hold(1, { "RIGHT" }, label .. "_fire_face_tap")
    session:hold(1, {}, label .. "_fire_face_settle")
  end
  if y_clear then
    for _ = 1, y_frames do
      local state = session:hold(1, { "Y" }, label .. "_save_clear")
      geometry.track_state(session, track, state, pol)
      if state.room_id ~= pol.ROOM_ID then
        return
      end
    end
  end
  if opts.crouch then
    local n = opts.crouch_frames or pol.SAVE_CROUCH_FRAMES
    for _ = 1, n do
      session:hold(1, { "DOWN" }, label .. "_fire_crouch")
    end
  end
end

function runway.runway_dash(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local n = opts.frames or pol.SAVE_RUN_FRAMES
  local pump = opts.arm_pump
  if pump == nil then
    pump = pol.SAVE_ARM_PUMP
  end
  local period = opts.arm_period or pol.SAVE_ARM_PUMP_PERIOD
  if period < 1 then
    period = 1
  end
  local direction = opts.direction or "RIGHT"
  for i = 0, n - 1 do
    local state
    if pump then
      local ang = takeoff.shoulder_pump_button(i, period)
      state = session:hold(1, { direction, "B", ang }, label .. "_run_ap")
    else
      state = session:hold(1, { direction, "B" }, label .. "_run")
    end
    geometry.track_state(session, track, state, pol)
    if state.room_id ~= pol.ROOM_ID then
      return
    end
  end
end

function runway.spin_glide(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local n = opts.frames or pol.SAVE_SPIN_FRAMES
  local height_box = opts.height_box
  for _ = 1, n do
    local state = session:hold(1, { "RIGHT", "B", "A" }, label .. "_spin_glide")
    if walljump._track_upd(session, track, state, pol, height_box) then
      return track.top_reached and true or false
    end
  end
  return track.top_reached and true or false
end

function runway.read_enemy_slot(session, slot)
  local base = 0x0F78 + tonumber(slot) * 0x40
  return {
    slot = tonumber(slot),
    enemy_id = ram.u16(base),
    x = ram.u16(base + 0x02),
    y = ram.u16(base + 0x06),
    hp = ram.u16(base + 0x14),
  }
end

runway.EnemySnap = runway.read_enemy_slot
runway.BubbleEnemySnap = runway.read_enemy_slot

local function in_box(x, y, box)
  return box[1] <= x and x <= box[2] and box[3] <= y and y <= box[4]
end

function runway.fire_phase_geometry(e4_x, e4_y, e6_x, e6_y, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  if in_box(e4_x, e4_y, pol.FIRE_PHASE_A_E4) and in_box(e6_x, e6_y, pol.FIRE_PHASE_A_E6) then
    return true
  end
  if in_box(e4_x, e4_y, pol.FIRE_PHASE_B_E4) and in_box(e6_x, e6_y, pol.FIRE_PHASE_B_E6) then
    return true
  end
  return false
end

function runway.fire_phase_clear(session, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local e4 = runway.read_enemy_slot(session, 4)
  local e6 = runway.read_enemy_slot(session, 6)
  if not e4 or not e6 then
    return false
  end
  return runway.fire_phase_geometry(e4.x, e4.y, e6.x, e6.y, { policy = pol })
end

function runway.wait_fire_phase(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local budget = opts.max_frames or pol.FIRE_PHASE_MAX_WAIT
  local human_lo, human_hi = pol.SAVE_HUMAN_SEAT_X[1], pol.SAVE_HUMAN_SEAT_X[2]
  if budget <= 0 then
    return runway.fire_phase_clear(session, { policy = pol })
  end
  if runway.fire_phase_clear(session, { policy = pol }) then
    return true
  end
  for _ = 1, budget do
    local state = session.state
    if state.room_id ~= pol.ROOM_ID then
      return false
    end
    local x = tonumber(state.samus_x) or 0
    local y = tonumber(state.samus_y) or 0
    if not (
      human_lo - 2 <= x and x <= human_hi + 4
      and pol.SAVE_RUNWAY_Y[1] <= y and y <= pol.SAVE_RUNWAY_Y[2]
    ) then
      return false
    end
    if geometry.POSE_KNOCKBACK[tonumber(state.pose)] then
      session:hold(1, {}, label .. "_phase_kb")
    else
      session:hold(1, {}, label .. "_phase_wait")
      geometry.track_state(session, track, session.state, pol)
      if runway.fire_phase_clear(session, { policy = pol }) then
        return true
      end
    end
  end
  return runway.fire_phase_clear(session, { policy = pol })
end

function runway.save_runway_fire_recipe(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local height_box = { false }
  local y_clear = opts.y_clear
  if y_clear == nil then
    y_clear = true
  end
  local phase_wait = opts.phase_wait
  if phase_wait == nil then
    phase_wait = true
  end
  if phase_wait then
    runway.wait_fire_phase(session, track, { policy = pol, max_frames = opts.phase_max_frames })
  end
  runway.prepare_fire_run(session, track, {
    policy = pol,
    y_clear = y_clear,
    crouch = opts.crouch,
  })
  runway.runway_dash(session, track, {
    policy = pol,
    frames = opts.run_frames,
    arm_pump = opts.arm_pump,
  })
  if runway.spin_glide(session, track, {
    policy = pol,
    frames = opts.spin_frames,
    height_box = height_box,
  }) then
    return true
  end
  if session.state.room_id ~= pol.ROOM_ID then
    return track.top_reached and true or false
  end
  local n = opts.wj_count or 2
  if n < 1 then
    n = 1
  end
  local jumps = { pol.R15_DOUBLE[1], pol.R15_DOUBLE[2] }
  while #jumps < n do
    jumps[#jumps + 1] = pol.R15_WJ2
  end
  local trimmed = {}
  for i = 1, n do
    trimmed[i] = jumps[i]
  end
  return walljump.consecutive_walljumps(session, track, trimmed, {
    policy = pol,
    pre_approach = true,
    extend_spin_ready = opts.extend_spin_ready,
    height_box = height_box,
    follow_spin = true,
  })
end

function runway.save_runway_open_loop(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local y_clear = opts.y_clear
  if y_clear == nil then
    y_clear = true
  end
  if opts.face_right then
    local label = track.label
    for _ = 1, 6 do
      session:hold(1, { "RIGHT" }, label .. "_save_face")
    end
    for _ = 1, 4 do
      session:hold(1, {}, label .. "_save_settle")
    end
  end
  return runway.save_runway_fire_recipe(session, track, {
    policy = pol,
    y_clear = y_clear,
    crouch = false,
    arm_pump = opts.arm_pump,
    extend_spin_ready = opts.extend_spin_ready,
    wj_count = 2,
  })
end

runway.bubble_stationary_missile_clear = runway.stationary_missile_clear
runway.bubble_walk_brake_to_x = runway.walk_brake_to_x
runway.bubble_seat_max_left_fire = runway.seat_max_left_fire
runway.bubble_prepare_fire_run = runway.prepare_fire_run
runway.bubble_runway_dash = runway.runway_dash
runway.bubble_spin_glide = runway.spin_glide
runway.bubble_read_enemy_slot = runway.read_enemy_slot
runway.bubble_fire_phase_geometry = runway.fire_phase_geometry
runway.bubble_fire_phase_clear = runway.fire_phase_clear
runway.bubble_wait_fire_phase = runway.wait_fire_phase
runway.bubble_save_runway_fire_recipe = runway.save_runway_fire_recipe
runway.bubble_save_runway_open_loop_r15 = runway.save_runway_open_loop

return runway
