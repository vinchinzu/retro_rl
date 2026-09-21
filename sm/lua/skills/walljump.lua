-- Room-agnostic wall-jump climb skills.

local geometry = require("skills.geometry")
local knockback = require("skills.knockback")
local controller = require("skills.controller")

local walljump = {}

walljump.POSE_WALL_LATCH = controller.POSE_WALL_LATCH
walljump.WallJumpTiming = controller.WallJumpTiming
walljump.is_wall_latch = controller.is_wall_latch
walljump.is_knockback = knockback.is_knockback

function walljump.PreciseWallJumpTiming(opts)
  opts = opts or {}
  return {
    into = opts.into,
    away = opts.away,
    coast_frames = opts.coast_frames or 0,
    into_frames = opts.into_frames or 0,
    release_frames = opts.release_frames or 2,
    jump_frames = opts.jump_frames or 0,
    coast_buttons = opts.coast_buttons or { "A" },
    approach_buttons = opts.approach_buttons or { "A" },
    jump_buttons = opts.jump_buttons or { "A" },
  }
end

local function join(dir, extra)
  local names = { dir }
  for i = 1, #(extra or {}) do
    names[#names + 1] = extra[i]
  end
  return names
end

function walljump.precise_walljump_once(session, timing, opts)
  opts = opts or {}
  local reason = opts.reason or "precise_wj"
  local start_when = opts.start_when
  local contact_when = opts.contact_when
  local success_when = opts.success_when
  local landing_buttons = opts.landing_buttons or {}
  local landing_timeout = opts.landing_timeout or 0
  if start_when and not start_when(session.state) then
    error(reason .. " outside start window: " .. tostring(session.state))
  end
  local state = session.state
  local contact_seen = contact_when == nil
  local function phase(frames, names, label)
    for _ = 1, frames do
      state = session:hold(1, names, reason .. "_" .. label)
      if contact_when and contact_when(state) then
        contact_seen = true
      end
      if success_when and success_when(state) then
        return true
      end
    end
    return false
  end
  local phases = {
    { timing.coast_frames, timing.coast_buttons, "coast" },
    { timing.into_frames, join(timing.into, timing.approach_buttons), "contact" },
    { timing.release_frames, { timing.away }, "release" },
    { timing.jump_frames, join(timing.away, timing.jump_buttons), "jump" },
  }
  local success = success_when == nil
  for i = 1, #phases do
    if phase(phases[i][1], phases[i][2], phases[i][3]) then
      success = true
      break
    end
  end
  if not contact_seen then
    error(reason .. " missed wall contact: " .. tostring(state))
  end
  if success_when and not success then
    success = phase(landing_timeout, landing_buttons, "land")
  end
  if success_when and not success then
    error(reason .. " missed outcome: " .. tostring(state))
  end
  return state
end

local _pol
local function default_policy()
  if not _pol then
    _pol = require("skills.policies.bubble_to_bat")
  end
  return _pol
end

function walljump.wall_approach_band(state, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local rid = opts.room_id or pol.ROOM_ID
  local x0 = opts.x_min or pol.WJ_APPROACH_X[1]
  local x1 = opts.x_max or pol.WJ_APPROACH_X[2]
  local y0 = opts.y_min or pol.WJ_APPROACH_Y[1]
  local y1 = opts.y_max or pol.WJ_APPROACH_Y[2]
  local x = tonumber(state.samus_x) or 0
  local y = tonumber(state.samus_y) or 0
  return tonumber(state.room_id) == rid and x0 <= x and x <= x1 and y0 <= y and y <= y1
end

local function track_upd(session, track, state, policy, height_box)
  geometry.track_state(session, track, state, policy)
  if state.room_id ~= policy.ROOM_ID then
    return true
  end
  if height_box and state.samus_y <= policy.HEIGHT_CLASS_Y then
    height_box[1] = true
  end
  if geometry.phase_d_top_band(state, policy) then
    track.top_reached = true
    return true
  end
  return false
end

walljump._track_upd = track_upd

function walljump.wait_wall_ready(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local budget = opts.max_frames or pol.WJ_LATCH_TIMEOUT
  local edge = opts.into_x or pol.WJ_INTO_X
  local height_box = opts.height_box
  for _ = 1, budget do
    local state = session.state
    if track_upd(session, track, state, pol, height_box) then
      return true
    end
    if controller.is_wall_latch(state) then
      return true
    end
    if tonumber(state.samus_x) >= edge and walljump.wall_approach_band(state, { policy = pol }) then
      return true
    end
    state = session:hold(1, { "RIGHT", "B", "A" }, label .. "_wj_ready")
    if track_upd(session, track, state, pol, height_box) then
      return true
    end
    if controller.is_wall_latch(state) then
      return true
    end
  end
  local st = session.state
  return controller.is_wall_latch(st)
    or tonumber(st.samus_x) >= edge
    or track.top_reached and true
    or false
end

walljump.wait_wall_latch = walljump.wait_wall_ready

function walljump.walljump_approach_coast(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local height_box = opts.height_box
  for _ = 1, pol.SAVE_APPROACH_BA do
    local state = session:hold(1, { "B", "A" }, label .. "_dwj_coast")
    if track_upd(session, track, state, pol, height_box) then
      return track.top_reached and true or false
    end
  end
  session:hold(1, { "B" }, label .. "_dwj_rel")
  for _ = 1, pol.SAVE_APPROACH_IDLE do
    session:hold(1, {}, label .. "_dwj_idle")
  end
  for _ = 1, pol.SAVE_APPROACH_TURN do
    session:hold(1, { "LEFT" }, label .. "_dwj_turn")
  end
  return track.top_reached and true or false
end

function walljump.walljump_once(session, track, timing, opts)
  -- Primitive: walljump_once(session, timing, {reason=...})
  -- Climb skill: walljump_once(session, track, timing, opts)
  if type(track) == "table" and track.into and track.label == nil then
    local o = timing or {}
    if type(o) == "string" then
      o = { reason = o }
    end
    return controller.walljump_once(
      session,
      track,
      o.reason or "wj",
      o.stop_when
    )
  end
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local t = timing or pol.R15_WJ1
  local height_box = opts.height_box
  local reason = opts.reason or "wj"
  local function stop(state)
    return track_upd(session, track, state, pol, height_box)
  end
  controller.walljump_once(session, t, label .. "_" .. reason, stop)
  return track.top_reached and true or false
end

function walljump.consecutive_walljumps(session, track, jumps, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local height_box = opts.height_box
  local chain
  if jumps == nil then
    chain = { pol.R15_DOUBLE[1], pol.R15_DOUBLE[2] }
    local n = opts.count
    if n == nil then
      n = 2
    elseif n < 2 then
      n = 2
    end
    while #chain < n do
      chain[#chain + 1] = pol.R15_WJ2
    end
    if #chain > n then
      local trimmed = {}
      for i = 1, n do
        trimmed[i] = chain[i]
      end
      chain = trimmed
    end
  else
    chain = {}
    for i = 1, #jumps do
      chain[i] = jumps[i]
    end
    if opts.count then
      local n = opts.count
      if n < 1 then
        n = 1
      end
      local trimmed = {}
      for i = 1, n do
        if chain[i] then
          trimmed[#trimmed + 1] = chain[i]
        end
      end
      chain = trimmed
    end
  end
  local pre_approach = opts.pre_approach
  if pre_approach == nil then
    pre_approach = true
  end
  if pre_approach then
    if walljump.walljump_approach_coast(session, track, { policy = pol, height_box = height_box }) then
      return true
    end
  end
  if opts.extend_spin_ready then
    walljump.wait_wall_ready(session, track, { policy = pol, height_box = height_box })
    if track.top_reached or session.state.room_id ~= pol.ROOM_ID then
      return track.top_reached and true or false
    end
  end
  for i = 1, #chain do
    if track.top_reached or session.state.room_id ~= pol.ROOM_ID then
      break
    end
    walljump.walljump_once(session, track, chain[i], {
      policy = pol,
      height_box = height_box,
      reason = "wj" .. tostring(i),
    })
  end
  local follow_spin = opts.follow_spin
  if follow_spin == nil then
    follow_spin = true
  end
  if follow_spin and not track.top_reached and session.state.room_id == pol.ROOM_ID then
    local n_follow = opts.follow_frames or pol.SAVE_WJ_FOLLOW
    for _ = 1, n_follow do
      local state = session:hold(1, { "RIGHT", "B", "A" }, label .. "_dwj_pd_spin")
      if track_upd(session, track, state, pol, height_box) then
        break
      end
      if state.samus_x >= pol.RIGHT_SHELF_X and state.samus_y <= pol.MIDHIGH_Y then
        break
      end
    end
  end
  return track.top_reached and true or false
end

function walljump.double_walljump(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local height_box = opts.height_class_out or { false }
  local ok = walljump.consecutive_walljumps(session, track, pol.R15_DOUBLE, {
    policy = pol,
    pre_approach = true,
    extend_spin_ready = opts.extend_spin_ready,
    height_box = height_box,
    follow_spin = true,
  })
  if opts.height_class_out then
    opts.height_class_out[1] = height_box[1] and true or false
  end
  return ok
end

function walljump.walljump_second_left_wall(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local height_box = opts.height_box
  local budget = opts.seek_frames or pol.WJ2_LEFT_SEEK
  local n_into = opts.into_frames or pol.WJ2_LEFT_INTO
  local n_flip = opts.flip_frames or pol.WJ2_LEFT_FLIP
  local y_band = pol.WJ2_LEFT_Y or 200
  for _ = 1, budget do
    local state = session.state
    if track_upd(session, track, state, pol, height_box) then
      return track.top_reached and true or false
    end
    local pose = tonumber(state.pose)
    if pose == 83 or pose == 84
        or (tonumber(state.samus_x) <= pol.WJ2_LEFT_X and tonumber(state.samus_y) <= y_band) then
      break
    end
    state = session:hold(1, { "LEFT", "B", "A" }, label .. "_wj2_left_seek")
    if track_upd(session, track, state, pol, height_box) then
      return track.top_reached and true or false
    end
  end
  for _ = 1, n_into do
    local state = session:hold(1, { "RIGHT", "A" }, label .. "_wj2_left_into")
    if track_upd(session, track, state, pol, height_box) then
      return track.top_reached and true or false
    end
  end
  for _ = 1, n_flip do
    local state = session:hold(1, { "RIGHT", "B", "A" }, label .. "_wj2_left_flip")
    if track_upd(session, track, state, pol, height_box) then
      return track.top_reached and true or false
    end
  end
  return track.top_reached and true or false
end

function walljump.period_walljump_climb(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local height_box = opts.height_box
  local frames = opts.frames or 48
  local per = opts.period or pol.RIGHT_WJ_PERIOD
  local n_into = opts.into or pol.RIGHT_WJ_INTO
  local n_bounce = opts.bounce or pol.RIGHT_WJ_BOUNCE
  for i = 0, frames - 1 do
    local state = session.state
    if track_upd(session, track, state, pol, height_box) then
      return track.top_reached and true or false
    end
    if controller.is_wall_latch(state) then
      return walljump.consecutive_walljumps(session, track, pol.R15_DOUBLE, {
        policy = pol,
        pre_approach = false,
        extend_spin_ready = false,
        height_box = height_box,
        follow_spin = true,
      })
    end
    local ph = i % per
    if ph < n_into then
      state = session:hold(1, { "LEFT", "A" }, label .. "_pwj_into")
    elseif ph < n_into + n_bounce then
      state = session:hold(1, { "RIGHT", "A" }, label .. "_pwj_bounce")
    else
      state = session:hold(1, { "RIGHT", "B", "A" }, label .. "_pwj_spin")
    end
    if track_upd(session, track, state, pol, height_box) then
      return track.top_reached and true or false
    end
  end
  return track.top_reached and true or false
end

function walljump.damage_boost_hold(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local direction = opts.direction or "RIGHT"
  local n = opts.frames or pol.DMG_BOOST_HOLD_FRAMES
  local height_box = opts.height_box
  for _ = 1, n do
    local state = session:hold(1, { direction, "A" }, label .. "_dmg_boost")
    if track_upd(session, track, state, pol, height_box) then
      return track.top_reached and true or false
    end
    local pose = tonumber(session.state.pose)
    if not knockback.is_knockback(session.state)
        and pose ~= 25 and pose ~= 26 and pose ~= 129 and pose ~= 130 then
      break
    end
  end
  return track.top_reached and true or false
end

walljump.bubble_is_wall_latch = walljump.is_wall_latch
walljump.bubble_is_knockback = walljump.is_knockback
walljump.bubble_wall_approach_band = walljump.wall_approach_band
walljump.bubble_wait_wall_ready = walljump.wait_wall_ready
walljump.bubble_wait_wall_latch = walljump.wait_wall_latch
walljump.bubble_walljump_approach_coast = walljump.walljump_approach_coast
walljump.bubble_walljump_once = walljump.walljump_once
walljump.bubble_consecutive_walljumps = walljump.consecutive_walljumps
walljump.bubble_double_walljump_r15 = walljump.double_walljump
walljump.bubble_walljump_second_left_wall = walljump.walljump_second_left_wall
walljump.bubble_period_walljump_climb = walljump.period_walljump_climb
walljump.bubble_damage_boost_hold = walljump.damage_boost_hold

return walljump
