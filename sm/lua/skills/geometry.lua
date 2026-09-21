-- Room-agnostic pose sets, xy bands, true-ground, and climb track.

local geometry = {}

local function set(list)
  local s = {}
  for i = 1, #list do
    s[list[i]] = true
  end
  return s
end

geometry.TRUE_GROUND = set({ 1, 2, 9, 10 })
geometry.STAND_PIN = set({ 1, 2, 9, 10, 25, 26, 27, 28 })
geometry.STANDING_POSES = set({ 1, 2, 9, 10, 25, 26, 27, 28, 37, 38, 137, 138 })
geometry.POSE_KNOCKBACK = set({ 137, 138 })
geometry.LEDGE_POSES = set({ 1, 2, 9, 10, 25, 26, 27, 28, 37, 38 })
geometry.POSE_STAND_LEFT = set({ 2, 10 })
geometry.POSE_STAND_RIGHT = set({ 1, 9 })
geometry.FACE_LEFT_POSES = set({ 2, 6, 8, 10 })
geometry.FACE_RIGHT_POSES = set({ 1, 5, 7, 9 })
geometry.STAND_LOCOMOTION_POSES = set({ 1, 2, 5, 6, 7, 8, 9, 10 })
geometry.CROUCH_POSES = set({ 37, 38, 39, 40, 41, 42 })
geometry.SPIN_POSES = set({ 25, 26, 27, 28 })
geometry.LAND_POSES = set({ 163, 164, 165, 166, 167 })
geometry.GUN_JUMP_POSES = set({ 47, 48, 81, 82, 83, 84 })
geometry.POSE_WALL_LATCH = 132

function geometry.is_true_ground(st, poses, max_vy)
  poses = poses or geometry.TRUE_GROUND
  max_vy = max_vy or 1
  local vy = tonumber(st.velocity_y) or 0
  if vy < 0 then
    vy = -vy
  end
  if vy > max_vy then
    return false
  end
  return poses[tonumber(st.pose) or -1] or false
end

function geometry.is_stand_pin_pose(st, poses, max_vy)
  poses = poses or geometry.STAND_PIN
  max_vy = max_vy or 1
  local vy = tonumber(st.velocity_y) or 0
  if vy < 0 then
    vy = -vy
  end
  if vy > max_vy then
    return false
  end
  return poses[tonumber(st.pose) or -1] or false
end

function geometry.in_xy_band(st, opts)
  opts = opts or {}
  if opts.room_id ~= nil and tonumber(st.room_id) ~= opts.room_id then
    return false
  end
  local x, y = tonumber(st.samus_x) or 0, tonumber(st.samus_y) or 0
  if opts.x_min ~= nil and x < opts.x_min then
    return false
  end
  if opts.x_max ~= nil and x > opts.x_max then
    return false
  end
  if opts.y_min ~= nil and y < opts.y_min then
    return false
  end
  if opts.y_max ~= nil and y > opts.y_max then
    return false
  end
  return true
end

function geometry.phase_c_usable_right_contact(st, policy)
  local x, y = tonumber(st.samus_x) or 0, tonumber(st.samus_y) or 0
  return tonumber(st.room_id) == policy.ROOM_ID
    and policy.PHASE_C_X_MIN <= x
    and x <= policy.CAVITY_X_MAX
    and policy.PHASE_C_Y_MIN <= y
    and y <= policy.PHASE_C_Y_MAX
end

function geometry.phase_d_top_band(st, policy)
  return tonumber(st.room_id) == policy.ROOM_ID
    and (tonumber(st.samus_y) or 0) <= policy.PHASE_D_Y
    and (tonumber(st.samus_x) or 0) >= policy.PHASE_D_X
end

function geometry.phase_d_near_top(st, policy, slack)
  slack = slack or 40
  if geometry.phase_d_top_band(st, policy) then
    return true
  end
  return tonumber(st.room_id) == policy.ROOM_ID
    and (tonumber(st.samus_y) or 0) <= policy.PHASE_D_Y + slack
    and (tonumber(st.samus_x) or 0) >= policy.PHASE_D_X - slack
end

function geometry.on_mid_iso_pin(st, policy)
  local stand_lo, stand_hi = policy.MID_STAND_X[1], policy.MID_STAND_X[2]
  local vy = tonumber(st.velocity_y) or 0
  if vy < 0 then
    vy = -vy
  end
  local x = tonumber(st.samus_x) or 0
  return vy <= 2
    and policy.STAND_PIN[tonumber(st.pose) or -1]
    and stand_lo <= x
    and x <= stand_hi
    and (tonumber(st.samus_y) or 0) <= policy.MID_Y + 10
end

function geometry.on_launch_lip(st, policy)
  local lip_lo, lip_hi = policy.LIP_X[1], policy.LIP_X[2]
  local lip_y_lo, lip_y_hi = policy.LIP_Y[1], policy.LIP_Y[2]
  return geometry.is_stand_pin_pose(st, policy.STAND_PIN)
    and lip_lo <= st.samus_x
    and st.samus_x <= lip_hi
    and lip_y_lo <= st.samus_y
    and st.samus_y <= lip_y_hi
end

function geometry.on_right_shelf(st, policy)
  return geometry.is_stand_pin_pose(st, policy.STAND_PIN)
    and st.samus_x >= policy.RIGHT_SHELF_X
    and st.samus_x <= policy.CAVITY_X_MAX
    and st.samus_y <= policy.RIGHT_SHELF_Y
    and st.samus_y >= 200
end

function geometry.on_save_runway(st, policy)
  local x_lo, x_hi = policy.SAVE_RUNWAY_X[1], policy.SAVE_RUNWAY_X[2]
  local y_lo, y_hi = policy.SAVE_RUNWAY_Y[1], policy.SAVE_RUNWAY_Y[2]
  local x, y = tonumber(st.samus_x) or 0, tonumber(st.samus_y) or 0
  return geometry.is_stand_pin_pose(st, policy.STAND_PIN)
    and x_lo <= x
    and x <= x_hi
    and y_lo <= y
    and y <= y_hi
end

function geometry.ClimbTrack(opts)
  opts = opts or {}
  return {
    label = opts.label or "climb",
    max_x = opts.max_x or 0,
    min_y = opts.min_y or 0,
    mid_reached = opts.mid_reached or false,
    top_reached = opts.top_reached or false,
    door_reached = opts.door_reached or false,
    standing_mid_pinned = opts.standing_mid_pinned or false,
    launched = opts.launched or false,
    phase_c_hit = opts.phase_c_hit or false,
    dump_path = opts.dump_path,
    phase_c_dumped = false,
    stop_at_phase_c = opts.stop_at_phase_c or false,
  }
end

geometry.BubbleTrack = geometry.ClimbTrack

function geometry.metrics_dict(track)
  return {
    max_x = track.max_x,
    min_y = track.min_y,
    mid_reached = track.mid_reached,
    top_reached = track.top_reached,
    phase_c_hit = track.phase_c_hit,
  }
end

function geometry.PhaseStop(phase, state, opts)
  opts = opts or {}
  error(string.format(
    "%s:%s room=0x%04X pose=%d xy=(%d,%d) vx=%d vy=%d",
    opts.label or "phase_stop",
    phase,
    state.room_id,
    state.pose,
    state.samus_x,
    state.samus_y,
    state.velocity_x,
    state.velocity_y
  ))
end

function geometry.BubblePhaseStop(phase, state, metrics)
  geometry.PhaseStop(phase, state, { label = "bubble_phase_stop", metrics = metrics })
end

function geometry.track_state(session, track, state, policy)
  if state.samus_x > track.max_x then
    track.max_x = state.samus_x
  end
  if state.samus_y < track.min_y then
    track.min_y = state.samus_y
  end
  if state.samus_y <= policy.MID_Y and state.samus_x >= 90 then
    track.mid_reached = true
  end
  if geometry.phase_d_top_band(state, policy) then
    track.top_reached = true
  end
  if geometry.phase_c_usable_right_contact(state, policy) then
    if not track.phase_c_hit then
      track.phase_c_hit = true
      if track.stop_at_phase_c then
        geometry.PhaseStop("C", state, {
          label = "bubble_phase_stop",
          metrics = geometry.metrics_dict(track),
        })
      end
    end
  end
end

function geometry.avoid_wrong_door(session, track, state, policy)
  local label = track.label
  local x, y = state.samus_x, state.samus_y
  local y_lo, y_hi = policy.SAVE_RUNWAY_Y[1], policy.SAVE_RUNWAY_Y[2]
  local fire_lo = policy.SAVE_RUNWAY_FIRE_X[1]
  local on_save_platform = y_lo <= y and y <= y_hi
  if x < 22 or (x < 55 and not on_save_platform) then
    session:hold(1, { "RIGHT", "B" }, label .. "_avoid_left")
    return true
  end
  if on_save_platform and x < fire_lo - 1 then
    session:hold(1, { "RIGHT", "B" }, label .. "_avoid_save_door")
    return true
  end
  if x > 470 and 300 <= y and y <= 430 then
    session:hold(1, { "LEFT", "B" }, label .. "_avoid_sc")
    return true
  end
  return false
end

function geometry.new_climb_track(session, opts)
  opts = opts or {}
  local st = session.state
  return geometry.ClimbTrack({
    label = opts.label or "climb",
    max_x = st.samus_x,
    min_y = st.samus_y,
    dump_path = opts.dump_phase_c,
    stop_at_phase_c = opts.stop_at_phase_c or false,
  })
end

return geometry
