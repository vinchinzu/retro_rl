-- Blue/gray door stage, beam-open, and period exit-push helpers.

local door_exit = {}

door_exit.JUMP_ENTER_PERIOD = 30
door_exit.JUMP_ENTER_JUMP_END = 4
door_exit.JUMP_ENTER_SPIN_END = 10
door_exit.JUMP_ENTER_RESHOT_END = 14

function door_exit.lip_stage(session, opts)
  opts = opts or {}
  local label = opts.label or "lip"
  local backoff = opts.backoff or "RIGHT"
  local face = opts.face or "LEFT"
  local backoff_frames = opts.backoff_frames or 8
  local face_frames = opts.face_frames or 8
  local release_frames = opts.release_frames or 6
  local settle_frames = opts.settle_frames or 0
  if settle_frames > 0 then
    session:hold(settle_frames, {}, label .. "_approach_settle")
  end
  session:hold(backoff_frames, { backoff }, label .. "_lip_backoff")
  session:hold(face_frames, { face }, label .. "_face")
  session:hold(release_frames, {}, label .. "_face_release")
end

function door_exit.beam_open_door(session, opts)
  opts = opts or {}
  local label = opts.label or "door"
  local shots = opts.shots or 6
  local shot_frames = opts.shot_frames or 4
  local fuse_frames = opts.fuse_frames or 14
  local shot_buttons = opts.shot_buttons or { "X" }
  for _ = 1, shots do
    session:hold(shot_frames, shot_buttons, label .. "_door_shot")
    session:hold(fuse_frames, {}, label .. "_door_fuse")
  end
end

function door_exit.drain_door_transition(session, target_room, opts)
  opts = opts or {}
  local max_frames = opts.max_frames or 80
  local reason = opts.reason or "transition"
  local state = session.state
  for _ = 1, max_frames do
    state = session:hold(1, {}, reason)
    if state.room_id == target_room and state.door_transition == 0 then
      break
    end
  end
  return state
end

function door_exit.period_exit_push(session, target_room, opts)
  opts = opts or {}
  local label = opts.label or "exit"
  local max_frames = opts.max_frames
  local period = opts.period
  local windows = opts.windows
  local transition_drain = opts.transition_drain or 0
  local drain_reason = opts.transition_reason or (label .. "_transition")
  local guard = opts.guard
  local on_wrong_room = opts.on_wrong_room
  local on_state = opts.on_state
  if not windows or #windows == 0 then
    error(label .. ": period_exit_push requires at least one window")
  end
  for index = 0, max_frames - 1 do
    local phase = index % period
    local buttons = windows[#windows][2]
    local reason_suffix = windows[#windows][3]
    for w = 1, #windows do
      local end_ex, btns, suffix = windows[w][1], windows[w][2], windows[w][3]
      if phase < end_ex then
        buttons = btns
        reason_suffix = suffix
        break
      end
    end
    local state = session:hold(1, buttons, label .. "_" .. reason_suffix)
    if guard then
      guard(state)
    end
    if on_state then
      on_state(state)
    end
    if state.room_id == target_room then
      return state
    end
    if on_wrong_room then
      on_wrong_room(state)
    end
    if transition_drain > 0 and state.door_transition ~= 0 then
      state = door_exit.drain_door_transition(session, target_room, {
        max_frames = transition_drain,
        reason = drain_reason,
      })
      if state.room_id == target_room then
        return state
      end
    end
  end
  error("TimeoutError: " .. label .. ": exit timed out: " .. tostring(session.state), 2)
end

function door_exit.jump_enter_exit(session, target_room, opts)
  opts = opts or {}
  local direction = opts.direction or "LEFT"
  local period = opts.period or door_exit.JUMP_ENTER_PERIOD
  local jump_end = opts.jump_end or door_exit.JUMP_ENTER_JUMP_END
  local spin_end = opts.spin_end or door_exit.JUMP_ENTER_SPIN_END
  local reshot_end = opts.reshot_end or door_exit.JUMP_ENTER_RESHOT_END
  return door_exit.period_exit_push(session, target_room, {
    label = opts.label or "jump_enter",
    max_frames = opts.max_frames or 700,
    period = period,
    windows = {
      { jump_end, { direction, "A" }, "jump" },
      { spin_end, { direction, "A", "B" }, "jump_spin" },
      { reshot_end, { "X" }, "reshot" },
      { period, { direction, "B" }, "exit" },
    },
    transition_drain = opts.transition_drain or 80,
    transition_reason = (opts.label or "jump_enter") .. "_transition",
    guard = opts.guard,
    on_wrong_room = opts.on_wrong_room,
  })
end

return door_exit
