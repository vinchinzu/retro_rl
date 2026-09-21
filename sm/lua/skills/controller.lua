-- Shared Samus primitives (controller_common.py). Session hold/step live in runtime.

local ram = require("ram")
local geometry = require("skills.geometry")

local controller = {}

controller.MORPH_POSES = {
  [29] = true,
  [30] = true,
  [31] = true,
  [32] = true,
  [49] = true,
  [50] = true,
  [65] = true,
  [66] = true,
}
controller.POSE_WALL_LATCH = geometry.POSE_WALL_LATCH

function controller.hold(session, n, names, reason)
  return session:hold(n, names, reason)
end

function controller.require_room(session, room_id, label)
  local state = session.state
  if state.room_id ~= room_id then
    error(string.format(
      "%s: expected room 0x%04X, got 0x%04X at frame %s",
      label,
      room_id,
      state.room_id,
      tostring(session.frame)
    ))
  end
end

function controller.select_weapon(session, target, max_cycles)
  max_cycles = max_cycles or 8
  for _ = 1, max_cycles do
    if session.state.selected_item == target then
      return
    end
    session:hold(1, { "SELECT" }, "select_weapon")
    session:hold(25, {}, "select_weapon_settle")
  end
  if session.state.selected_item ~= target then
    error(string.format(
      "could not select weapon %s, still %s",
      tostring(target),
      tostring(session.state.selected_item)
    ))
  end
end

function controller.is_morph(pose)
  return controller.MORPH_POSES[tonumber(pose) or -1] or false
end

function controller.unmorph(session)
  local pose = session.state.pose
  if pose == 39 or pose == 40 or pose == 137 or pose == 138
      or pose == 9 or pose == 10 or controller.is_morph(pose) then
    session:hold(8, { "UP" }, "unmorph")
    if not controller.is_morph(session.state.pose) then
      session:hold(8, { "A" }, "unmorph")
    end
    session:hold(10, {}, "unmorph_settle")
  end
end

function controller.wait_until(session, pred, timeout, reason)
  return session:wait_until(pred, timeout or 120, reason or "wait")
end

function controller.settle_hold(session, frames, reason)
  return session:hold(frames or 12, {}, reason or "settle")
end

function controller.short_hop(session, direction, frames, extra, reason)
  extra = extra or {}
  local names = { direction }
  for i = 1, #extra do
    names[#names + 1] = extra[i]
  end
  return session:hold(frames, names, reason or "short_hop")
end

function controller.vertical_hop(session, frames, reason)
  return session:hold(frames, { "A" }, reason or "vertical_hop")
end

function controller.WallJumpTiming(opts)
  opts = opts or {}
  return {
    into = opts.into or "LEFT",
    flip = opts.flip or "RIGHT",
    into_frames = opts.into_frames or 20,
    amid_frames = opts.amid_frames or 4,
    flip_frames = opts.flip_frames or 8,
    delay_into_frames = opts.delay_into_frames or 0,
  }
end

function controller.is_wall_latch(state)
  return tonumber(state.pose) == controller.POSE_WALL_LATCH
end

function controller.walljump_once(session, timing, reason, stop_when)
  reason = reason or "wj"
  local phases = {
    { timing.delay_into_frames, { timing.into }, reason .. "_delay" },
    { timing.into_frames, { timing.into, "A" }, reason .. "_into" },
    { timing.amid_frames, { "A" }, reason .. "_amid" },
    { timing.flip_frames, { timing.flip, "A" }, reason .. "_flip" },
  }
  local state = session.state
  for p = 1, #phases do
    local n, buttons, phase_reason = phases[p][1], phases[p][2], phases[p][3]
    for _ = 1, n do
      state = session:hold(1, buttons, phase_reason)
      if stop_when and stop_when(state) then
        return state
      end
    end
  end
  return state
end

function controller.consecutive_walljumps(session, jumps, reason, gap_frames, stop_when)
  reason = reason or "wj_chain"
  gap_frames = gap_frames or 0
  local state = session.state
  for i = 1, #jumps do
    if stop_when and stop_when(state) then
      return state
    end
    state = controller.walljump_once(
      session,
      jumps[i],
      reason .. "_wj" .. tostring(i),
      stop_when
    )
    if stop_when and stop_when(state) then
      return state
    end
    if gap_frames > 0 and i < #jumps then
      for _ = 1, gap_frames do
        state = session:hold(1, {}, reason .. "_gap")
        if stop_when and stop_when(state) then
          return state
        end
      end
    end
  end
  return state
end

function controller.hold_until(session, pred, buttons, timeout, reason)
  timeout = timeout or 120
  reason = reason or "hold_until"
  buttons = buttons or {}
  for _ = 1, timeout do
    if pred(session.state) then
      return session.state
    end
    session:hold(1, buttons, reason)
  end
  error(string.format(
    "TimeoutError: %s timed out: frame=%s room=0x%04X xy=(%d,%d)",
    reason,
    tostring(session.frame),
    session.state.room_id,
    session.state.samus_x,
    session.state.samus_y
  ), 2)
end

function controller.wait_ordinary_room(session, room_id, opts)
  opts = opts or {}
  local settle_frames = opts.settle_frames or 200
  local label = opts.label or "settle"
  local x_range = opts.x_range
  local y_range = opts.y_range
  local min_settle_frame = opts.min_settle_frame or 15
  local state = session.state
  for frame = 0, settle_frames - 1 do
    state = session:hold(1, {}, label .. "_settle")
    if state.room_id == room_id
        and state.game_state == ram.GS_ORDINARY
        and state.door_transition == 0
        and frame > min_settle_frame then
      local x_ok = not x_range or (x_range[1] <= state.samus_x and state.samus_x <= x_range[2])
      local y_ok = not y_range or (y_range[1] <= state.samus_y and state.samus_y <= y_range[2])
      if x_ok and y_ok then
        return state
      end
    end
  end
  state = session.state
  if state.room_id ~= room_id then
    error(string.format(
      "%s: expected 0x%04X, got 0x%04X @ %s",
      label,
      room_id,
      state.room_id,
      tostring(state)
    ))
  end
  if x_range or y_range then
    error(string.format(
      "TimeoutError: %s: settled in room but position window missed xy=(%d,%d)",
      label,
      state.samus_x,
      state.samus_y
    ), 2)
  end
  return state
end

function controller.collect_item_mask(session, item_mask, opts)
  opts = opts or {}
  local timeout = opts.timeout or 600
  local reason = opts.reason or "collect_item"
  local buttons = opts.buttons or {}
  local band = ram.band
  if band(session.state.collected_items, item_mask) == item_mask then
    return session.state
  end
  local function has_items(state)
    return band(state.collected_items, item_mask) == item_mask
  end
  if #buttons > 0 then
    return controller.hold_until(session, has_items, buttons, timeout, reason)
  end
  local ok, err = pcall(function()
    return session:wait_until(has_items, timeout, reason)
  end)
  if not ok then
    error(string.format(
      "TimeoutError: %s: items still 0x%04X, want mask 0x%04X",
      reason,
      session.state.collected_items,
      item_mask
    ), 2)
  end
  return session.state
end

local MorphPolicy = {}
MorphPolicy.__index = MorphPolicy

function MorphPolicy:up_frames(attempt)
  local extra = attempt
  if extra > self.max_up_extra_steps then
    extra = self.max_up_extra_steps
  end
  return self.base_up + extra * self.up_step
end

function MorphPolicy:idle_frames(attempt)
  return self.base_idle + attempt
end

function MorphPolicy:tap1_frames(attempt)
  return self.base_tap1 + attempt * self.tap1_step
end

function MorphPolicy:tap2_frames(attempt)
  return self.base_tap2 + attempt * self.tap2_step
end

function MorphPolicy:poll_timeout(attempt)
  return self.base_poll + attempt * self.poll_step
end

function controller.MorphPolicy(opts)
  opts = opts or {}
  return setmetatable({
    max_attempts = opts.max_attempts or 5,
    base_up = opts.base_up or 4,
    up_step = opts.up_step or 2,
    max_up_extra_steps = opts.max_up_extra_steps or 3,
    base_idle = opts.base_idle or 3,
    base_tap1 = opts.base_tap1 or 5,
    tap1_step = opts.tap1_step or 2,
    base_tap2 = opts.base_tap2 or 6,
    tap2_step = opts.tap2_step or 3,
    release = opts.release or 3,
    base_poll = opts.base_poll or 28,
    poll_step = opts.poll_step or 4,
  }, MorphPolicy)
end

controller.DEFAULT_MORPH_POLICY = controller.MorphPolicy()

function controller.ensure_morph(session, opts)
  opts = opts or {}
  local pol = opts.policy or controller.DEFAULT_MORPH_POLICY
  local attempts = opts.max_attempts or pol.max_attempts
  for attempt = 0, attempts - 1 do
    if controller.is_morph(session.state.pose) then
      return session.state
    end
    session:hold(pol:up_frames(attempt), { "UP" }, "morph_pre")
    session:hold(pol:idle_frames(attempt), {}, "morph_idle")
    session:hold(pol:tap1_frames(attempt), { "DOWN" }, "morph_tap1")
    session:hold(pol.release, {}, "morph_release")
    session:hold(pol:tap2_frames(attempt), { "DOWN" }, "morph_tap2")
    local ok = pcall(function()
      session:wait_until(function(s)
        return controller.is_morph(s.pose)
      end, pol:poll_timeout(attempt), "morph_poll")
    end)
    if ok and controller.is_morph(session.state.pose) then
      return session.state
    end
  end
  error("TimeoutError: ensure_morph failed, pose=" .. tostring(session.state.pose), 2)
end

function controller.play_run_shoot_exit(session, opts)
  opts = opts or {}
  local from_room = opts.from_room
  local to_room = opts.to_room
  local direction = opts.direction
  local label = opts.label
  local run_frames = opts.run_frames or 40
  local shoot_frames = opts.shoot_frames or 6
  local spin_frames = opts.spin_frames or 40
  local hold_frames = opts.hold_frames or 160
  local settle_frames = opts.settle_frames or 200
  local super_door = opts.super_door
  controller.require_room(session, from_room, label)
  if super_door then
    pcall(controller.select_weapon, session, 2)
  else
    pcall(controller.select_weapon, session, 0)
  end
  session:hold(run_frames, { direction, "B" }, label .. "_run")
  if super_door then
    session:hold(shoot_frames, { direction, "X" }, label .. "_shoot")
    session:hold(20, {}, label .. "_super_fuse")
    session:hold(8, { direction, "X" }, label .. "_super2")
    session:hold(20, {}, label .. "_super_fuse2")
  else
    session:hold(shoot_frames, { direction, "B", "X" }, label .. "_shoot")
  end
  session:hold(spin_frames, { direction, "B", "A" }, label .. "_spin")
  local entered = false
  for _ = 1, hold_frames do
    local state = session:hold(1, { direction }, label .. "_hold")
    if state.room_id == to_room then
      entered = true
      break
    end
  end
  if not entered then
    error(string.format(
      "TimeoutError: %s: did not reach 0x%04X: %s",
      label,
      to_room,
      tostring(session.state)
    ), 2)
  end
  return controller.wait_ordinary_room(session, to_room, {
    settle_frames = settle_frames,
    label = label,
  })
end

function controller.traverse_door(session, opts)
  controller.play_run_shoot_exit(session, opts)
  if opts.entry_x_range == nil and opts.entry_y_range == nil then
    return session.state
  end
  local to_room = opts.to_room
  local st = session.state
  local x_ok = not opts.entry_x_range
    or (opts.entry_x_range[1] <= st.samus_x and st.samus_x <= opts.entry_x_range[2])
  local y_ok = not opts.entry_y_range
    or (opts.entry_y_range[1] <= st.samus_y and st.samus_y <= opts.entry_y_range[2])
  if st.room_id == to_room and st.game_state == ram.GS_ORDINARY and x_ok and y_ok then
    return st
  end
  local timeout = opts.settle_frames or 200
  if timeout > 90 then
    timeout = 90
  end
  session:wait_until(function(state)
    local xr = not opts.entry_x_range
      or (opts.entry_x_range[1] <= state.samus_x and state.samus_x <= opts.entry_x_range[2])
    local yr = not opts.entry_y_range
      or (opts.entry_y_range[1] <= state.samus_y and state.samus_y <= opts.entry_y_range[2])
    return state.room_id == to_room and state.game_state == ram.GS_ORDINARY and xr and yr
  end, timeout, (opts.label or "door") .. "_entry_window")
  return session.state
end

return controller
