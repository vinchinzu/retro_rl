-- Knockback pose checks and open-loop escape helpers.

local geometry = require("skills.geometry")
local controller = require("skills.controller")

local knockback = {}

function knockback.is_knockback(state, poses)
  poses = poses or geometry.POSE_KNOCKBACK
  return poses[tonumber(state.pose) or -1] or false
end

function knockback.hold_through_knockback(session, frames, opts)
  opts = opts or {}
  local label = opts.label or "kb"
  local reason = opts.reason or "kb"
  return session:hold(frames, {}, label .. "_" .. reason)
end

function knockback.escape_knockback_spin(session, opts)
  opts = opts or {}
  local prefer_dir = opts.prefer_dir or "RIGHT"
  local run_frames = opts.run_frames or 6
  local spin_frames = opts.spin_frames or 20
  local label = opts.label or "kb"
  local run_with = opts.run_with or { "B" }
  local spin_with = opts.spin_with or { "B", "A" }
  local run_reason = opts.run_reason or "kb_run"
  local spin_reason = opts.spin_reason or "kb_spin"
  local stop_room_id = opts.stop_room_id
  local break_on_motion_clear = opts.break_on_motion_clear
  local motion_clear_px = opts.motion_clear_px or 2
  local start_x = tonumber(session.state.samus_x) or 0
  if opts.ensure_beam and tonumber(session.state.selected_item) ~= 0 then
    controller.select_weapon(session, 0)
  end
  local run_btns = { prefer_dir }
  for i = 1, #run_with do
    run_btns[#run_btns + 1] = run_with[i]
  end
  local spin_btns = { prefer_dir }
  for i = 1, #spin_with do
    spin_btns[#spin_btns + 1] = spin_with[i]
  end
  local st = session.state
  for _ = 1, run_frames do
    st = session:hold(1, run_btns, label .. "_" .. run_reason)
    if stop_room_id ~= nil and tonumber(st.room_id) == stop_room_id then
      return st
    end
  end
  for _ = 1, spin_frames do
    st = session:hold(1, spin_btns, label .. "_" .. spin_reason)
    if stop_room_id ~= nil and tonumber(st.room_id) == stop_room_id then
      return st
    end
    if break_on_motion_clear then
      local dx = (tonumber(st.samus_x) or 0) - start_x
      if dx < 0 then
        dx = -dx
      end
      if not knockback.is_knockback(st) and dx > motion_clear_px then
        break
      end
    end
  end
  return session.state
end

function knockback.escape_kb(session, label, prefer, opts)
  opts = opts or {}
  return knockback.escape_knockback_spin(session, {
    prefer_dir = prefer,
    run_frames = opts.run_frames or 6,
    spin_frames = opts.spin_frames or 18,
    label = label,
    stop_room_id = opts.stop_room_id,
  })
end

return knockback
