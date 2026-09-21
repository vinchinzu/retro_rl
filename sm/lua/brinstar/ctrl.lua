-- Shared session wrappers for brinstar hops (Lua 5.1).
-- Session API: step / hold / wait_until / span / raw_actions.

local M = {}

M.GS_ORDINARY = 8
M.GS_DEAD = { [26] = true, [36] = true }
M.MORPH_POSES = {
  [29] = true, [30] = true, [31] = true, [32] = true,
  [49] = true, [50] = true, [65] = true, [66] = true,
}
M.POSE_KNOCKBACK = { [137] = true, [138] = true }
M.POSE_WALL_LATCH = 132

function M.num(v, default)
  v = tonumber(v)
  if v == nil then
    return default or 0
  end
  return v
end

function M.band(a, b)
  a = M.num(a)
  b = M.num(b)
  if bit and bit.band then
    return bit.band(a, b)
  end
  local r, p = 0, 1
  local i
  for i = 1, 16 do
    local aa = a % 2
    local bb = b % 2
    if aa == 1 and bb == 1 then
      r = r + p
    end
    a = (a - aa) / 2
    b = (b - bb) / 2
    p = p * 2
  end
  return r
end

function M.brief(state)
  if type(state) ~= "table" then
    return tostring(state)
  end
  return string.format(
    "room=0x%04X gs=%s xy=(%s,%s) pose=%s",
    M.num(state.room_id),
    tostring(state.game_state),
    tostring(state.samus_x or state.x),
    tostring(state.samus_y or state.y),
    tostring(state.pose)
  )
end

function M.is_morph(pose)
  return M.MORPH_POSES[M.num(pose)] == true
end

function M.is_ordinary(state)
  if state.phase == "ordinary_gameplay" then
    return true
  end
  return M.num(state.game_state) == M.GS_ORDINARY
    and M.num(state.door_transition) == 0
end

function M.is_dead(state)
  if state.phase == "death_or_game_over" then
    return true
  end
  if M.GS_DEAD[M.num(state.game_state)] then
    return true
  end
  return M.num(state.health) == 0
end

function M.area_boss_bits(state, area_index)
  local bits = state.boss_bits
  if type(bits) ~= "table" then
    return 0
  end
  if bits[0] ~= nil then
    return M.num(bits[area_index])
  end
  return M.num(bits[area_index + 1])
end

function M.hold(session, frames, names, reason)
  frames = M.num(frames)
  if frames <= 0 then
    return session.state
  end
  names = names or {}
  if session.hold then
    session:hold(frames, names, reason)
    return session.state
  end
  local span = { names = names, frames = frames, reason = reason }
  if session.span then
    session:span(span)
    return session.state
  end
  local i
  for i = 1, frames do
    session:step(names, reason)
  end
  return session.state
end

function M.span(session, s)
  if session.span then
    session:span(s)
    return session.state
  end
  return M.hold(session, s.frames, s.names, s.reason)
end

function M.raw_actions(session, actions, reason)
  if session.raw_actions then
    session:raw_actions(actions, reason)
    return session.state
  end
  local i
  for i = 1, #actions do
    local a = actions[i]
    if type(a) == "table" and type(a.names) == "table" then
      M.hold(session, a.frames or 1, a.names, a.reason or reason)
    else
      session:step(a or {}, reason)
    end
  end
  return session.state
end

function M.play_tape(session, spans, reason, hold_frames)
  hold_frames = hold_frames or 16
  if session.raw_actions then
    local actions = {}
    local i, j
    for i = 1, #spans do
      for j = 1, hold_frames do
        actions[#actions + 1] = spans[i]
      end
    end
    session:raw_actions(actions, reason)
    return session.state
  end
  local i
  for i = 1, #spans do
    M.span(session, {
      names = spans[i],
      frames = hold_frames,
      reason = reason,
    })
  end
  return session.state
end

function M.wait_until(session, pred, timeout, reason, names)
  timeout = timeout or 120
  reason = reason or "wait"
  if session.wait_until and (names == nil or (type(names) == "table" and names[1] == nil)) then
    session:wait_until(pred, timeout, reason)
    if pred(session.state) then
      return session.state
    end
  end
  local i
  for i = 1, timeout do
    if pred(session.state) then
      return session.state
    end
    M.hold(session, 1, names or {}, reason)
  end
  error(reason .. " timed out: " .. M.brief(session.state))
end

function M.require_room(session, room_id, label)
  local st = session.state
  if M.num(st.room_id) ~= room_id then
    error(string.format(
      "%s: expected room 0x%04X, got 0x%04X at frame %s",
      label,
      room_id,
      M.num(st.room_id),
      tostring(session.frame)
    ))
  end
end

function M.require_ordinary_room(session, room_id, label)
  M.require_room(session, room_id, label)
  if not M.is_ordinary(session.state) then
    error(string.format(
      "%s: expected room 0x%04X ordinary, got %s",
      label,
      room_id,
      M.brief(session.state)
    ))
  end
end

function M.select_weapon(session, target, max_cycles)
  max_cycles = max_cycles or 8
  local i
  for i = 1, max_cycles do
    if M.num(session.state.selected_item) == target then
      return true
    end
    M.hold(session, 1, {"SELECT"}, "select_weapon")
    M.hold(session, 25, {}, "select_weapon_settle")
  end
  return M.num(session.state.selected_item) == target
end

function M.try_select(session, target)
  M.select_weapon(session, target, 8)
end

function M.unmorph(session)
  local pose = M.num(session.state.pose)
  if pose == 39 or pose == 40 or pose == 137 or pose == 138
      or pose == 9 or pose == 10 or M.is_morph(pose) then
    M.hold(session, 8, {"UP"}, "unmorph")
    if not M.is_morph(session.state.pose) then
      M.hold(session, 8, {"A"}, "unmorph")
    end
    M.hold(session, 10, {}, "unmorph_settle")
  end
end

function M.ensure_morph(session, max_attempts)
  max_attempts = max_attempts or 5
  local attempt
  for attempt = 0, max_attempts - 1 do
    if M.is_morph(session.state.pose) then
      return session.state
    end
    local up = 4 + math.min(attempt, 3) * 2
    local idle = 3 + attempt
    local tap1 = 5 + attempt * 2
    local tap2 = 6 + attempt * 3
    local poll = 28 + attempt * 4
    M.hold(session, up, {"UP"}, "morph_pre")
    M.hold(session, idle, {}, "morph_idle")
    M.hold(session, tap1, {"DOWN"}, "morph_tap1")
    M.hold(session, 3, {}, "morph_release")
    M.hold(session, tap2, {"DOWN"}, "morph_tap2")
    local p
    local got = false
    for p = 1, poll do
      if M.is_morph(session.state.pose) then
        got = true
        break
      end
      M.hold(session, 1, {}, "morph_poll")
    end
    if got then
      return session.state
    end
  end
  error("ensure_morph failed, pose=" .. tostring(session.state.pose))
end

function M.wait_ordinary_room(session, room_id, settle_frames, label, min_settle)
  settle_frames = settle_frames or 200
  min_settle = min_settle or 15
  local frame
  for frame = 0, settle_frames - 1 do
    local state = M.hold(session, 1, {}, label .. "_settle")
    if M.num(state.room_id) == room_id
        and M.num(state.game_state) == M.GS_ORDINARY
        and M.num(state.door_transition) == 0
        and frame > min_settle then
      return state
    end
  end
  local state = session.state
  if M.num(state.room_id) ~= room_id then
    error(string.format(
      "%s: expected 0x%04X, got 0x%04X @ %s",
      label,
      room_id,
      M.num(state.room_id),
      M.brief(state)
    ))
  end
  return state
end

function M.unique_names(list)
  local seen, out = {}, {}
  local i
  for i = 1, #list do
    local n = list[i]
    if n and n ~= "" and not seen[n] then
      seen[n] = true
      out[#out + 1] = n
    end
  end
  return out
end

function M.vertical_hop(session, frames, reason)
  return M.hold(session, frames, {"A"}, reason or "vertical_hop")
end

function M.room(rooms, key, fallback)
  if type(rooms) == "table" and rooms[key] then
    return rooms[key]
  end
  return fallback
end

return M
