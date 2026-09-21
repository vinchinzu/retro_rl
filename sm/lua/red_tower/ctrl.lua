-- Shared session helpers for red_tower / wrecked_ship hops.
-- Lua 5.1. session:step(names, reason). Names: B Y SELECT START UP DOWN LEFT RIGHT A X L R.

local ram = require("ram")
local rooms = require("rooms")
local knockback = require("skills.knockback")

local M = {}

M.GS_ORDINARY = ram.GS_ORDINARY or 8
M.GRAVITY_MASK = ram.GRAVITY_MASK or 0x0020
M.HI_JUMP_MASK = ram.HI_JUMP_MASK or 0x0100
M.ICE_BEAM_MASK = ram.ICE_BEAM_MASK or 0x0002
M.FACING_LEFT = ram.FACING_LEFT or 4
M.FACING_RIGHT = ram.FACING_RIGHT or 8
M.ROOM_RED_TOWER = rooms.ROOM_RED_TOWER or 0xA253
M.ROOM_BAT = rooms.ROOM_BAT or 0xA3DD
M.ROOM_BELOW_SPAZER = rooms.ROOM_BELOW_SPAZER or 0xA408
M.ROOM_WEST_TUNNEL = rooms.ROOM_WEST_TUNNEL or 0xCF54
M.ROOM_HELLWAY = rooms.ROOM_HELLWAY or 0xA2F7
M.ROOM_CATERPILLAR = rooms.ROOM_CATERPILLAR or 0xA322
M.ROOM_ALPHA_PB = rooms.ROOM_ALPHA_PB or 0xA3AE
M.ROOM_WAREHOUSE = rooms.ROOM_WAREHOUSE or 0xA6A1
M.ROOM_EAST_TUNNEL = rooms.ROOM_EAST_TUNNEL or 0xCF80
M.ROOM_GLASS = rooms.ROOM_GLASS or 0xCEFB
M.MORPH_POSES = {
  [29] = true, [30] = true, [31] = true, [32] = true,
  [49] = true, [50] = true, [65] = true, [66] = true,
}

function M.num(v, default)
  v = tonumber(v)
  if v == nil then
    return default or 0
  end
  return v
end

function M.band(a, b)
  if bit and bit.band then
    return bit.band(a, b)
  end
  a = M.num(a) % 65536
  b = M.num(b) % 65536
  local r, bitv = 0, 1
  while a > 0 and b > 0 do
    if (a % 2) == 1 and (b % 2) == 1 then
      r = r + bitv
    end
    a = math.floor(a / 2)
    b = math.floor(b / 2)
    bitv = bitv * 2
  end
  return r
end

function M.brief(state)
  if type(state) ~= "table" then
    return tostring(state)
  end
  return string.format(
    "room=0x%04X gs=%s xy=(%s,%s) p=%s",
    M.num(state.room_id),
    tostring(state.game_state),
    tostring(state.samus_x or state.x),
    tostring(state.samus_y or state.y),
    tostring(state.pose)
  )
end

M.fmt_state = M.brief

function M.set(...)
  local out, i, n = {}, 1, select("#", ...)
  for i = 1, n do
    out[select(i, ...)] = true
  end
  return out
end

function M.settle_hold(session, frames, reason)
  return M.hold(session, frames, {}, reason or "settle")
end

function M.timeout(msg)
  local ok, runtime = pcall(require, "runtime")
  if ok and runtime and runtime.timeout then
    runtime.timeout(msg)
  end
  error(msg)
end

function M.x(state)
  return M.num(state.samus_x or state.x)
end

function M.y(state)
  return M.num(state.samus_y or state.y)
end

function M.hold(session, frames, names, reason)
  local i
  for i = 1, frames do
    if names and names[1] then
      session:step(names, reason)
    elseif session.idle then
      session:idle(reason)
    else
      session:step({}, reason)
    end
  end
  return session.state
end

function M.step(session, names, reason)
  if names and names[1] then
    session:step(names, reason)
  elseif session.idle then
    session:idle(reason)
  else
    session:step({}, reason)
  end
  return session.state
end

function M.require_room(session, room_id, label)
  local st = session.state
  if M.num(st.room_id) ~= room_id then
    M.timeout(string.format(
      "%s: expected room 0x%04X, got 0x%04X at frame %s",
      label, room_id, M.num(st.room_id), tostring(session.frame)
    ))
  end
end

function M.is_morph(pose)
  return M.MORPH_POSES[M.num(pose)] == true
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
    M.hold(session, 4 + math.min(attempt, 3) * 2, {"UP"}, "morph_pre")
    M.hold(session, 3 + attempt, {}, "morph_idle")
    M.hold(session, 5 + attempt * 2, {"DOWN"}, "morph_tap1")
    M.hold(session, 3, {}, "morph_release")
    M.hold(session, 6 + attempt * 3, {"DOWN"}, "morph_tap2")
    local poll
    for poll = 1, 28 + attempt * 4 do
      if M.is_morph(session.state.pose) then
        return session.state
      end
      M.hold(session, 1, {}, "morph_poll")
    end
  end
  M.timeout("ensure_morph failed, pose=" .. tostring(session.state.pose))
end

function M.select_weapon(session, target, max_cycles)
  max_cycles = max_cycles or 8
  local i
  for i = 1, max_cycles do
    if M.num(session.state.selected_item) == target then
      return
    end
    M.hold(session, 1, {"SELECT"}, "select_weapon")
    M.hold(session, 25, {}, "select_weapon_settle")
  end
  if M.num(session.state.selected_item) ~= target then
    M.timeout(string.format(
      "could not select weapon %s, still %s",
      tostring(target), tostring(session.state.selected_item)
    ))
  end
end

function M.wait_ordinary_room(session, room_id, opts)
  opts = opts or {}
  local settle = opts.settle_frames or 200
  local label = opts.label or "settle"
  local min_settle = opts.min_settle_frame or 15
  local x_range = opts.x_range
  local y_range = opts.y_range
  local frame
  for frame = 0, settle - 1 do
    local state = M.hold(session, 1, {}, label .. "_settle")
    if M.num(state.room_id) == room_id
        and M.num(state.game_state) == M.GS_ORDINARY
        and M.num(state.door_transition) == 0
        and frame > min_settle then
      local sx, sy = M.x(state), M.y(state)
      if x_range and not (x_range[1] <= sx and sx <= x_range[2]) then
        -- continue
      elseif y_range and not (y_range[1] <= sy and sy <= y_range[2]) then
        -- continue
      else
        return state
      end
    end
  end
  local state = session.state
  if M.num(state.room_id) ~= room_id then
    M.timeout(string.format(
      "%s: expected 0x%04X, got %s", label, room_id, M.brief(state)
    ))
  end
  if x_range or y_range then
    M.timeout(string.format(
      "%s: settled in room but position window missed %s", label, M.brief(state)
    ))
  end
  return state
end

function M.play_script(session, runs, opts)
  opts = opts or {}
  local reason = opts.reason or "tape"
  local room_id = opts.room_id
  local stop_when = opts.stop_when
  local on_lag = opts.on_lag or "ignore"
  local i, f
  for i = 1, #runs do
    local n = runs[i][1]
    local names = runs[i][2] or {}
    for f = 1, n do
      if room_id and M.num(session.state.room_id) ~= room_id then
        return session.state
      end
      if stop_when and stop_when(session.state) then
        return session.state
      end
      if on_lag == "break" then
        local pose = M.num(session.state.pose)
        if pose == 137 or pose == 138 or pose == 164 then
          M.break_rle_lag(session, reason .. "_lag")
        end
      end
      M.step(session, names, reason)
      if stop_when and stop_when(session.state) then
        return session.state
      end
    end
  end
  return session.state
end

function M.play_rle_room_exit(session, opts)
  M.require_room(session, opts.from_room, opts.label)
  M.play_script(session, opts.script, {
    reason = opts.label .. "_body",
    room_id = opts.from_room,
  })
  return M.wait_ordinary_room(session, opts.to_room, {
    settle_frames = opts.settle_frames or 300,
    label = opts.label,
  })
end

function M.play_run_shoot_exit(session, opts)
  local from_room = opts.from_room
  local to_room = opts.to_room
  local direction = opts.direction
  local label = opts.label
  M.require_room(session, from_room, label)
  local ok_sel, err = pcall(M.select_weapon, session, opts.super_door and 2 or 0)
  if not ok_sel then
    -- keep current weapon
  end
  M.hold(session, opts.run_frames or 40, {direction, "B"}, label .. "_run")
  if opts.super_door then
    M.hold(session, opts.shoot_frames or 6, {direction, "X"}, label .. "_shoot")
    M.hold(session, 20, {}, label .. "_super_fuse")
    M.hold(session, 8, {direction, "X"}, label .. "_super2")
    M.hold(session, 20, {}, label .. "_super_fuse2")
  else
    M.hold(session, opts.shoot_frames or 6, {direction, "B", "X"}, label .. "_shoot")
  end
  M.hold(session, opts.spin_frames or 40, {direction, "B", "A"}, label .. "_spin")
  local entered = false
  local i
  for i = 1, opts.hold_frames or 160 do
    local state = M.hold(session, 1, {direction}, label .. "_hold")
    if M.num(state.room_id) == to_room then
      entered = true
      break
    end
  end
  if not entered then
    M.timeout(string.format("%s: did not reach 0x%04X: %s", label, to_room, M.brief(session.state)))
  end
  return M.wait_ordinary_room(session, to_room, {
    settle_frames = opts.settle_frames or 200,
    label = label,
  })
end

function M.is_knockback(state)
  if knockback and knockback.is_knockback then
    return knockback.is_knockback(state)
  end
  local pose = M.num(state.pose)
  return pose == 137 or pose == 138
end

function M.escape_kb(session, opts)
  opts = opts or {}
  if knockback and knockback.escape_knockback_spin then
    return knockback.escape_knockback_spin(session, opts)
  end
  local dir = opts.prefer_dir or "RIGHT"
  local run_frames = opts.run_frames or 6
  local spin_frames = opts.spin_frames or 20
  local label = opts.label or "kb"
  local i
  for i = 1, run_frames do
    M.hold(session, 1, {dir, "B"}, label .. "_kb_run")
    if opts.stop_room_id and M.num(session.state.room_id) == opts.stop_room_id then
      return session.state
    end
  end
  for i = 1, spin_frames do
    M.hold(session, 1, {dir, "B", "A"}, label .. "_kb_spin")
    if opts.stop_room_id and M.num(session.state.room_id) == opts.stop_room_id then
      return session.state
    end
    if opts.break_on_motion_clear and not M.is_knockback(session.state) then
      return session.state
    end
  end
  return session.state
end

function M.u16(buf, addr)
  if not buf then
    return 0
  end
  local lo = M.num(buf[addr] or buf[addr + 1])
  local hi = M.num(buf[addr + 1] or buf[addr + 2])
  if type(buf.read_u16) == "function" then
    return buf:read_u16(addr)
  end
  if type(buf.u16) == "function" then
    return buf:u16(addr)
  end
  return lo + hi * 256
end

function M.wram(session)
  local st = session.state
  if st and st.wram then
    return st.wram
  end
  if session.ram then
    return session.ram
  end
  if ram.snapshot then
    return ram.snapshot(session)
  end
  return st
end

-- Aliases used by the Kraid-path ports (red_stack / spazer / kraid).
local rooms = require("rooms")
for k, v in pairs(rooms) do
  if M[k] == nil then
    M[k] = v
  end
end

local sgeo = require("skills.geometry")
M.TRUE_GROUND = sgeo.TRUE_GROUND
M.STANDING = sgeo.STANDING_POSES
M.POSE_KNOCKBACK = sgeo.POSE_KNOCKBACK
M.ITEM_HI_JUMP = ram.HI_JUMP_MASK or 0x0100
M.ITEM_VARIA = ram.VARIA_MASK or 0x0001
M.SPAZER_BEAM_MASK = 0x0004
M.fmt_state = M.brief

function M.fmt_hex(n)
  return string.format("0x%04X", M.num(n))
end

function M.set(...)
  local s = {}
  local n = select("#", ...)
  local i
  for i = 1, n do
    s[select(i, ...)] = true
  end
  return s
end

function M.settle_hold(session, frames, reason)
  return M.hold(session, frames or 12, {}, reason or "settle")
end

function M.try_select_weapon(session, slot)
  pcall(M.select_weapon, session, slot)
end

function M.has_item(state, mask)
  return M.band(state.collected_items or 0, mask) == mask
end

function M.has_beam(state, mask)
  return M.band(state.collected_beams or 0, mask) ~= 0
end

function M.has_hi_jump(state)
  return M.band(state.collected_items or 0, M.ITEM_HI_JUMP) ~= 0
end

function M.has_varia(state)
  return M.band(state.collected_items or 0, M.ITEM_VARIA) ~= 0
end

function M.has_spazer(state)
  return M.has_beam(state, M.SPAZER_BEAM_MASK)
end

function M.break_rle_lag(session, reason, budget)
  reason = reason or "rle_lag"
  budget = budget or 40
  local i
  for i = 1, budget do
    local pose = M.num(session.state.pose)
    if pose ~= 137 and pose ~= 138 and pose ~= 164 then
      return
    end
    M.hold(session, 1, {"A"}, reason)
    M.hold(session, 2, {}, reason)
  end
end

local ok_ctrl, controller = pcall(require, "skills.controller")
if ok_ctrl and controller then
  M.consecutive_walljumps = controller.consecutive_walljumps
  M.walljump_once = controller.walljump_once
end

return M
