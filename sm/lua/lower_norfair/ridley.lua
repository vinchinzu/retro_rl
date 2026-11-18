-- Ridley room sketch. Reads enemy0_hp from the session snapshot the way
-- combat/kraid.lua and combat/phantoon.lua do. Does not write HP, items,
-- beams, doors, pose, or xy.
-- sm/lua names Kraid at $D829 and Phantoon at $D82B. It does not name a
-- Ridley-dead bit. enemy0_hp == 0 is not that bit.

local rooms = require("finish.rooms")

local M = {}

M.proven = false
M.ROOM_RIDLEY = rooms.ROOM_RIDLEY
M.GS_FIGHT = 8
M.MAX_FIGHT_FRAMES = 30000
M.BODY_ZERO_GRACE = 1200
M.FIRE_PERIOD = 12
M.FIRE_HOLD = 6

function M.body_hp(state)
  return tonumber(state and state.enemy0_hp) or 0
end

function M.in_fight(state)
  if type(state) ~= "table" then
    return false
  end
  return (tonumber(state.room_id) or 0) == M.ROOM_RIDLEY
    and (tonumber(state.game_state) or 0) == M.GS_FIGHT
end

-- combat/init.lua reserves combat.ridley. Nothing there currently exports
-- a Ridley-dead reader. Do not borrow another boss byte.
function M.find_dead_reader()
  local ok, mod = pcall(require, "combat.ridley")
  if not ok or type(mod) ~= "table" then
    return nil
  end
  if type(mod.ridley_defeated) == "function" then
    return mod.ridley_defeated
  end
  if type(mod.ridley_dead) == "function" then
    return mod.ridley_dead
  end
  return nil
end

function M.require_seat(session)
  local st = session and session.state or {}
  local room = tonumber(st.room_id) or 0
  local gs = tonumber(st.game_state) or 0
  if room ~= M.ROOM_RIDLEY or gs ~= M.GS_FIGHT then
    error(string.format(
      "Ridley fight expected room 0x%04X gs=%d, got 0x%04X gs=%d",
      M.ROOM_RIDLEY, M.GS_FIGHT, room, gs
    ))
  end
end

function M.fight_action(state, frame_index)
  local names = {}
  local sx = tonumber(state.samus_x) or 0
  local ex = tonumber(state.enemy0_x) or 0
  if sx <= 60000 and ex > 0 and ex <= 60000 then
    if sx > ex then
      names[#names + 1] = "LEFT"
    elseif sx < ex then
      names[#names + 1] = "RIGHT"
    end
  end
  if frame_index % M.FIRE_PERIOD < M.FIRE_HOLD then
    names[#names + 1] = "X"
  end
  return names
end

local function hold(session, names, reason)
  if type(session.hold) == "function" then
    session:hold(1, names, reason)
    return
  end
  session:step(names, reason)
end

function M.play_fight(session, opts)
  opts = opts or {}
  M.require_seat(session)
  local max_frames = tonumber(opts.max_fight_frames) or M.MAX_FIGHT_FRAMES
  if max_frames < 1 then
    max_frames = 1
  end
  local grace = tonumber(opts.boss_bit_grace_frames) or M.BODY_ZERO_GRACE
  local reader = M.find_dead_reader()
  local prev = M.body_hp(session.state)
  local body_zero = prev == 0
  local zero_frame = body_zero and (tonumber(session.frame) or 0) or nil
  local index
  for index = 0, max_frames - 1 do
    local st = session.state or {}
    if not M.in_fight(st) or (tonumber(st.health) or 1) == 0 then
      break
    end
    local hp = M.body_hp(st)
    if body_zero == false and hp == 0 and prev > 0 then
      body_zero = true
      zero_frame = tonumber(session.frame) or 0
    end
    prev = hp
    if reader ~= nil and reader(session) then
      break
    end
    if body_zero then
      local now = tonumber(session.frame) or 0
      if zero_frame ~= nil and (now - zero_frame) > grace then
        break
      end
      hold(session, {}, "ridley_body_zero_unproven")
    else
      hold(session, M.fight_action(st, index), "fight_ridley")
    end
  end
  if reader == nil or not reader(session) then
    error("UNPROVEN: ridley dead bit")
  end
  -- A reader flipping true is still not a route proof. Do not return success.
  error("UNPROVEN: ridley dead bit")
end

return M
