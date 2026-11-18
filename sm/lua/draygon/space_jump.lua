-- Space Jump collect in 0xD9AA. The chozo must flip collected_items
-- bit 0x0200 on contact. Never write the bit. Not a measured approach.

local ram = require("ram")

local M = {}

M.proven = false
M.ROOM = 0xD9AA
M.SPACE_JUMP_MASK = 0x0200
M.GRAVITY_MASK = 0x0020
M.GRAPPLE_MASK = 0x4000
-- Gravity + Hi-Jump + Space Jump + Grapple and the earlier majors.
-- Read-only note from the post-collect states. Never poked.
M.POST_SJ_ITEMS = 0x7325
M.GS_ORDINARY = 8

function M.has_space_jump(state)
  local items = tonumber(state and state.collected_items) or 0
  return ram.band(items, M.SPACE_JUMP_MASK) ~= 0
end

-- True only when the mask goes from off to on. A write would not pass
-- through here; callers compare two reads.
function M.bit_flipped(before_items, state)
  local before = ram.band(tonumber(before_items) or 0, M.SPACE_JUMP_MASK)
  if before ~= 0 then
    return false
  end
  return M.has_space_jump(state)
end

function M.succeeded(state)
  if state == nil then
    return false
  end
  local rid = tonumber(state.room_id) or 0
  local gs = tonumber(state.game_state) or 0
  return rid == M.ROOM and gs == M.GS_ORDINARY and M.has_space_jump(state)
end

local function stopped(rid)
  error(string.format("UNPROVEN: space jump collect stopped in 0x%04X", rid))
end

function M.play_collect(session)
  local state = session and session.state
  if state == nil then
    stopped(0)
  end
  local rid = tonumber(state.room_id) or 0
  if rid ~= M.ROOM then
    stopped(rid)
  end
  local before = tonumber(state.collected_items) or 0
  if M.succeeded(state) then
    return state
  end
  -- Fanfare: bit already flipped, gameplay not ordinary yet. Idle only.
  if M.has_space_jump(state) and session.wait_until then
    local ok = pcall(function()
      session:wait_until(function(s)
        return M.succeeded(s)
      end, 600, "space_jump_fanfare")
    end)
    if ok and M.succeeded(session.state) then
      return session.state
    end
    stopped(tonumber(session.state.room_id) or rid)
  end
  -- No measured door-to-chozo tape. Do not swim, shoot, or poke 0x0200.
  if not M.bit_flipped(before, state) then
    stopped(rid)
  end
  if M.succeeded(session.state) then
    return session.state
  end
  stopped(rid)
end

return M
