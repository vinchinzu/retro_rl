-- Tourian escape. Verified rooms: 0xDE4D then 0xDEDE.
-- No Lua hop between them. Do not step an unproven door.
-- Success is gs 38 or 39 only. This module does not write a leave file.

local M = {}
M.proven = false
M.ROOM_ESCAPE_1 = 0xDE4D
M.ROOM_ESCAPE_4 = 0xDEDE

local function num(v)
  return tonumber(v) or 0
end

function M.play_escape_to_credits(session)
  local st = session.state or {}
  local gs = num(st.game_state)
  if gs == 38 or gs == 39 then
    return st
  end
  error(string.format(
    "UNPROVEN: door escape_1_to_escape_4 0x%04X -> 0x%04X (gs=%d)",
    M.ROOM_ESCAPE_1, M.ROOM_ESCAPE_4, gs
  ))
end

return M
