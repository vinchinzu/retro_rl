-- Mother Brain room 0xDD58. Shoot X only while room is 0xDD58 and gs==8.
-- Bounded. Reads enemy HP. Does not write energy, ammo, items, bosses, doors, pose, or xy.
-- Leaving the room is not credits. Credits are gs 38 or 39 only.

local M = {}
M.proven = false
M.ROOM_MOTHER_BRAIN = 0xDD58
M.GS_ORDINARY = 8
M.GS_CREDITS = { [38] = true, [39] = true }
M.MAX_FRAMES = 30000

local function num(v)
  return tonumber(v) or 0
end

local function snap(session)
  if session.refresh then
    session:refresh()
  end
  return session.state or {}
end

local function credits(st)
  return M.GS_CREDITS[num(st.game_state)] == true
end

function M.play_mother_brain(session)
  local st = snap(session)
  if credits(st) then
    return st
  end
  if num(st.room_id) ~= M.ROOM_MOTHER_BRAIN or num(st.game_state) ~= M.GS_ORDINARY then
    error(string.format(
      "UNPROVEN: mother brain shoot requires room 0xDD58 gs=8, got room 0x%04X gs=%d",
      num(st.room_id), num(st.game_state)
    ))
  end

  local i
  for i = 0, M.MAX_FRAMES - 1 do
    st = snap(session)
    if credits(st) then
      return st
    end
    local gs = num(st.game_state)
    if gs == 26 or gs == 36 or num(st.health) == 0 then
      error(string.format("mother brain: dead gs=%d health=%d", gs, num(st.health)))
    end
    if num(st.room_id) ~= M.ROOM_MOTHER_BRAIN or gs ~= M.GS_ORDINARY then
      return st
    end
    local names = {}
    if num(st.enemy0_hp) > 0 and (i % 3) == 0 then
      names = { "X" }
    end
    session:step(names, "mother_brain")
  end

  error("UNPROVEN: mother brain loop hit " .. tostring(M.MAX_FRAMES) .. " frames without gs 38 or 39")
end

return M
