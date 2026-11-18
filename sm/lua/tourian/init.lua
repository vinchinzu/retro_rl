-- Ridley → credits. Verified ids only. Not a live proof.
-- ram.phase_for_game_state maps gs 38 and 39 to "ending_or_credits".
-- The first door with no Lua hop errors UNPROVEN. No energy/item/boss writes.

local mb = require("tourian.mother_brain")
local escape = require("tourian.escape")

local M = {}
M.proven = false

local VERIFIED = {
  ROOM_RIDLEY = 0xB32E,
  ROOM_STATUES = 0xA66A,
  ROOM_TOURIAN_FIRST = 0xDAAE,
  ROOM_MOTHER_BRAIN = 0xDD58,
  ROOM_ESCAPE_1 = 0xDE4D,
  ROOM_ESCAPE_4 = 0xDEDE,
}

local function finish_rooms_file()
  local src = debug.getinfo(1, "S").source:gsub("^@", "")
  local dir = src:match("^(.*)/") or src:match("^(.*)\\") or "."
  return dir .. "/../finish/rooms.lua"
end

local function load_finish_rooms()
  local path = finish_rooms_file()
  local f = io.open(path, "r")
  if not f then
    return nil
  end
  f:close()
  return require("finish.rooms")
end

M.finish_rooms = load_finish_rooms()

local function room_id(key)
  local expect = VERIFIED[key]
  local got = M.finish_rooms and M.finish_rooms[key]
  if got == nil then
    return expect
  end
  got = tonumber(got) or 0
  if got ~= expect then
    error(string.format("finish.rooms.%s mismatch (got 0x%04X)", key, got))
  end
  return expect
end

M.ROOM_RIDLEY = room_id("ROOM_RIDLEY")
M.ROOM_STATUES = room_id("ROOM_STATUES")
M.ROOM_TOURIAN_FIRST = room_id("ROOM_TOURIAN_FIRST")
M.ROOM_MOTHER_BRAIN = room_id("ROOM_MOTHER_BRAIN")
M.ROOM_ESCAPE_1 = room_id("ROOM_ESCAPE_1")
M.ROOM_ESCAPE_4 = room_id("ROOM_ESCAPE_4")

-- A hop is a Lua function. Room-id constants are not a door.
local function door_hop(src, dst)
  local rooms = M.finish_rooms
  if type(rooms) ~= "table" then
    return nil
  end
  if type(rooms.door_hop) == "function" then
    local hop = rooms.door_hop(src, dst)
    if type(hop) == "function" then
      return hop
    end
  end
  local hops = rooms.DOOR_HOPS
  if type(hops) == "table" and type(hops[src]) == "table" then
    local hop = hops[src][dst]
    if type(hop) == "function" then
      return hop
    end
  end
  return nil
end

local function cross(session, src, dst, name)
  local hop = door_hop(src, dst)
  if type(hop) ~= "function" then
    error(string.format("UNPROVEN: door %s 0x%04X -> 0x%04X", name, src, dst))
  end
  return hop(session)
end

local function credits(session)
  local gs = tonumber(session.state and session.state.game_state) or 0
  return gs == 38 or gs == 39
end

function M.play_ridley_to_credits(session)
  if credits(session) then
    return session.state
  end
  -- Existing movement Lua stops at Gravity 0xCE40. These edges have no hop.
  cross(session, M.ROOM_RIDLEY, M.ROOM_STATUES, "ridley_to_statues")
  cross(session, M.ROOM_STATUES, M.ROOM_TOURIAN_FIRST, "statues_to_tourian_first")
  cross(session, M.ROOM_TOURIAN_FIRST, M.ROOM_MOTHER_BRAIN, "tourian_first_to_mother_brain")
  mb.play_mother_brain(session)
  if credits(session) then
    return session.state
  end
  cross(session, M.ROOM_MOTHER_BRAIN, M.ROOM_ESCAPE_1, "mother_brain_to_escape_1")
  escape.play_escape_to_credits(session)
  if not credits(session) then
    error(string.format(
      "UNPROVEN: credits require gs 38 or 39, got %s",
      tostring(session.state and session.state.game_state)
    ))
  end
  return session.state
end

return M
