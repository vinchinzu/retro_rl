-- Gravity Suit room toward Main Street. Not a proven leave.
-- M.proven stays false until a live lsnes leave exists.
-- Room ids: finish.rooms, else only the four verified literals below.

local ram = require("ram")
local tube = require("maridia.tube")
local main_street = require("maridia.main_street")

local M = {}

-- No live leave has crossed this segment. Do not set this true.
M.proven = false

local FALLBACK = {
  GRAVITY = 0xCE40,
  MAIN_STREET = 0xCFC9,
  CATERPILLAR = 0xA322,
  BUSINESS = 0xA7DE,
}

local KEYS = {
  GRAVITY = {"GRAVITY", "ROOM_GRAVITY"},
  MAIN_STREET = {"MAIN_STREET", "ROOM_MAIN_STREET"},
  CATERPILLAR = {"CATERPILLAR", "ROOM_CATERPILLAR"},
  BUSINESS = {"BUSINESS", "ROOM_BUSINESS"},
}

local function lookup(mod, names)
  if type(mod) ~= "table" then
    return nil
  end
  local i
  for i = 1, #names do
    local v = mod[names[i]]
    if type(v) == "number" then
      return v
    end
  end
  if type(mod.rooms) == "table" and mod.rooms ~= mod then
    return lookup(mod.rooms, names)
  end
  return nil
end

function M.load_rooms()
  local ok, mod = pcall(require, "finish.rooms")
  if not ok then
    mod = nil
  end
  local ids, name = {}, nil
  for name, keys in pairs(KEYS) do
    ids[name] = lookup(mod, keys) or FALLBACK[name]
  end
  return ids
end

local function num(v)
  return tonumber(v) or 0
end

local function has_gravity(state)
  if type(state.gravity) == "function" then
    local ok, bit_on = pcall(state.gravity, state)
    if ok then
      return bit_on and true or false
    end
  end
  return ram.band(num(state.collected_items), ram.GRAVITY_MASK or 0x0020) ~= 0
end

local function require_gravity_start(session, ids)
  if type(session) ~= "table" or type(session.state) ~= "table" then
    error("TimeoutError: gravity_to_main_street: no session state", 2)
  end
  local st = session.state
  local room = num(st.room_id)
  local gs = num(st.game_state)
  local items = num(st.collected_items)
  if room ~= ids.GRAVITY or gs ~= 8 or not has_gravity(st) then
    error(string.format(
      "TimeoutError: gravity_to_main_street: expected room 0x%04X gs=8 gravity bit 0x0020, got room=0x%04X gs=%s items=0x%04X frame=%s",
      ids.GRAVITY, room, tostring(st.game_state), items, tostring(session.frame)
    ), 2)
  end
end

function M.play_gravity_to_main_street(session)
  local ids = M.load_rooms()
  require_gravity_start(session, ids)
  local st = tube.leave_gravity(session, ids)
  if main_street.at(st, ids) then
    -- Seen in RAM only. No measured tube entry, so this still errors.
    main_street.play(session, ids)
  end
  tube.stop_unproven(session, ids)
end

return M
