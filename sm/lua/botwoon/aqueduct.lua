-- Main Street through the aqueduct to Botwoon. Not a measured Clean path.
-- Room headers come from finish.rooms when that file exists. Otherwise only
-- Main Street 0xCFC9, Botwoon 0xD95E, and Crab Hole 0xD21C are known.
-- Crab Hole is not on this spine. Missing connectors are not guessed.

local M = {}

M.proven = false
M.measured = nil

local MAIN_STREET = 0xCFC9
local BOTWOON = 0xD95E
local CRAB_HOLE = 0xD21C

-- Door order from Main Street into the aqueduct, then Botwoon.
-- Names only. Hex for the rooms between the verified ends is not stored here.
local SPINE = {
  "Main Street",
  "Mt. Everest",
  "Crab Shaft",
  "Aqueduct",
  "Botwoon Hallway",
  "Botwoon's Room",
}

local FALLBACK = {
  MAIN_STREET = MAIN_STREET,
  ROOM_MAIN_STREET = MAIN_STREET,
  ["Main Street"] = MAIN_STREET,
  BOTWOON = BOTWOON,
  ROOM_BOTWOON = BOTWOON,
  ["Botwoon's Room"] = BOTWOON,
  CRAB_HOLE = CRAB_HOLE,
  ROOM_CRAB_HOLE = CRAB_HOLE,
  ["Crab Hole"] = CRAB_HOLE,
}

local function module_root()
  local src = debug.getinfo(1, "S").source or ""
  src = string.gsub(src, "^@", "")
  return string.match(src, "^(.*)/botwoon/[^/]+$")
end

local function load_catalog()
  local root = module_root()
  if root then
    local path = root .. "/finish/rooms.lua"
    local f = io.open(path, "r")
    if f ~= nil then
      f:close()
      local ok, rooms = pcall(require, "finish.rooms")
      if (not ok) or type(rooms) ~= "table" then
        rooms = dofile(path)
      end
      if type(rooms) ~= "table" then
        error("UNPROVEN: finish.rooms did not return a room table")
      end
      return rooms
    end
  end
  return FALLBACK
end

local function key_of(name)
  local key = string.upper(name)
  key = string.gsub(key, "'", "")
  key = string.gsub(key, "[^%w]+", "_")
  key = string.gsub(key, "^_+", "")
  key = string.gsub(key, "_+$", "")
  return key
end

local function lookup(rooms, name)
  if type(rooms) ~= "table" then
    return nil
  end
  if type(rooms[name]) == "number" then
    return rooms[name]
  end
  local nested = rooms.by_name or rooms.ids or rooms.rooms
  if type(nested) == "table" and type(nested[name]) == "number" then
    return nested[name]
  end
  local key = key_of(name)
  local keys = { key, "ROOM_" .. key }
  if name == "Main Street" then
    keys[#keys + 1] = "MAIN_STREET"
  elseif name == "Botwoon's Room" then
    keys[#keys + 1] = "BOTWOON"
    keys[#keys + 1] = "ROOM_BOTWOON"
  elseif name == "Crab Hole" then
    keys[#keys + 1] = "CRAB_HOLE"
    keys[#keys + 1] = "ROOM_CRAB_HOLE"
  end
  local i
  for i = 1, #keys do
    local v = rooms[keys[i]]
    if type(v) == "number" then
      return v
    end
  end
  return nil
end

local function measured_segment(src_name, dst_name)
  if type(M.measured) ~= "table" then
    return nil
  end
  local seg = M.measured[src_name .. "->" .. dst_name]
  if type(seg) ~= "table" or #seg == 0 then
    return nil
  end
  return seg
end

local function play_segment(session, seg)
  local i
  for i = 1, #seg do
    local row = seg[i]
    local names = row.names or row[1] or {}
    local frames = tonumber(row.frames or row[2]) or 1
    if type(names) == "string" then
      names = { names }
    end
    if frames < 1 then
      frames = 1
    end
    local f
    for f = 1, frames do
      session:step(names, row.reason or "aqueduct")
    end
  end
end

function M.play_main_street_to_botwoon(session)
  local rooms = load_catalog()
  local ids = {}
  local i
  for i = 1, #SPINE do
    local name = SPINE[i]
    local id = lookup(rooms, name)
    if id == nil then
      error("UNPROVEN: need room id for " .. name)
    end
    ids[i] = id
  end
  for i = 1, #SPINE - 1 do
    if measured_segment(SPINE[i], SPINE[i + 1]) == nil then
      error("UNPROVEN: measured path " .. SPINE[i] .. " to " .. SPINE[i + 1])
    end
  end
  if type(session) ~= "table" or type(session.state) ~= "table" or type(session.step) ~= "function" then
    error("UNPROVEN: measured path Main Street to Botwoon")
  end
  local here = tonumber(session.state.room_id) or 0
  local gs = tonumber(session.state.game_state) or 0
  if here ~= ids[1] or gs ~= 8 then
    error("UNPROVEN: measured path Main Street to Botwoon")
  end
  for i = 1, #SPINE - 1 do
    if here ~= ids[i] then
      error("UNPROVEN: measured path " .. SPINE[i] .. " to " .. SPINE[i + 1])
    end
    play_segment(session, measured_segment(SPINE[i], SPINE[i + 1]))
    here = tonumber(session.state.room_id) or 0
    if here ~= ids[i + 1] then
      error("UNPROVEN: measured path " .. SPINE[i] .. " to " .. SPINE[i + 1])
    end
  end
  if here ~= ids[#ids] then
    error("UNPROVEN: measured path Main Street to Botwoon")
  end
end

return M
