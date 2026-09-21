-- SNES-12 names <-> lsnes joyset table.
-- Order matches lsnes P1 and the gym: B Y select start up down left right A X L R.

local input_mod = {}

input_mod.ORDER = {
  "B", "Y", "select", "start", "up", "down", "left", "right", "A", "X", "L", "R",
}

local ORDER = input_mod.ORDER

local NAME_MAP = {
  LEFT = "left",
  RIGHT = "right",
  UP = "up",
  DOWN = "down",
  START = "start",
  SELECT = "select",
  left = "left",
  right = "right",
  up = "up",
  down = "down",
  start = "start",
  select = "select",
  B = "B",
  Y = "Y",
  A = "A",
  X = "X",
  L = "L",
  R = "R",
  b = "B",
  y = "Y",
  a = "A",
  x = "X",
}

function input_mod.normalize_name(name)
  if name == nil then
    return nil
  end
  local s = tostring(name)
  return NAME_MAP[s] or s
end

local function empty_joyset()
  local t = {}
  for i = 1, #ORDER do
    t[ORDER[i]] = false
  end
  return t
end

local function press(t, name)
  local k = input_mod.normalize_name(name)
  if k and t[k] ~= nil then
    t[k] = true
  end
end

-- names: array of Python/lsnes button names, or a boolean/set map.
function input_mod.joyset_table(names)
  local t = empty_joyset()
  if names == nil then
    return t
  end
  if type(names) ~= "table" then
    press(t, names)
    return t
  end
  if names[1] ~= nil then
    for i = 1, #names do
      press(t, names[i])
    end
    return t
  end
  for k, v in pairs(names) do
    if type(k) == "number" then
      if v then
        press(t, v)
      end
    elseif v then
      press(t, k)
    end
  end
  return t
end

-- Seed tapes store 12 ints in ORDER. Returns a names array.
function input_mod.from_bits(arr12)
  local names = {}
  if not arr12 then
    return names
  end
  for i = 1, 12 do
    local v = tonumber(arr12[i]) or 0
    if v ~= 0 then
      names[#names + 1] = ORDER[i]
    end
  end
  return names
end

function input_mod.is_bits12(item)
  if type(item) ~= "table" or #item ~= 12 then
    return false
  end
  for i = 1, 12 do
    local v = tonumber(item[i])
    if v ~= 0 and v ~= 1 then
      return false
    end
  end
  return true
end

function input_mod.names_of(item)
  if item == nil then
    return {}
  end
  if type(item) ~= "table" then
    return { item }
  end
  if input_mod.is_bits12(item) then
    return input_mod.from_bits(item)
  end
  if item[1] ~= nil then
    return item
  end
  local names = {}
  for k, v in pairs(item) do
    if type(k) == "number" then
      if v then
        names[#names + 1] = v
      end
    elseif v then
      names[#names + 1] = k
    end
  end
  return names
end

return input_mod
