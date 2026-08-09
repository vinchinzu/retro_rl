-- Phase 1 oracle: prove a BK2 plays under BizHawk BSNES with real progression.
--
-- Reads Super Metroid WRAM (same offsets as harness ram.py), logs room/item
-- changes, writes proof under out_dir, exits on GREEN milestone or movie end.
--
-- out_dir resolution (first hit wins):
--   1) oracle_out_dir.txt next to this script (written by run_verify_100.sh)
--   2) oracle_out_dir.txt in cwd
--   3) userdata out_dir
--   4) "."

local ADDR_ROOM_ID = 0x079B
local ADDR_AREA_INDEX = 0x079F
local ADDR_GAME_STATE = 0x0998
local ADDR_COLLECTED_ITEMS = 0x09A4
local ADDR_COLLECTED_BEAMS = 0x09A8
local ADDR_HEALTH = 0x09C2
local ADDR_MAX_HEALTH = 0x09C4
local ADDR_MAX_MISSILES = 0x09C8
local ADDR_SAMUS_POSE = 0x0A1C
local ADDR_SAMUS_X = 0x0AF6
local ADDR_SAMUS_Y = 0x0AFA

local MORPH_MASK = 0x0004
local BOMBS_MASK = 0x1000
-- Ignore RAM until this frame (boot noise / pre-init WRAM).
local TRUST_RAM_AFTER = 60

local AREA_NAMES = {
  [0] = "Crateria",
  [1] = "Brinstar",
  [2] = "Norfair",
  [3] = "Wrecked Ship",
  [4] = "Maridia",
  [5] = "Tourian",
  [6] = "Ceres",
}

local function band(a, b)
  if bit and bit.band then
    return bit.band(a, b)
  end
  a = math.floor(a) % 65536
  b = math.floor(b) % 65536
  local res, bitval = 0, 1
  for _ = 0, 15 do
    if (a % 2 == 1) and (b % 2 == 1) then
      res = res + bitval
    end
    a = math.floor(a / 2)
    b = math.floor(b / 2)
    bitval = bitval * 2
  end
  return res
end

local function dirname(path)
  if not path then
    return nil
  end
  local d = path:match("^(.*)/[^/]+$")
  return d
end

local function read_out_dir_file(path)
  local f = io.open(path, "r")
  if not f then
    return nil
  end
  local line = f:read("*l")
  f:close()
  if line and line ~= "" then
    return line
  end
  return nil
end

local function ud(key, default)
  if userdata and userdata.getdata then
    local ok, data = pcall(userdata.getdata)
    if ok and type(data) == "table" and data[key] ~= nil then
      return tostring(data[key])
    end
  end
  return default
end

local script_path = nil
if debug and debug.getinfo then
  local info = debug.getinfo(1, "S")
  if info and info.source then
    script_path = info.source:gsub("^@", "")
  end
end

local out_dir = nil
if script_path then
  local sd = dirname(script_path)
  if sd then
    out_dir = read_out_dir_file(sd .. "/oracle_out_dir.txt")
  end
end
if not out_dir then
  out_dir = read_out_dir_file("oracle_out_dir.txt")
end
if not out_dir then
  out_dir = ud("out_dir", nil)
end
if not out_dir or out_dir == "" then
  out_dir = "."
end

local early_exit = ud("early_exit", "1") ~= "0"
local max_frames = tonumber(ud("max_frames", "230000")) or 230000
-- Sibling flags file (written by run_verify_100.sh next to this script).
local flags_path = nil
if script_path then
  local sd = dirname(script_path)
  if sd then
    flags_path = sd .. "/oracle_flags.txt"
  end
end
if not flags_path then
  flags_path = "oracle_flags.txt"
end
do
  local ff = io.open(flags_path, "r")
  if ff then
    for line in ff:lines() do
      local k, v = line:match("^([%w_]+)=(.*)$")
      if k == "early_exit" then
        early_exit = (v ~= "0")
      elseif k == "max_frames" then
        max_frames = tonumber(v) or max_frames
      elseif k == "out_dir" and v and v ~= "" then
        out_dir = v
      end
    end
    ff:close()
  end
end

local log_path = out_dir .. "/verify_log.txt"
local proof_path = out_dir .. "/verify_proof.json"
local events_path = out_dir .. "/verify_events.jsonl"

local log_fh = io.open(log_path, "w")
local events_fh = io.open(events_path, "w")

local function log(msg)
  local line = string.format("[%d] %s", emu.framecount(), msg)
  print(line)
  if log_fh then
    log_fh:write(line .. "\n")
    log_fh:flush()
  end
end

local function emit(kind, payload)
  if not events_fh then
    return
  end
  local parts = { string.format('"frame":%d', emu.framecount()), string.format('"kind":"%s"', kind) }
  if payload then
    for k, v in pairs(payload) do
      if type(v) == "number" then
        table.insert(parts, string.format('"%s":%s', k, tostring(v)))
      else
        table.insert(parts, string.format('"%s":"%s"', k, tostring(v):gsub('"', "'")))
      end
    end
  end
  events_fh:write("{" .. table.concat(parts, ",") .. "}\n")
  events_fh:flush()
end

local function ru16(addr)
  return mainmemory.read_u16_le(addr)
end

local function ru8(addr)
  return mainmemory.read_u8(addr)
end

local last_room = -1
local last_items = -1
local last_beams = -1
local unique_rooms = {}
local unique_room_count = 0
local unique_areas = {}
local unique_area_count = 0
local zebes_rooms = 0
local ceres_rooms = 0
local first_control_frame = nil
local morph_frame = nil
local bombs_frame = nil
local green = false
local finished = false
local core_name = "(unknown)"
local started = false

local function snapshot()
  return {
    room_id = ru16(ADDR_ROOM_ID),
    area = ru8(ADDR_AREA_INDEX),
    game_state = ru16(ADDR_GAME_STATE),
    items = ru16(ADDR_COLLECTED_ITEMS),
    beams = ru16(ADDR_COLLECTED_BEAMS),
    energy = ru16(ADDR_HEALTH),
    max_energy = ru16(ADDR_MAX_HEALTH),
    max_missiles = ru16(ADDR_MAX_MISSILES),
    pose = ru16(ADDR_SAMUS_POSE),
    x = ru16(ADDR_SAMUS_X),
    y = ru16(ADDR_SAMUS_Y),
  }
end

local function write_proof(status, reason)
  local s = snapshot()
  local rooms_list = {}
  for rid, fr in pairs(unique_rooms) do
    table.insert(rooms_list, string.format('{"room_id":%d,"first_frame":%d}', rid, fr))
  end
  table.sort(rooms_list)
  local areas_list = {}
  for a, fr in pairs(unique_areas) do
    local name = AREA_NAMES[a] or ("area_" .. tostring(a))
    table.insert(areas_list, string.format('{"area":%d,"name":"%s","first_frame":%d}', a, name, fr))
  end
  local fh = io.open(proof_path, "w")
  if not fh then
    log("ERROR: cannot write proof to " .. proof_path)
    return
  end
  fh:write("{\n")
  fh:write(string.format('  "status": "%s",\n', status))
  fh:write(string.format('  "reason": "%s",\n', reason:gsub('"', "'")))
  fh:write(string.format('  "frame": %d,\n', emu.framecount()))
  fh:write(string.format('  "movie_loaded": %s,\n', movie.isloaded() and "true" or "false"))
  if movie.isloaded() and movie.length then
    local ok, len = pcall(movie.length)
    if ok then
      fh:write(string.format('  "movie_length": %d,\n', len))
    end
  end
  fh:write(string.format('  "core_hint": "%s",\n', core_name))
  fh:write(string.format('  "out_dir": "%s",\n', out_dir:gsub('"', "'")))
  fh:write(string.format('  "unique_rooms": %d,\n', unique_room_count))
  fh:write(string.format('  "zebes_rooms": %d,\n', zebes_rooms))
  fh:write(string.format('  "ceres_rooms": %d,\n', ceres_rooms))
  fh:write(string.format('  "unique_areas": %d,\n', unique_area_count))
  fh:write(string.format('  "first_control_frame": %s,\n', first_control_frame and tostring(first_control_frame) or "null"))
  fh:write(string.format('  "morph_frame": %s,\n', morph_frame and tostring(morph_frame) or "null"))
  fh:write(string.format('  "bombs_frame": %s,\n', bombs_frame and tostring(bombs_frame) or "null"))
  fh:write(string.format('  "final_room_id": %d,\n', s.room_id))
  fh:write(string.format('  "final_area": %d,\n', s.area))
  fh:write(string.format('  "final_items": %d,\n', s.items))
  fh:write(string.format('  "final_beams": %d,\n', s.beams))
  fh:write(string.format('  "final_energy": %d,\n', s.energy))
  fh:write(string.format('  "final_max_missiles": %d,\n', s.max_missiles))
  fh:write(string.format('  "rooms": [%s],\n', table.concat(rooms_list, ",")))
  fh:write(string.format('  "areas": [%s]\n', table.concat(areas_list, ",")))
  fh:write("}\n")
  fh:close()
  log("proof written: " .. proof_path .. " status=" .. status)
end

local function try_speedup()
  -- Speed only — do NOT frameskip (can desync movies).
  if client.speedmode then
    pcall(function()
      client.speedmode(6400)
    end)
  end
  if client.unpause then
    pcall(function()
      client.unpause()
    end)
  end
end

local function is_green()
  -- Past Ceres with inventory progress (not Ceres-only thrash / garbage).
  local has_items = morph_frame ~= nil or (last_items > 0)
  local left_ceres = zebes_rooms >= 2 and ceres_rooms >= 1
  -- Also accept: morph + any non-Ceres area room
  if morph_frame ~= nil and zebes_rooms >= 1 then
    return true
  end
  return has_items and left_ceres
end

local function finish(status, reason)
  if finished then
    return
  end
  finished = true
  write_proof(status, reason)
  if log_fh then
    log_fh:close()
    log_fh = nil
  end
  if events_fh then
    events_fh:close()
    events_fh = nil
  end
  if client.exit then
    client.exit()
  end
end

log("verify_movie_sync.lua start out_dir=" .. out_dir)
log("early_exit=" .. tostring(early_exit) .. " max_frames=" .. tostring(max_frames))
log("movie.isloaded=" .. tostring(movie.isloaded()))
if movie.isloaded() and movie.filename then
  local ok, name = pcall(movie.filename)
  if ok then
    log("movie.filename=" .. tostring(name))
  end
end
if movie.isloaded() and movie.getheader then
  local ok, hdr = pcall(movie.getheader)
  if ok and type(hdr) == "table" then
    for k, v in pairs(hdr) do
      log("movie.header " .. tostring(k) .. "=" .. tostring(v))
      if tostring(k):lower():find("core") then
        core_name = tostring(v)
      end
    end
  end
end

try_speedup()
emit("start", { out_dir = out_dir })
started = true

local function on_frame()
  if finished then
    return
  end

  local f = emu.framecount()
  if f > max_frames then
    finish(green and "GREEN" or "PARTIAL", "max_frames reached")
    return
  end

  if movie.isloaded() then
    local mode = movie.mode and movie.mode() or ""
    local len_ok, len = pcall(function()
      return movie.length()
    end)
    if len_ok and f >= len and f > 0 then
      finish(green and "GREEN" or "PARTIAL", "movie end frame=" .. tostring(f))
      return
    end
    if (mode == "FINISHED" or mode == "INACTIVE") and f > 100 then
      finish(green and "GREEN" or "PARTIAL", "movie mode=" .. tostring(mode))
      return
    end
  end

  if f < TRUST_RAM_AFTER then
    if f > 0 and f % 5000 == 0 then
      log("boot wait frame=" .. f)
    end
    return
  end

  local s = snapshot()
  -- Plausible area is 0..6; game_state ordinary gameplay is 8 (among others).
  local area_ok = s.area >= 0 and s.area <= 6
  local room_ok = s.room_id ~= 0 and s.room_id ~= 0xFFFF

  if first_control_frame == nil and s.game_state == 8 and room_ok and area_ok then
    first_control_frame = f
    log(string.format(
      "first_control room=0x%04X area=%d items=0x%04X energy=%d",
      s.room_id, s.area, s.items, s.energy
    ))
    emit("control", { room_id = s.room_id, area = s.area, items = s.items })
    -- seed last_* without treating as gains
    last_items = s.items
    last_beams = s.beams
    last_room = s.room_id
  end

  if not area_ok then
    if f % 5000 == 0 then
      log(string.format("heartbeat (untrusted area) area=%d room=0x%04X", s.area, s.room_id))
    end
    return
  end

  if room_ok and s.room_id ~= last_room then
    local prev = last_room
    last_room = s.room_id
    if unique_rooms[s.room_id] == nil then
      unique_rooms[s.room_id] = f
      unique_room_count = unique_room_count + 1
      if s.area == 6 then
        ceres_rooms = ceres_rooms + 1
      else
        zebes_rooms = zebes_rooms + 1
      end
    end
    if unique_areas[s.area] == nil then
      unique_areas[s.area] = f
      unique_area_count = unique_area_count + 1
    end
    local aname = AREA_NAMES[s.area] or ("?" .. tostring(s.area))
    log(string.format(
      "room_enter room=0x%04X area=%s(%d) items=0x%04X energy=%d pose=%d xy=(%d,%d) unique=%d zebes=%d ceres=%d",
      s.room_id, aname, s.area, s.items, s.energy, s.pose, s.x, s.y,
      unique_room_count, zebes_rooms, ceres_rooms
    ))
    emit("room_enter", {
      room_id = s.room_id,
      area = s.area,
      items = s.items,
      energy = s.energy,
    })
  end

  if last_items >= 0 and s.items ~= last_items then
    local prev = last_items
    last_items = s.items
    log(string.format("items 0x%04X -> 0x%04X", prev, s.items))
    emit("item_gain", { items = s.items, prev = prev })
    if morph_frame == nil and band(s.items, MORPH_MASK) ~= 0 then
      morph_frame = f
      log("MORPH gained at frame " .. f)
      emit("morph", { frame = f, items = s.items })
    end
    if bombs_frame == nil and band(s.items, BOMBS_MASK) ~= 0 then
      bombs_frame = f
      log("BOMBS gained at frame " .. f)
      emit("bombs", { frame = f, items = s.items })
    end
  elseif last_items < 0 and first_control_frame ~= nil then
    last_items = s.items
  end

  if last_beams >= 0 and s.beams ~= last_beams then
    local prev = last_beams
    last_beams = s.beams
    log(string.format("beams 0x%04X -> 0x%04X", prev, s.beams))
    emit("beam_gain", { beams = s.beams, prev = prev })
  elseif last_beams < 0 and first_control_frame ~= nil then
    last_beams = s.beams
  end

  if not green and is_green() then
    green = true
    log("GREEN milestone: items + multi-room progression past Ceres thrash class")
    emit("green", {
      items = last_items,
      zebes_rooms = zebes_rooms,
      ceres_rooms = ceres_rooms,
      unique_rooms = unique_room_count,
      morph_frame = morph_frame or -1,
    })
    write_proof("GREEN", "milestone: items and zebes rooms")
    if early_exit then
      finish("GREEN", "early exit after milestone")
      return
    end
  end

  if f > 0 and f % 5000 == 0 then
    log(string.format(
      "heartbeat rooms=%d zebes=%d ceres=%d items=0x%04X area=%d room=0x%04X gs=%d",
      unique_room_count, zebes_rooms, ceres_rooms, s.items, s.area, s.room_id, s.game_state
    ))
  end
end

event.onframeend(on_frame)

event.onexit(function()
  -- Only write PARTIAL if we actually ran and never finished cleanly.
  if finished or not started then
    return
  end
  if emu.framecount() < 2 then
    return
  end
  write_proof(green and "GREEN" or "PARTIAL", "lua exit")
end)
