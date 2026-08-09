-- Long-run movie counter: wait out SM intro (~3–5 min / ~11–18k frames)
-- and only trust WRAM when area∈[0,6] and room looks like a real SM pointer.
--
-- GREEN early-exit: Landing Site (0x91F8) or morph bit, after first Ceres elev.
-- Hard stop: MAX_FRAMES (default 60000 ≈ past Ceres+early Zebes on a synced TAS).

local OUT = nil
do
  local f = io.open("oracle_out_dir.txt", "r")
  if f then
    OUT = f:read("*l")
    f:close()
  end
end
if not OUT or OUT == "" then
  local info = debug.getinfo(1, "S")
  local src = info and info.source and info.source:gsub("^@", "") or nil
  if src then
    local dir = src:match("^(.*)/[^/]+$")
    if dir then
      local f = io.open(dir .. "/oracle_out_dir.txt", "r")
      if f then
        OUT = f:read("*l")
        f:close()
      end
    end
  end
end
OUT = OUT or "."

local MAX_FRAMES = 60000
local EARLY_EXIT = true
do
  local f = io.open("oracle_flags.txt", "r")
  if not f then
    local info = debug.getinfo(1, "S")
    local src = info and info.source and info.source:gsub("^@", "") or nil
    if src then
      local dir = src:match("^(.*)/[^/]+$")
      if dir then
        f = io.open(dir .. "/oracle_flags.txt", "r")
      end
    end
  end
  if f then
    for line in f:lines() do
      local k, v = line:match("^([%w_]+)=(.*)$")
      if k == "max_frames" then
        MAX_FRAMES = tonumber(v) or MAX_FRAMES
      elseif k == "early_exit" then
        EARLY_EXIT = (v ~= "0")
      elseif k == "out_dir" and v ~= "" then
        OUT = v
      end
    end
    f:close()
  end
end

local ROOM_ELEV = 0xDF45
local ROOM_LANDING = 0x91F8
local ROOM_RIDLEY = 0xE0B5
local ROOM_FLAT = 0xE06B
local MORPH = 0x0004

local log_path = OUT .. "/long_count.txt"
local proof_path = OUT .. "/long_proof.json"
local fh = io.open(log_path, "w")

local function band(a, b)
  if bit and bit.band then
    return bit.band(a, b)
  end
  a = math.floor(a) % 65536
  b = math.floor(b) % 65536
  local res, bv = 0, 1
  for _ = 0, 15 do
    if a % 2 == 1 and b % 2 == 1 then
      res = res + bv
    end
    a = math.floor(a / 2)
    b = math.floor(b / 2)
    bv = bv * 2
  end
  return res
end

local function L(msg)
  local line = string.format("[%d] %s", emu.framecount(), msg)
  print(line)
  if fh then
    fh:write(line .. "\n")
    fh:flush()
  end
end

-- Force WRAM domain (not System Bus).
pcall(function()
  memory.usememorydomain("WRAM")
end)

local function ru16(a)
  return memory.read_u16_le(a)
end
local function ru8(a)
  return memory.read_u8(a)
end

local function trusted(room, area, gs)
  if area < 0 or area > 6 then
    return false
  end
  if room == 0 or room == 0xFFFF then
    return false
  end
  -- SM room headers live in bank $8F; pointers we store are usually 0x9xxx–0xExxx.
  if room < 0x9000 or room > 0xEFFF then
    return false
  end
  -- During pure intro, gs is often 30+; require gameplay-ish for "room" claims.
  if gs ~= 8 and gs ~= 9 and gs ~= 11 and gs ~= 14 and gs ~= 15 and gs ~= 20 and gs ~= 21 and gs ~= 22 and gs ~= 23 and gs ~= 24 and gs ~= 25 and gs ~= 26 and gs ~= 27 and gs ~= 28 and gs ~= 31 and gs ~= 32 then
    -- still allow elevator first-control style gs=8 primarily; keep loose set
    if gs < 8 or gs > 40 then
      return false
    end
  end
  return true
end

local first_elev = nil
local first_landing = nil
local first_morph = nil
local first_ridley = nil
local last_room = -1
local unique = {}
local unique_n = 0
local done = false

local function write_proof(status, reason)
  local room = ru16(0x079B)
  local area = ru8(0x079F)
  local gs = ru16(0x0998)
  local items = ru16(0x09A4)
  local energy = ru16(0x09C2)
  local f = io.open(proof_path, "w")
  if not f then
    return
  end
  f:write("{\n")
  f:write(string.format('  "status": "%s",\n', status))
  f:write(string.format('  "reason": "%s",\n', reason:gsub('"', "'")))
  f:write(string.format('  "frame": %d,\n', emu.framecount()))
  f:write(string.format('  "first_elev_frame": %s,\n', first_elev and tostring(first_elev) or "null"))
  f:write(string.format('  "first_ridley_frame": %s,\n', first_ridley and tostring(first_ridley) or "null"))
  f:write(string.format('  "first_landing_frame": %s,\n', first_landing and tostring(first_landing) or "null"))
  f:write(string.format('  "first_morph_frame": %s,\n', first_morph and tostring(first_morph) or "null"))
  f:write(string.format('  "unique_trusted_rooms": %d,\n', unique_n))
  f:write(string.format('  "room": %d,\n', room))
  f:write(string.format('  "area": %d,\n', area))
  f:write(string.format('  "game_state": %d,\n', gs))
  f:write(string.format('  "items": %d,\n', items))
  f:write(string.format('  "energy": %d\n', energy))
  f:write("}\n")
  f:close()
  L("proof " .. status .. " " .. reason)
end

local function finish(status, reason)
  if done then
    return
  end
  done = true
  write_proof(status, reason)
  if fh then
    fh:close()
    fh = nil
  end
  if client.exit then
    client.exit()
  end
end

L("long_count start out=" .. OUT .. " max_frames=" .. MAX_FRAMES)
L("note: SM intro/menus ~3–5 min (~11–18k frames) before first_control @ Ceres elev 0xDF45")
if movie.isloaded() then
  local ok, len = pcall(movie.length)
  if ok then
    L("movie_length=" .. tostring(len))
  end
end

-- Speed without frameskip: unthrottled if available.
pcall(function()
  if client.speedmode then
    client.speedmode(400)
  end
end)
pcall(function()
  if client.unpause then
    client.unpause()
  end
end)

event.onframeend(function()
  if done then
    return
  end
  local fr = emu.framecount()
  if fr >= MAX_FRAMES then
    local st = "PARTIAL"
    if first_landing or first_morph then
      st = "GREEN"
    elseif first_elev then
      st = "PARTIAL_CERES"
    end
    finish(st, "max_frames")
    return
  end
  if movie.isloaded() then
    local ok, len = pcall(movie.length)
    if ok and fr >= len and fr > 0 then
      finish((first_landing or first_morph) and "GREEN" or "PARTIAL", "movie_end")
      return
    end
  end

  pcall(function()
    memory.usememorydomain("WRAM")
  end)
  local room = ru16(0x079B)
  local area = ru8(0x079F)
  local gs = ru16(0x0998)
  local items = ru16(0x09A4)
  local energy = ru16(0x09C2)
  local ok = trusted(room, area, gs)

  -- Heartbeat every 2k always (shows intro progress)
  if fr > 0 and fr % 2000 == 0 then
    L(string.format(
      "hb room=0x%04X area=%d gs=%d items=0x%04X e=%d trusted=%s elev=%s land=%s morph=%s",
      room, area, gs, items, energy, tostring(ok),
      tostring(first_elev), tostring(first_landing), tostring(first_morph)
    ))
  end

  if not ok then
    return
  end

  if room ~= last_room then
    last_room = room
    if unique[room] == nil then
      unique[room] = fr
      unique_n = unique_n + 1
    end
    L(string.format(
      "ROOM room=0x%04X area=%d gs=%d items=0x%04X e=%d unique=%d",
      room, area, gs, items, energy, unique_n
    ))
  end

  if first_elev == nil and room == ROOM_ELEV then
    first_elev = fr
    L("FIRST_CONTROL Ceres Elevator 0xDF45 (intro done)")
  end
  if first_ridley == nil and room == ROOM_RIDLEY then
    first_ridley = fr
    L("Ceres Ridley 0xE0B5")
  end
  if first_landing == nil and room == ROOM_LANDING then
    first_landing = fr
    L("Landing Site 0x91F8 — past Ceres escape")
  end
  if first_morph == nil and band(items, MORPH) ~= 0 then
    first_morph = fr
    L("MORPH items=0x" .. string.format("%04X", items))
  end

  if EARLY_EXIT and (first_landing or first_morph) and first_elev then
    finish("GREEN", "landing_or_morph_after_elev")
  end
end)
