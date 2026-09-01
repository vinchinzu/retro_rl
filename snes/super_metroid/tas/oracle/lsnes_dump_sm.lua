-- Native-core dump for Sniq 100% LSMV under lsnes + bsnes v085.
--
-- Reads Super Metroid WRAM (same offsets as harness ram.py). Writes
-- proof/events under out_dir, exits on GREEN (Landing/morph) or movie end.
--
-- Flags (oracle_flags.txt next to this script, or in cwd / out_dir):
--   out_dir=...  early_exit=1  max_frames=60000  series_stride=0
--   playback_speed=turbo  (use 1 for the native-speed sync diagnostic)

local ADDR_ROOM_ID = 0x079B
local ADDR_AREA_INDEX = 0x079F
local ADDR_GAME_STATE = 0x0998
local ADDR_COLLECTED_ITEMS = 0x09A4
local ADDR_COLLECTED_BEAMS = 0x09A8
local ADDR_HEALTH = 0x09C2
local ADDR_MAX_HEALTH = 0x09C4
local ADDR_MAX_MISSILES = 0x09C8
local ADDR_SAMUS_POSE = 0x0A1C
local ADDR_SAMUS_FACING = 0x0A1E
local ADDR_MOVEMENT_TYPE = 0x0A1F
local ADDR_SAMUS_X = 0x0AF6
local ADDR_SAMUS_X_SUB = 0x0AF8
local ADDR_SAMUS_Y = 0x0AFA
local ADDR_SAMUS_Y_SUB = 0x0AFC
local ADDR_VELOCITY_Y_SUB = 0x0B2C
local ADDR_VELOCITY_Y = 0x0B2E
local ADDR_VERTICAL_DIRECTION = 0x0B36
local ADDR_SPEED_FLAG = 0x0B3C
local ADDR_SPEED_COUNTER = 0x0B3E
local ADDR_VELOCITY_X = 0x0B42
local ADDR_VELOCITY_X_SUB = 0x0B44
local ADDR_MOMENTUM_X = 0x0B46
local ADDR_MOMENTUM_X_SUB = 0x0B48
local ADDR_TIMER_TYPE = 0x0943
local ADDR_TIMER_FRAMES = 0x0945
local ADDR_TIMER_SECONDS = 0x0946
local ADDR_TIMER_MINUTES = 0x0947
local ADDR_INVINCIBILITY_TIMER = 0x18A8
local ADDR_KNOCKBACK_TIMER = 0x18AA

local ENEMY_BASE = 0x0F78
local ENEMY_STRIDE = 0x40
local ENEMY_SLOTS = 32
local ADDR_SAMUS_PROJ_TYPE = 0x0C04
local ADDR_SAMUS_PROJ_X = 0x0C18
local ADDR_SAMUS_PROJ_Y = 0x0C4A
local SAMUS_PROJ_SLOTS = 10
local ADDR_ENEMY_PROJ_ID = 0x1997
local ADDR_ENEMY_PROJ_X = 0x1A4B
local ADDR_ENEMY_PROJ_Y = 0x1A93
local ADDR_ENEMY_PROJ_ILIST = 0x1B47
local ENEMY_PROJ_SLOTS = 18

local MORPH_MASK = 0x0004
local BOMBS_MASK = 0x1000
local TRUST_RAM_AFTER = 60
local WRAM = 0x7E0000

local ROOM_ELEV = 0xDF45
local ROOM_LANDING = 0x91F8

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

local function dirname(path)
  if not path then
    return nil
  end
  return path:match("^(.*)/[^/]+$") or path:match("^(.*)\\[^\\]+$")
end

local function read_line(path)
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

local script_path = nil
if debug and debug.getinfo then
  local info = debug.getinfo(1, "S")
  if info and info.source then
    script_path = info.source:gsub("^@", "")
  end
end

local out_dir = nil
local flag_paths = {}
if script_path then
  local sd = dirname(script_path)
  if sd then
    out_dir = read_line(sd .. "/oracle_out_dir.txt")
    table.insert(flag_paths, sd .. "/oracle_flags.txt")
  end
end
out_dir = out_dir or read_line("oracle_out_dir.txt")
table.insert(flag_paths, "oracle_flags.txt")

local early_exit = true
local max_frames = 60000
local series_stride = 0
local playback_speed = "turbo"
local rom_path = "C:/lsnes/SuperMetroid.sfc"
local movie_path = "C:/lsnes/sniq_100_4010M.lsmv"

local function apply_flags(path)
  local f = io.open(path, "r")
  if not f then
    return
  end
  for line in f:lines() do
    local k, v = line:match("^([%w_]+)=(.*)$")
    if k == "out_dir" and v ~= "" then
      out_dir = v
    elseif k == "early_exit" then
      early_exit = (v ~= "0")
    elseif k == "max_frames" then
      max_frames = tonumber(v) or max_frames
    elseif k == "series_stride" then
      series_stride = tonumber(v) or 0
    elseif k == "playback_speed" and v ~= "" then
      playback_speed = v
    elseif k == "rom" and v ~= "" then
      rom_path = v
    elseif k == "movie" and v ~= "" then
      movie_path = v
    end
  end
  f:close()
end

for _, p in ipairs(flag_paths) do
  apply_flags(p)
end
if out_dir then
  apply_flags(out_dir .. "/oracle_flags.txt")
end
out_dir = out_dir or "."

local function ru16(addr)
  return memory.readword(WRAM + addr)
end

local function ru8(addr)
  return memory.readbyte(WRAM + addr)
end

local log_fh = io.open(out_dir .. "/dump_log.txt", "w")
local events_fh = io.open(out_dir .. "/events.jsonl", "w")
local timeline_fh = io.open(out_dir .. "/room_timeline.csv", "w")
local series_fh = nil
local entities_fh = nil
if series_stride > 0 then
  series_fh = io.open(out_dir .. "/series.jsonl", "w")
  entities_fh = io.open(out_dir .. "/entities.jsonl", "w")
end
if timeline_fh then
  timeline_fh:write("frame,kind,room_id,area,game_state,items,beams,energy,pose,x,y\n")
end

local function frame_now()
  if movie and movie.currentframe then
    return movie.currentframe()
  end
  return 0
end

local function json_str(s)
  s = tostring(s or ""):gsub("\\", "\\\\"):gsub('"', '\\"'):gsub("\n", " ")
  return s
end

local function log(msg)
  local line = string.format("[%d] %s", frame_now(), msg)
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
  local parts = { string.format('"frame":%d', frame_now()), string.format('"kind":"%s"', kind) }
  if payload then
    for k, v in pairs(payload) do
      if type(v) == "number" then
        table.insert(parts, string.format('"%s":%s', k, tostring(v)))
      else
        table.insert(parts, string.format('"%s":"%s"', k, json_str(v)))
      end
    end
  end
  events_fh:write("{" .. table.concat(parts, ",") .. "}\n")
  events_fh:flush()
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
local first_elev_frame = nil
local morph_frame = nil
local bombs_frame = nil
local landing_frame = nil
local green = false
local finished = false
local kicked = false
local vma_logged = false
local movie_ready = movie and movie.framecount and movie.framecount() >= 1000 or false

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
    facing = ru8(ADDR_SAMUS_FACING),
    movement_type = ru8(ADDR_MOVEMENT_TYPE),
    x = ru16(ADDR_SAMUS_X),
    x_sub = ru16(ADDR_SAMUS_X_SUB),
    y = ru16(ADDR_SAMUS_Y),
    y_sub = ru16(ADDR_SAMUS_Y_SUB),
    velocity_y = ru16(ADDR_VELOCITY_Y),
    velocity_y_sub = ru16(ADDR_VELOCITY_Y_SUB),
    vertical_direction = ru16(ADDR_VERTICAL_DIRECTION),
    speed_flag = ru16(ADDR_SPEED_FLAG),
    speed_counter = math.floor(ru16(ADDR_SPEED_COUNTER) / 256),
    velocity_x = ru16(ADDR_VELOCITY_X),
    velocity_x_sub = ru16(ADDR_VELOCITY_X_SUB),
    momentum_x = ru16(ADDR_MOMENTUM_X),
    momentum_x_sub = ru16(ADDR_MOMENTUM_X_SUB),
    timer_type = ru8(ADDR_TIMER_TYPE),
    timer_frames = ru8(ADDR_TIMER_FRAMES),
    timer_seconds = ru8(ADDR_TIMER_SECONDS),
    timer_minutes = ru8(ADDR_TIMER_MINUTES),
    invincibility_timer = ru16(ADDR_INVINCIBILITY_TIMER),
    knockback_timer = ru16(ADDR_KNOCKBACK_TIMER),
  }
end

local function live_projectiles()
  local rows = {}
  for slot = 0, SAMUS_PROJ_SLOTS - 1 do
    local kind = ru16(ADDR_SAMUS_PROJ_TYPE + slot * 2)
    if kind ~= 0 then
      table.insert(rows, {
        slot = slot,
        kind = kind,
        x = ru16(ADDR_SAMUS_PROJ_X + slot * 2),
        y = ru16(ADDR_SAMUS_PROJ_Y + slot * 2),
      })
    end
  end
  return rows
end

local function write_ceres_entities(f)
  if not entities_fh then
    return
  end
  for slot = 0, ENEMY_SLOTS - 1 do
    local base = ENEMY_BASE + slot * ENEMY_STRIDE
    local enemy_id = ru16(base)
    local hp = ru16(base + 0x14)
    local x = ru16(base + 0x02)
    local y = ru16(base + 0x06)
    if enemy_id ~= 0 and hp > 0 and x < 0xFE00 and y < 0xFE00 then
      entities_fh:write(string.format(
        '{"frame":%d,"kind":"enemy","slot":%d,"id":%d,"x":%d,"y":%d,"x_radius":%d,"y_radius":%d,"extra":%d,"hp":%d,"spritemap":%d}\n',
        f, slot, enemy_id, x, y, ru16(base + 0x0A), ru16(base + 0x0C),
        ru16(base + 0x10), hp, ru16(base + 0x16)
      ))
    end
  end
  -- Ceres dust / debris is bank-$86 enemy-projectile state, not an enemy
  -- slot. Without this table $9734/$9742 pieces are invisible to the oracle.
  for slot = 0, ENEMY_PROJ_SLOTS - 1 do
    local proj_id = ru16(ADDR_ENEMY_PROJ_ID + slot * 2)
    local x = ru16(ADDR_ENEMY_PROJ_X + slot * 2)
    local y = ru16(ADDR_ENEMY_PROJ_Y + slot * 2)
    if proj_id ~= 0 and x < 0xFE00 and y < 0xFE00 then
      entities_fh:write(string.format(
        '{"frame":%d,"kind":"enemy_projectile","slot":%d,"id":%d,"x":%d,"y":%d,"ilist":%d}\n',
        f, slot, proj_id, x, y,
        ru16(ADDR_ENEMY_PROJ_ILIST + slot * 2)
      ))
    end
  end
end

local function csv_row(kind, s)
  if not timeline_fh then
    return
  end
  timeline_fh:write(string.format(
    "%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
    frame_now(), kind, s.room_id, s.area, s.game_state, s.items, s.beams,
    s.energy, s.pose, s.x, s.y
  ))
  timeline_fh:flush()
end

local function write_proof(status, reason)
  local s = snapshot()
  local rooms_list = {}
  for rid, fr in pairs(unique_rooms) do
    table.insert(rooms_list, string.format('{"room_id":%d,"first_frame":%d}', rid, fr))
  end
  table.sort(rooms_list)
  local fh = io.open(out_dir .. "/proof.json", "w")
  if not fh then
    log("ERROR: cannot write proof")
    return
  end
  fh:write("{\n")
  fh:write(string.format('  "status": "%s",\n', status))
  fh:write(string.format('  "reason": "%s",\n', json_str(reason)))
  fh:write(string.format('  "source": "lsnes_oracle",\n'))
  fh:write(string.format('  "core": "bsnes v085 (Compatibility core)",\n'))
  fh:write(string.format('  "frame": %d,\n', frame_now()))
  local mlen = 0
  if movie and movie.framecount then
    mlen = movie.framecount() or 0
  end
  fh:write(string.format('  "movie_length": %d,\n', mlen))
  fh:write(string.format('  "out_dir": "%s",\n', json_str(out_dir)))
  fh:write(string.format('  "unique_rooms": %d,\n', unique_room_count))
  fh:write(string.format('  "zebes_rooms": %d,\n', zebes_rooms))
  fh:write(string.format('  "ceres_rooms": %d,\n', ceres_rooms))
  fh:write(string.format('  "unique_areas": %d,\n', unique_area_count))
  fh:write(string.format('  "first_control_frame": %s,\n', first_control_frame and tostring(first_control_frame) or "null"))
  fh:write(string.format('  "first_elev_frame": %s,\n', first_elev_frame and tostring(first_elev_frame) or "null"))
  fh:write(string.format('  "landing_frame": %s,\n', landing_frame and tostring(landing_frame) or "null"))
  fh:write(string.format('  "morph_frame": %s,\n', morph_frame and tostring(morph_frame) or "null"))
  fh:write(string.format('  "bombs_frame": %s,\n', bombs_frame and tostring(bombs_frame) or "null"))
  fh:write(string.format('  "final_room_id": %d,\n', s.room_id))
  fh:write(string.format('  "final_area": %d,\n', s.area))
  fh:write(string.format('  "final_items": %d,\n', s.items))
  fh:write(string.format('  "final_beams": %d,\n', s.beams))
  fh:write(string.format('  "final_energy": %d,\n', s.energy))
  fh:write(string.format('  "rooms": [%s]\n', table.concat(rooms_list, ",")))
  fh:write("}\n")
  fh:close()
  log("proof written status=" .. status)
end

local function finish(status, reason)
  if finished then
    return
  end
  finished = true
  write_proof(status, reason)
  if log_fh then log_fh:close(); log_fh = nil end
  if events_fh then events_fh:close(); events_fh = nil end
  if timeline_fh then timeline_fh:close(); timeline_fh = nil end
  if series_fh then series_fh:close(); series_fh = nil end
  if entities_fh then entities_fh:close(); entities_fh = nil end
  exec("quit-emulator")
end

local function is_green()
  if landing_frame ~= nil then
    return true
  end
  if morph_frame ~= nil and zebes_rooms >= 1 then
    return true
  end
  return first_elev_frame ~= nil and zebes_rooms >= 1 and (morph_frame ~= nil or last_items > 0)
end

local function log_vmas()
  if vma_logged or not memory.vma_count then
    return
  end
  vma_logged = true
  local n = memory.vma_count()
  log("vma_count=" .. tostring(n))
  for i = 0, n - 1 do
    local v = memory.read_vma(i)
    if v then
      log(string.format(
        "vma[%d] %s base=0x%X size=%d",
        i, tostring(v.region_name or "?"), tonumber(v.baseaddr) or 0, tonumber(v.size) or 0
      ))
    end
  end
end

function on_pre_load(name)
  log("pre_load " .. tostring(name))
end

function on_err_load(name)
  log("ERR load " .. tostring(name))
  emit("load_error", { name = tostring(name) })
end

function on_post_load(name, is_state)
  movie_ready = true
  local mlen = movie and movie.framecount and movie.framecount() or 0
  log(string.format(
    "post_load name=%s is_state=%s frames=%d readonly=%s",
    tostring(name), tostring(is_state), mlen,
    tostring(movie and movie.readonly and movie.readonly())
  ))
  emit("post_load", { name = tostring(name), frames = mlen })
  if mlen < 1000 then
    log("WARN movie.framecount too small — load may have missed the LSMV")
  end
  exec("enable-sound off")
  exec("set-speed " .. playback_speed)
  exec("clear-pause-on-end")
end

local function tick()
  if finished then
    return
  end
  local f = frame_now()
  if f == last_tick_f then
    return
  end
  last_tick_f = f
  if f > max_frames then
    finish(green and "GREEN" or "PARTIAL", "max_frames reached")
    return
  end
  local mlen = movie and movie.framecount and movie.framecount() or 0
  -- --lua runs before --load; do not treat the empty boot movie as EOF.
  if not movie_ready then
    if f > 0 and f % 120 == 0 then
      log(string.format("waiting for movie load f=%d mlen=%d", f, mlen))
    end
    if mlen >= 1000 then
      movie_ready = true
      log("movie_ready inferred from framecount=" .. tostring(mlen))
    end
    if not movie_ready then
      return
    end
  end
  if mlen > 1000 and f >= mlen and f > 0 then
    finish(green and "GREEN" or "PARTIAL", "movie end")
    return
  end
  if f < TRUST_RAM_AFTER then
    return
  end
  if not vma_logged then
    log_vmas()
  end

  local s = snapshot()
  local area_ok = s.area >= 0 and s.area <= 6
  local room_ok = s.room_id ~= 0 and s.room_id ~= 0xFFFF

  local projectiles = live_projectiles()
  if series_fh and series_stride > 0 and f % series_stride == 0 then
    series_fh:write(string.format(
      '{"frame":%d,"room_id":%d,"area":%d,"gs":%d,"pose":%d,"facing":%d,"movement_type":%d,"x":%d,"x_sub":%d,"y":%d,"y_sub":%d,"velocity_x":%d,"velocity_x_sub":%d,"momentum_x":%d,"momentum_x_sub":%d,"velocity_y":%d,"velocity_y_sub":%d,"vertical_direction":%d,"speed_flag":%d,"speed_counter":%d,"invincibility_timer":%d,"knockback_timer":%d,"timer_type":%d,"timer_frames":%d,"timer_seconds":%d,"timer_minutes":%d,"projectiles":%d,"items":%d,"beams":%d,"energy":%d}\n',
      f, s.room_id, s.area, s.game_state, s.pose, s.facing, s.movement_type,
      s.x, s.x_sub, s.y, s.y_sub, s.velocity_x, s.velocity_x_sub,
      s.momentum_x, s.momentum_x_sub, s.velocity_y, s.velocity_y_sub,
      s.vertical_direction, s.speed_flag, s.speed_counter,
      s.invincibility_timer, s.knockback_timer, s.timer_type, s.timer_frames,
      s.timer_seconds, s.timer_minutes, #projectiles, s.items, s.beams, s.energy
    ))
    if s.area == 6 then
      write_ceres_entities(f)
    end
  end

  if first_control_frame == nil and s.game_state == 8 and room_ok and area_ok then
    first_control_frame = f
    log(string.format(
      "first_control room=0x%04X area=%d items=0x%04X energy=%d",
      s.room_id, s.area, s.items, s.energy
    ))
    emit("control", { room_id = s.room_id, area = s.area, items = s.items, pose = s.pose, x = s.x, y = s.y })
    csv_row("control", s)
    last_items = s.items
    last_beams = s.beams
    last_room = s.room_id
  end

  if not area_ok then
    if f % 5000 == 0 then
      log(string.format("heartbeat (untrusted) area=%d room=0x%04X gs=%d", s.area, s.room_id, s.game_state))
    end
    return
  end

  if room_ok and s.room_id ~= last_room then
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
    if s.room_id == ROOM_ELEV and first_elev_frame == nil then
      first_elev_frame = f
      log("FIRST Ceres elev 0xDF45")
    end
    if s.room_id == ROOM_LANDING and landing_frame == nil then
      landing_frame = f
      log("LANDING 0x91F8")
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
      pose = s.pose,
      x = s.x,
      y = s.y,
    })
    csv_row("room_enter", s)
  end

  if last_items >= 0 and s.items ~= last_items then
    local prev = last_items
    last_items = s.items
    log(string.format("items 0x%04X -> 0x%04X", prev, s.items))
    emit("item_gain", { items = s.items, prev = prev })
    csv_row("item_gain", s)
    if morph_frame == nil and band(s.items, MORPH_MASK) ~= 0 then
      morph_frame = f
      log("MORPH gained")
      emit("morph", { items = s.items })
    end
    if bombs_frame == nil and band(s.items, BOMBS_MASK) ~= 0 then
      bombs_frame = f
      log("BOMBS gained")
      emit("bombs", { items = s.items })
    end
  elseif last_items < 0 and first_control_frame ~= nil then
    last_items = s.items
  end

  if last_beams >= 0 and s.beams ~= last_beams then
    local prev = last_beams
    last_beams = s.beams
    log(string.format("beams 0x%04X -> 0x%04X", prev, s.beams))
    emit("beam_gain", { beams = s.beams, prev = prev })
    csv_row("beam_gain", s)
  elseif last_beams < 0 and first_control_frame ~= nil then
    last_beams = s.beams
  end

  if not green and is_green() then
    green = true
    log("GREEN native-core milestone")
    emit("green", {
      items = last_items,
      zebes_rooms = zebes_rooms,
      ceres_rooms = ceres_rooms,
      unique_rooms = unique_room_count,
      morph_frame = morph_frame or -1,
      landing_frame = landing_frame or -1,
    })
    write_proof("GREEN", "landing or morph past Ceres")
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

function on_frame()
  tick()
end

function on_frame_emulated()
  -- Same tick; some lsnes builds fire this and not on_frame.
  tick()
end

local last_tick_f = -1
local load_attempts = 0

function on_idle()
  if finished then
    return
  end
  if not movie_ready then
    load_attempts = load_attempts + 1
    log(string.format("load attempt %d rom=%s movie=%s", load_attempts, rom_path, movie_path))
    exec("load-rom " .. rom_path)
    exec("load-movie " .. movie_path)
    exec("load-readonly " .. movie_path)
    if load_attempts >= 6 then
      finish("FAIL", "load-movie never posted")
      return
    end
    set_idle_timeout(500000)
    return
  end
  if kicked then
    return
  end
  kicked = true
  exec("enable-sound off")
  exec("set-speed " .. playback_speed)
  exec("clear-pause-on-end")
  exec("unpause-emulator")
  log("unpaused speed=" .. playback_speed)
end

function on_quit()
  if finished then
    return
  end
  if frame_now() < 2 then
    return
  end
  write_proof(green and "GREEN" or "PARTIAL", "lsnes quit")
end

log("lsnes_dump_sm.lua start out_dir=" .. out_dir)
log("early_exit=" .. tostring(early_exit) .. " max_frames=" .. tostring(max_frames))
log("playback_speed=" .. playback_speed)
if movie and movie.framecount then
  log("movie.framecount=" .. tostring(movie.framecount()))
end
set_idle_timeout(200000)
emit("start", { out_dir = out_dir, max_frames = max_frames })
