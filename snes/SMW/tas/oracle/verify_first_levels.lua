-- Verify converted SMW TAS input in BizHawk from power-on through early exits.
--
-- Output is evidence, not merely a frame counter: level entry/exit boundaries
-- are detected from WRAM, named BizHawk states are saved, and compact segment
-- bounds are written for exact input-skill extraction.

local GAME_MODE = 0x0100
local PLAYER_X = 0x00D1
local PLAYER_Y = 0x00D3
local LIVES = 0x0DBE
local TIMER_HUNDREDS = 0x0F31
local TIMER_TENS = 0x0F32
local TIMER_ONES = 0x0F33
local TRANSLEVEL = 0x13BF
local END_LEVEL_TIMER = 0x1493
local EXITS_COMPLETED = 0x1F2E

local MODE_LEVEL = 0x14
local TRUST_RAM_AFTER = 60

local function ud(key, default)
  if userdata and userdata.get then
    local ok, value = pcall(userdata.get, key)
    if ok and value ~= nil then
      return tostring(value)
    end
  end
  return default
end

local out_dir = ud("out_dir", ".")
local target_levels = tonumber(ud("target_levels", "2")) or 2
local max_frames = tonumber(ud("max_frames", "60000")) or 60000

local events_fh = assert(io.open(out_dir .. "/events.jsonl", "w"))
local log_fh = assert(io.open(out_dir .. "/verify.log", "w"))

local function json_escape(value)
  return tostring(value):gsub("\\", "\\\\"):gsub('"', '\\"'):gsub("\n", "\\n")
end

local function ru8(address)
  return mainmemory.read_u8(address)
end

local function ru16(address)
  return mainmemory.read_u16_le(address)
end

local function snapshot()
  return {
    game_mode = ru8(GAME_MODE),
    translevel = ru8(TRANSLEVEL),
    player_x = ru16(PLAYER_X),
    player_y = ru16(PLAYER_Y),
    lives = ru8(LIVES),
    timer_hundreds = ru8(TIMER_HUNDREDS),
    timer_tens = ru8(TIMER_TENS),
    timer_ones = ru8(TIMER_ONES),
    end_level_timer = ru8(END_LEVEL_TIMER),
    exits_completed = ru8(EXITS_COMPLETED),
  }
end

local function snapshot_json(value)
  return string.format(
    '{"game_mode":%d,"translevel":%d,"player_x":%d,"player_y":%d,' ..
      '"lives":%d,"timer_hundreds":%d,"timer_tens":%d,"timer_ones":%d,' ..
      '"end_level_timer":%d,"exits_completed":%d}',
    value.game_mode,
    value.translevel,
    value.player_x,
    value.player_y,
    value.lives,
    value.timer_hundreds,
    value.timer_tens,
    value.timer_ones,
    value.end_level_timer,
    value.exits_completed
  )
end

local function log(message)
  local line = string.format("[%d] %s", emu.framecount(), message)
  print(line)
  log_fh:write(line .. "\n")
  log_fh:flush()
end

local function emit(kind, fields)
  local parts = {
    string.format('"frame":%d', emu.framecount()),
    string.format('"kind":"%s"', json_escape(kind)),
  }
  if fields then
    for key, value in pairs(fields) do
      if type(value) == "number" then
        table.insert(parts, string.format('"%s":%d', key, value))
      elseif type(value) == "boolean" then
        table.insert(parts, string.format('"%s":%s', key, value and "true" or "false"))
      else
        table.insert(parts, string.format('"%s":"%s"', key, json_escape(value)))
      end
    end
  end
  events_fh:write("{" .. table.concat(parts, ",") .. "}\n")
  events_fh:flush()
end

local segments = {}
local active = nil
local saw_end_timer = false
local max_player_x = 0
local last_game_mode = nil
local outside_active_level = false
local previous_lives = nil
local lives_drops_at_outside = 0

local function save_checkpoint(name)
  local path = out_dir .. "/" .. name .. ".State"
  local ok, result = pcall(savestate.save, path, true)
  emit("checkpoint", { name = name, saved = ok and result == true, path = path })
  return path, ok and result == true
end

local function segment_json(segment)
  return string.format(
    '{"index":%d,"translevel":%d,"entry_frame":%d,"exit_frame":%d,' ..
      '"max_player_x":%d,"retry_count":%d,"sublevel_count":%d,' ..
      '"lives_drops":%d,"completion_signal":"%s",' ..
      '"entry_state":"%s","exit_state":"%s",' ..
      '"entry_ram":%s,"exit_ram":%s}',
    segment.index,
    segment.translevel,
    segment.entry_frame,
    segment.exit_frame,
    segment.max_player_x,
    segment.retry_count,
    segment.sublevel_count,
    segment.lives_drops,
    json_escape(segment.completion_signal),
    json_escape(segment.entry_state),
    json_escape(segment.exit_state),
    snapshot_json(segment.entry_ram),
    snapshot_json(segment.exit_ram)
  )
end

local function abort_active(reason, ram)
  if not active then
    return
  end
  log(string.format(
    "level %d aborted trans=0x%02X reason=%s",
    active.index,
    active.translevel,
    reason
  ))
  emit("level_abort", {
    index = active.index,
    translevel = active.translevel,
    current_translevel = ram.translevel,
    reason = reason,
  })
  active = nil
  saw_end_timer = false
  max_player_x = 0
  outside_active_level = false
  previous_lives = nil
  lives_drops_at_outside = 0
end

local function write_proof(status, reason)
  local final = snapshot()
  local segment_parts = {}
  for _, segment in ipairs(segments) do
    table.insert(segment_parts, segment_json(segment))
  end
  local movie_length = 0
  if movie.isloaded() then
    local ok, value = pcall(movie.length)
    if ok then
      movie_length = value
    end
  end
  local fh = assert(io.open(out_dir .. "/proof.json", "w"))
  fh:write("{\n")
  fh:write(string.format('  "status": "%s",\n', json_escape(status)))
  fh:write(string.format('  "reason": "%s",\n', json_escape(reason)))
  fh:write(string.format('  "frame": %d,\n', emu.framecount()))
  fh:write(string.format('  "movie_loaded": %s,\n', movie.isloaded() and "true" or "false"))
  fh:write(string.format('  "movie_mode": "%s",\n', json_escape(movie.mode())))
  fh:write(string.format('  "movie_length": %d,\n', movie_length))
  fh:write(string.format('  "movie_file": "%s",\n', json_escape(movie.filename())))
  fh:write(string.format('  "system_id": "%s",\n', json_escape(emu.getsystemid())))
  fh:write(string.format('  "rom_hash": "%s",\n', json_escape(gameinfo.getromhash())))
  fh:write(string.format('  "target_levels": %d,\n', target_levels))
  fh:write(string.format('  "levels_completed": %d,\n', #segments))
  fh:write(string.format('  "final_ram": %s,\n', snapshot_json(final)))
  fh:write('  "segments": [' .. table.concat(segment_parts, ",") .. "]\n")
  fh:write("}\n")
  fh:close()
  log(status .. ": " .. reason)
end

pcall(function()
  client.speedmode(400)
end)
pcall(function()
  client.unpause()
end)

if not movie.isloaded() then
  write_proof("RED", "BizHawk did not load a movie")
  events_fh:close()
  log_fh:close()
  client.exit()
  return
end

emit("start", {
  movie = movie.filename(),
  movie_length = movie.length(),
  rom_hash = gameinfo.getromhash(),
  system_id = emu.getsystemid(),
})

while true do
  local frame = emu.framecount()
  local ram = snapshot()
  local timer_started = ram.timer_hundreds > 0 or ram.timer_tens > 0 or ram.timer_ones > 0
  local in_playable_level = frame >= TRUST_RAM_AFTER and ram.game_mode == MODE_LEVEL and timer_started

  if ram.game_mode ~= last_game_mode then
    emit("game_mode", {
      from = last_game_mode or -1,
      to = ram.game_mode,
      translevel = ram.translevel,
      player_x = ram.player_x,
    })
    last_game_mode = ram.game_mode
  end
  if frame % 1000 == 0 then
    emit("heartbeat", {
      game_mode = ram.game_mode,
      translevel = ram.translevel,
      player_x = ram.player_x,
      end_level_timer = ram.end_level_timer,
      levels_completed = #segments,
    })
  end

  if not active and in_playable_level then
    local index = #segments + 1
    local state_name = string.format("level_%02d_trans_%02X_entry", index, ram.translevel)
    local state_path = save_checkpoint(state_name)
    active = {
      index = index,
      translevel = ram.translevel,
      current_translevel = ram.translevel,
      entry_frame = frame,
      entry_state = state_path,
      entry_ram = ram,
      retry_count = 0,
      sublevel_count = 0,
      lives_drops = 0,
    }
    saw_end_timer = ram.end_level_timer > 0
    max_player_x = ram.player_x
    outside_active_level = false
    previous_lives = ram.lives
    lives_drops_at_outside = 0
    log(string.format("level %d entry trans=0x%02X x=%d", index, ram.translevel, ram.player_x))
    emit("level_entry", { index = index, translevel = ram.translevel, player_x = ram.player_x })
  elseif active then
    if previous_lives and ram.lives < previous_lives then
      active.lives_drops = active.lives_drops + (previous_lives - ram.lives)
      emit("lives_drop", {
        index = active.index,
        translevel = active.translevel,
        from = previous_lives,
        to = ram.lives,
      })
    end
    previous_lives = ram.lives
    if ram.player_x > max_player_x then
      max_player_x = ram.player_x
    end
    if ram.end_level_timer > 0 then
      saw_end_timer = true
    end
    local exits_increased = ram.exits_completed > active.entry_ram.exits_completed
    if ram.game_mode ~= MODE_LEVEL and (saw_end_timer or exits_increased) then
      local state_name = string.format("level_%02d_trans_%02X_exit", active.index, active.translevel)
      local state_path = save_checkpoint(state_name)
      active.exit_frame = frame
      active.exit_state = state_path
      active.exit_ram = ram
      active.max_player_x = max_player_x
      active.completion_signal = saw_end_timer and "end_level_timer" or "exits_completed"
      table.insert(segments, active)
      log(string.format(
        "level %d exit trans=0x%02X mode=0x%02X max_x=%d",
        active.index,
        active.translevel,
        ram.game_mode,
        max_player_x
      ))
      emit("level_exit", {
        index = active.index,
        translevel = active.translevel,
        game_mode = ram.game_mode,
        max_player_x = max_player_x,
      })
      active = nil
      saw_end_timer = false
      max_player_x = 0
      outside_active_level = false
      previous_lives = nil
      lives_drops_at_outside = 0
      if #segments >= target_levels then
        write_proof("GREEN", "target early level exits reproduced")
        break
      end
    elseif ram.game_mode ~= MODE_LEVEL then
      if not outside_active_level then
        lives_drops_at_outside = active.lives_drops
      end
      outside_active_level = true
      if ram.game_mode <= 0x0A then
        abort_active("returned to title/file-select flow", ram)
      end
    elseif outside_active_level then
      local life_was_lost = active.lives_drops > lives_drops_at_outside
      if ram.translevel ~= active.current_translevel or not life_was_lost then
        active.sublevel_count = active.sublevel_count + 1
        active.current_translevel = ram.translevel
        outside_active_level = false
        emit("level_sublevel", {
          index = active.index,
          translevel = active.translevel,
          current_translevel = ram.translevel,
          sublevel_count = active.sublevel_count,
        })
      else
        active.retry_count = active.retry_count + 1
        outside_active_level = false
        emit("level_retry", {
          index = active.index,
          translevel = active.translevel,
          retry_count = active.retry_count,
        })
      end
    end
  end

  if frame >= max_frames then
    write_proof("RED", "maximum verification frame reached before target exits")
    break
  end
  if movie.mode() == "FINISHED" then
    write_proof("RED", "movie ended before target exits")
    break
  end
  emu.frameadvance()
end

events_fh:close()
log_fh:close()
client.exit()
