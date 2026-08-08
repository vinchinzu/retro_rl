-- FCEUX 2.6.6 RAM trace dumper for HappyLee #1715M oracle checkpoints.
-- Invoked via: fceux --playmov <fm2> --loadlua dump_ram_trace.lua <rom>
--
-- Config: companion _run_config.lua next to this script (preferred), or
-- SMB_ORACLE_CFG global set by a generated entry script. Output paths in
-- the config MUST be absolute so FCEUX CWD cannot mis-place artifacts.

if emu.speedmode then
  emu.speedmode("maximum")
end

local function script_dir()
  local src = debug.getinfo(1, "S").source
  if type(src) == "string" and src:sub(1, 1) == "@" then
    src = src:sub(2)
    local dir = src:match("(.*[/\\])")
    if dir then
      return dir
    end
  end
  return "./"
end

local function load_cfg()
  if type(SMB_ORACLE_CFG) == "table" then
    return SMB_ORACLE_CFG
  end
  local dir = script_dir()
  local candidates = {
    dir .. "_run_config.lua",
    -- absolute fallbacks are injected by the Python driver when present
  }
  if type(SMB_ORACLE_CFG_PATH) == "string" then
    table.insert(candidates, 1, SMB_ORACLE_CFG_PATH)
  end
  for _, path in ipairs(candidates) do
    local ok, cfg = pcall(dofile, path)
    if ok and type(cfg) == "table" then
      return cfg
    end
  end
  return nil
end

local cfg = load_cfg()
if type(cfg) ~= "table" then
  error(
    "oracle dump_ram_trace.lua: missing _run_config.lua with absolute out_* paths"
  )
end
if type(cfg.out_trace) ~= "string" or cfg.out_trace:sub(1, 1) ~= "/" then
  error("oracle dump_ram_trace.lua: out_trace must be an absolute path")
end
if type(cfg.out_named) ~= "string" or cfg.out_named:sub(1, 1) ~= "/" then
  error("oracle dump_ram_trace.lua: out_named must be an absolute path")
end

local function s8(v)
  if v >= 128 then return v - 256 end
  return v
end

local function open_w(path)
  local f, err = io.open(path, "w")
  if not f then
    error("cannot open " .. tostring(path) .. ": " .. tostring(err))
  end
  return f
end

local trace_f = open_w(cfg.out_trace)
local named_f = open_w(cfg.out_named)

local function enemies()
  local list = {}
  for slot = 0, 4 do
    local flag = memory.readbyte(0x000F + slot)
    if flag ~= 0 then
      local typ = memory.readbyte(0x0016 + slot)
      local st = memory.readbyte(0x001E + slot)
      local x = memory.readbyte(0x006E + slot) * 256 + memory.readbyte(0x0087 + slot)
      local y = memory.readbyte(0x00CF + slot)
      list[#list + 1] = string.format(
        '{"slot":%d,"type":%d,"state":%d,"x":%d,"y":%d}',
        slot, typ, st, x, y
      )
    end
  end
  return "[" .. table.concat(list, ",") .. "]"
end

local function grounded(ps, y_speed)
  -- Mirror smb.ram.is_in_air / grounded heuristics (y_speed primary).
  if ps == 0x0B then return false end
  if y_speed ~= 0 then return false end
  if ps == 0x00 or ps == 0x08 then return true end
  if ps == 0x01 or ps == 0x0A then return false end
  return true
end

local function snap_json(fc)
  local ps = memory.readbyte(0x000E)
  local facing = memory.readbyte(0x0033)
  local x_speed = s8(memory.readbyte(0x0057))
  local x_page = memory.readbyte(0x006D)
  local x_off = memory.readbyte(0x0086)
  local y_speed = s8(memory.readbyte(0x009F))
  local y = memory.readbyte(0x00CE)
  local screen_x_player = memory.readbyte(0x03AD)
  local x_frac = memory.readbyte(0x0400)
  local y_frac = memory.readbyte(0x0416)
  local area = memory.readbyte(0x0750)
  local power = memory.readbyte(0x0756)
  local lives = memory.readbyte(0x075A)
  local world = memory.readbyte(0x075F)
  local level = memory.readbyte(0x0760)
  local oper = memory.readbyte(0x0770)
  local cam_page = memory.readbyte(0x071A)
  local cam_x = memory.readbyte(0x071C)
  local th = memory.readbyte(0x07F8)
  local tt = memory.readbyte(0x07F9)
  local to = memory.readbyte(0x07FA)
  local frame_counter = memory.readbyte(0x0009)
  local px = x_page * 256 + x_off
  local cam = cam_page * 256 + cam_x
  local timer = th * 100 + tt * 10 + to
  local g = grounded(ps, y_speed)
  local ens = enemies()
  return string.format(
    '{"movie_frame":%d,"world":%d,"level":%d,"area_pointer":%d,"oper_mode":%d,'
      .. '"player_state":%d,"player_x":%d,"player_y":%d,"x_page":%d,"x_offset":%d,'
      .. '"x_frac":%d,"y_frac":%d,"x_speed":%d,"y_speed":%d,"grounded":%s,"in_air":%s,'
      .. '"facing":%d,"player_power":%d,"timer":%d,"timer_hundreds":%d,"timer_tens":%d,'
      .. '"timer_ones":%d,"timer_mod21":%d,"lives":%d,"screen_x":%d,"player_screen_x":%d,'
      .. '"frame_counter":%d,"enemies":%s}',
    fc, world, level, area, oper,
    ps, px, y, x_page, x_off,
    x_frac, y_frac, x_speed, y_speed,
    g and "true" or "false", (not g) and "true" or "false",
    facing, power, timer, th, tt, to, timer % 21, lives, cam, screen_x_player,
    frame_counter, ens
  )
end

-- Named landmark state machine
local seen = {
  control_8_2 = false,
  leave_8_2 = false,
  control_8_3 = false,
  early_8_3 = false,
  mid_8_3 = false,
  mid2_8_3 = false,
  hammer_bro_8_3 = false,
  flag_approach_8_3 = false,
  flagpole_8_3 = false,
  leave_8_3 = false,
  control_8_4 = false,
  axe_8_4 = false,
  ending = false,
}

local function emit_named(name, fc)
  local line = snap_json(fc)
  -- inject name field after first {
  line = '{"name":"' .. name .. '",' .. line:sub(2)
  named_f:write(line .. "\n")
  named_f:flush()
end

local function is_control(world, level, oper, ps, px, dying)
  if dying then return false end
  if oper ~= 1 then return false end
  if ps ~= 7 and ps ~= 8 then return false end
  if world ~= 7 then return false end
  return true
end

local max_x_83 = 0
local prev_world, prev_level = -1, -1
local n_adv = 0
local START = cfg.start_frame or 0
local ENDF = cfg.end_frame or 18000
local EVERY = cfg.sample_every or 1
local DENSE_FROM = cfg.dense_from or START
local DENSE_TO = cfg.dense_to or ENDF
local finished = false

local function finish(fc)
  if finished then return end
  finished = true
  named_f:write(string.format(
    '{"name":"_summary","end_frame":%d,"seen":{"control_8_2":%s,"leave_8_2":%s,"control_8_3":%s,'
      .. '"early_8_3":%s,"mid_8_3":%s,"mid2_8_3":%s,"hammer_bro_8_3":%s,'
      .. '"flag_approach_8_3":%s,"flagpole_8_3":%s,"leave_8_3":%s,"control_8_4":%s,"ending":%s}}\n',
    fc or ENDF,
    tostring(seen.control_8_2), tostring(seen.leave_8_2), tostring(seen.control_8_3),
    tostring(seen.early_8_3), tostring(seen.mid_8_3), tostring(seen.mid2_8_3),
    tostring(seen.hammer_bro_8_3), tostring(seen.flag_approach_8_3),
    tostring(seen.flagpole_8_3), tostring(seen.leave_8_3), tostring(seen.control_8_4),
    tostring(seen.ending)
  ))
  trace_f:close()
  named_f:close()
  emu.exit()
end

while true do
  local fc = emu.framecount()
  if fc >= START and fc <= ENDF then
    local sample = false
    if fc >= DENSE_FROM and fc <= DENSE_TO then
      sample = (fc % EVERY == 0)
    else
      sample = (fc % math.max(EVERY * 5, 5) == 0)
    end

    local world = memory.readbyte(0x075F)
    local level = memory.readbyte(0x0760)
    local oper = memory.readbyte(0x0770)
    local ps = memory.readbyte(0x000E)
    local px = memory.readbyte(0x006D) * 256 + memory.readbyte(0x0086)
    local dying = (ps == 0x0B)

    if sample then
      trace_f:write(snap_json(fc) .. "\n")
    end

    -- 8-2 control
    if not seen.control_8_2
      and is_control(world, level, oper, ps, px, dying)
      and level == 1
      and px >= 20 and px <= 120
    then
      seen.control_8_2 = true
      emit_named("control_8_2", fc)
    end

    -- 8-2 -> 8-3 level transition (first frame on 8-3 load)
    if not seen.leave_8_2
      and prev_world == 7 and prev_level == 1
      and world == 7 and level == 2
    then
      seen.leave_8_2 = true
      emit_named("leave_8_2_to_8_3", fc)
    end

    -- 8-3 controllable handoff
    if not seen.control_8_3
      and is_control(world, level, oper, ps, px, dying)
      and level == 2
      and px >= 20 and px <= 120
    then
      seen.control_8_3 = true
      emit_named("control_8_3", fc)
      max_x_83 = px
    end

    if seen.control_8_3 and not seen.leave_8_3 and world == 7 and level == 2 then
      if px > max_x_83 then max_x_83 = px end

      -- early 8-3 after first obstacle band (~x 200-400)
      if not seen.early_8_3 and max_x_83 >= 280 and px >= 200 then
        seen.early_8_3 = true
        emit_named("early_8_3_after_first_obstacle", fc)
      end

      -- middle sections
      if not seen.mid_8_3 and max_x_83 >= 900 then
        seen.mid_8_3 = true
        emit_named("mid_8_3_x900", fc)
      end
      if not seen.mid2_8_3 and max_x_83 >= 1600 then
        seen.mid2_8_3 = true
        emit_named("mid_8_3_x1600", fc)
      end

      -- Hammer Bro presence (type 0x05)
      if not seen.hammer_bro_8_3 then
        for slot = 0, 4 do
          if memory.readbyte(0x000F + slot) ~= 0
            and memory.readbyte(0x0016 + slot) == 0x05
          then
            local ex = memory.readbyte(0x006E + slot) * 256 + memory.readbyte(0x0087 + slot)
            if math.abs(ex - px) < 200 then
              seen.hammer_bro_8_3 = true
              emit_named("hammer_bro_nearby_8_3", fc)
              break
            end
          end
        end
      end

      -- flagpole approach
      if not seen.flag_approach_8_3 and max_x_83 >= 3050 then
        seen.flag_approach_8_3 = true
        emit_named("flag_approach_8_3", fc)
      end

      -- flagpole grab (player_state 0x04)
      if not seen.flagpole_8_3 and ps == 0x04 then
        seen.flagpole_8_3 = true
        emit_named("flagpole_grab_8_3", fc)
      end
    end

    -- 8-3 -> 8-4 leave
    if not seen.leave_8_3
      and prev_world == 7 and prev_level == 2
      and world == 7 and level == 3
    then
      seen.leave_8_3 = true
      emit_named("leave_8_3_to_8_4", fc)
    end

    -- 8-4 control entry
    if not seen.control_8_4
      and is_control(world, level, oper, ps, px, dying)
      and level == 3
      and px >= 20 and px <= 200
    then
      seen.control_8_4 = true
      emit_named("control_8_4", fc)
    end

    -- axe / ending (oper_mode 2 on 8-4)
    if not seen.ending and world == 7 and level == 3 and oper == 2 then
      seen.ending = true
      emit_named("ending_oper_mode_2", fc)
    end

    prev_world, prev_level = world, level
  end

  if fc >= ENDF then
    finish(fc)
    return
  end

  emu.frameadvance()
  n_adv = n_adv + 1
  if n_adv > (ENDF + 5000) then
    finish(fc)
    return
  end
end
