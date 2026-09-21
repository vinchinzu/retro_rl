-- lsnes entry: power-on Ceres boot → Ridley tail-tank → Landing Site.
-- This script *is* the controller. No mailbox protocol. No startup movie.
-- Lua 5.1 / lsnes rr2-β25. TURBO=1 → exec("set-speed turbo").

dofile(debug.getinfo(1, "S").source:gsub("^@", ""):match("^(.*)/") .. "/path.lua")

local runtime = require("runtime")
local spine = require("ceres.spine")
local input_mod = require("input")

local src = debug.getinfo(1, "S").source:gsub("^@", "")
local lua_dir = src:match("^(.*)/") or src:match("^(.*)\\") or "."

local session = runtime.new_session()
local pending_names = {}
local pending_reason = "idle"
local finished = false
local failed = false
local leave_written = false

local co = coroutine.create(function()
  spine.play(session)
end)

local function log(msg)
  print("[ceres] " .. tostring(msg))
end

local function apply_joyset(names)
  local buttons
  if input_mod and input_mod.joyset_table then
    buttons = input_mod.joyset_table(names or {})
  else
    buttons = {
      B = false, Y = false, select = false, start = false,
      up = false, down = false, left = false, right = false,
      A = false, X = false, L = false, R = false,
    }
  end
  if input and input.joyset then
    input.joyset(1, buttons)
  end
end

local function extract_yield(a, b)
  if type(b) == "string" and type(a) == "table" then
    return a, b
  end
  if type(a) ~= "table" then
    return {}, tostring(a or "idle")
  end
  if type(a.names) == "table" then
    return a.names, a.reason or ""
  end
  if type(a[1]) == "table" then
    return a[1], a[2] or a.reason or ""
  end
  return a, a.reason or pending_reason or ""
end

local function num(v, default)
  v = tonumber(v)
  if v == nil then
    return default or 0
  end
  return v
end

local function state_xy(st)
  return num(st and (st.samus_x or st.x)), num(st and (st.samus_y or st.y))
end

local function leave_paths()
  return {
    lua_dir .. "/../recordings/ceres_leave.json",
    "recordings/ceres_leave.json",
    "sm/recordings/ceres_leave.json",
    "ceres_leave.json",
  }
end

local function ensure_dir(path)
  local dir = path:match("^(.*)/[^/]+$") or path:match("^(.*)\\[^\\]+$")
  if dir and dir ~= "" then
    os.execute('mkdir -p "' .. dir .. '"')
  end
end

local function write_leave(rec)
  local body = string.format(
    '{\n  "room": %d,\n  "gs": %d,\n  "xy": [%d, %d],\n  "pose": %d,\n  "health": %d,\n  "timer_type": %d,\n  "frame": %d\n}\n',
    rec.room, rec.gs, rec.x, rec.y, rec.pose, rec.health, rec.timer_type, rec.frame
  )
  local paths = leave_paths()
  local i
  for i = 1, #paths do
    local path = paths[i]
    ensure_dir(path)
    local f = io.open(path, "w")
    if f then
      f:write(body)
      f:close()
      return path
    end
  end
  return nil
end

local function pause_emu()
  if exec then
    pcall(function()
      exec("pause-emulator")
    end)
  end
end

local function on_success()
  if leave_written then
    apply_joyset({})
    return
  end
  leave_written = true
  finished = true
  if session.refresh then
    session:refresh()
  end
  local st = session.state or {}
  local x, y = state_xy(st)
  local rec = {
    room = num(st.room_id),
    gs = num(st.game_state),
    x = x,
    y = y,
    pose = num(st.pose),
    health = num(st.health),
    timer_type = num(st.timer_type),
    frame = num(session.frame or st.frame),
  }
  local path = write_leave(rec)
  log(string.format(
    "leave room=0x%04X gs=%d xy=%d,%d pose=%d health=%d timer_type=%d frame=%d",
    rec.room, rec.gs, rec.x, rec.y, rec.pose, rec.health, rec.timer_type, rec.frame
  ))
  if path then
    log("wrote " .. path)
  else
    log("failed to write ceres_leave.json")
  end
  apply_joyset({})
  pause_emu()
end

local function on_error(err)
  failed = true
  log("error: " .. tostring(err))
  apply_joyset({})
  pause_emu()
end

local function on_script_input()
  if finished or failed then
    apply_joyset({})
    return
  end
  local ok, a, b = coroutine.resume(co)
  if not ok then
    on_error(a)
    return
  end
  if coroutine.status(co) == "dead" then
    on_success()
    return
  end
  pending_names, pending_reason = extract_yield(a, b)
  apply_joyset(pending_names)
end

local function on_script_frame()
  if session and session.refresh then
    session:refresh()
  end
end

local function boot_emu()
  if not exec then
    return
  end
  pcall(function()
    exec("enable-sound off")
  end)
  if os.getenv("TURBO") == "1" then
    pcall(function()
      exec("set-speed turbo")
    end)
  end
  pcall(function()
    exec("clear-pause-on-end")
  end)
  pcall(function()
    exec("unpause-emulator")
  end)
end

if callback and callback.register then
  callback.register("input", on_script_input)
  callback.register("frame", on_script_frame)
  callback.register("frame_emulated", on_script_frame)
else
  function on_input()
    on_script_input()
  end
  function on_frame()
    on_script_frame()
  end
  function on_frame_emulated()
    on_script_frame()
  end
end

boot_emu()
log("run_ceres.lua start (no movie; spine.play drives joyset)")
