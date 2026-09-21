-- Ceres station spine: open-loop boot, then outbound + escape play functions.
-- No DoorEdge / SpineHop tables. No TAS hop-clock JSON comparison.

local M = {}

M.BOOT_MENU_MASH_FRAMES = 400
M.BOOT_MAX_FRAMES = 12000
M.ROOM_CERES_ELEVATOR = 0xDF45

local function num(v, default)
  v = tonumber(v)
  if v == nil then
    return default or 0
  end
  return v
end

local function i16(v)
  v = num(v)
  if v >= 0x8000 then
    return v - 0x10000
  end
  return v
end

local function brief(state)
  if not state then
    return "nil"
  end
  return string.format(
    "room=0x%04X gs=%s xy=%s,%s",
    num(state.room_id),
    tostring(state.game_state),
    tostring(state.samus_x or state.x),
    tostring(state.samus_y or state.y)
  )
end

local function hold(session, frames, names, reason)
  local i
  for i = 1, frames do
    session:step(names or {}, reason)
  end
end

function M.boot_spans()
  local spans = {
    { names = {}, frames = 2100, reason = "boot_title_wait" },
    { names = { "A" }, frames = 10, reason = "boot_title_confirm" },
    { names = {}, frames = 120, reason = "boot_file_menu_wait" },
    { names = { "A" }, frames = 10, reason = "boot_file_confirm" },
    { names = {}, frames = 300, reason = "boot_prologue_wait" },
    { names = { "A" }, frames = 10, reason = "boot_prologue_confirm" },
    { names = {}, frames = 30, reason = "boot_prologue_settle" },
  }
  local i
  for i = 1, 69 do
    spans[#spans + 1] = { names = { "A" }, frames = 10, reason = "boot_intro_mash" }
    spans[#spans + 1] = { names = {}, frames = 110, reason = "boot_intro_wait" }
  end
  return spans
end

function M.play_boot(session)
  local spans = M.boot_spans()
  local i
  for i = 1, #spans do
    local span = spans[i]
    hold(session, span.frames, span.names, span.reason)
  end
  local st = session.state
  if not (num(st.room_id) == M.ROOM_CERES_ELEVATOR and num(st.game_state) == 8) then
    error("boot missed first Ceres control: " .. brief(st))
  end
end

function M.play_boot_tas(session)
  local reached = false
  local i
  for i = 0, M.BOOT_MAX_FRAMES - 1 do
    local st = session.state
    if num(st.room_id) == M.ROOM_CERES_ELEVATOR and num(st.game_state) == 8 then
      reached = true
      break
    end
    if i < M.BOOT_MENU_MASH_FRAMES then
      local name
      if (i % 2) == 0 then
        name = "START"
      else
        name = "A"
      end
      session:step({ name }, "boot_menu_mash")
    elseif (i % 2) == 0 then
      session:step({ "A" }, "boot_cutscene_mash")
    else
      session:step({}, "boot_cutscene_wait")
    end
  end
  if not reached then
    error(
      "TAS boot missed Ceres control after "
        .. tostring(M.BOOT_MAX_FRAMES)
        .. "f: "
        .. brief(session.state)
    )
  end
  local waited
  local settled = false
  for waited = 0, 200 do
    local st = session.state
    local y = num(st.samus_y or st.y)
    local vy = i16(st.velocity_y)
    if
      num(st.room_id) == M.ROOM_CERES_ELEVATOR
      and num(st.game_state) == 8
      and y >= 60
      and math.abs(vy) <= 1
    then
      settled = true
      break
    end
    session:step({}, "boot_elev_settle")
  end
  if not settled then
    error("boot_elev_settle timed out at frame " .. tostring(session.frame) .. ": " .. brief(session.state))
  end
  hold(session, 4, {}, "boot_elev_plant")
end

function M.play(session)
  M.play_boot(session)
  local outbound = require("ceres.outbound")
  outbound.play_outbound_to_ridley(session)
  outbound.play_escape_to_landing(session)
end

return M
