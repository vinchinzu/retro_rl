-- Coroutine session so Python play_* ports stay imperative.
-- on_input: refresh WRAM, resume coroutine, apply joyset from yielded names.
-- Timeouts: error("TimeoutError: "..msg). No mailbox.

local ram = require("ram")
local input_mod = require("input")

local runtime = {}

local Session = {}
Session.__index = Session
runtime.Session = Session

function Session:refresh()
  self.state = ram.read()
  self.frame = self.state.frame or self.frame or 0
  return self.state
end

function Session:step(names, reason)
  names = names or {}
  self.info = self.info or {}
  self.info.reason = reason
  coroutine.yield(names, reason)
  return self.state
end

function Session:hold(n, names, reason)
  if type(names) == "string" and reason == nil then
    reason = names
    names = {}
  end
  names = names or {}
  n = tonumber(n) or 1
  local st = self.state
  for _ = 1, n do
    st = self:step(names, reason)
  end
  return st
end

function Session:idle(reason)
  return self:step({}, reason or "idle")
end

function Session:wait_until(pred, timeout, reason)
  timeout = timeout or 120
  reason = reason or "wait"
  for waited = 0, timeout do
    if pred(self.state) then
      return waited
    end
    if waited < timeout then
      self:idle(reason)
    end
  end
  error(
    "TimeoutError: " .. reason .. " timed out at frame " .. tostring(self.frame),
    2
  )
end

function Session:span(names, frames, reason)
  if type(names) == "table" and names.frames ~= nil then
    frames = names.frames
    reason = names.reason
    names = names.names
  end
  return self:hold(frames or 1, names or {}, reason)
end

function Session:spans(list)
  for i = 1, #list do
    local s = list[i]
    if type(s) == "table" and (s.names ~= nil or s.frames ~= nil) then
      self:span(s.names, s.frames, s.reason)
    else
      self:span(s[1], s[2], s[3])
    end
  end
end

function Session:raw_actions(list, reason)
  for i = 1, #list do
    self:step(input_mod.names_of(list[i]), reason)
  end
end

function runtime.new_session()
  local session = {
    state = nil,
    frame = 0,
    info = {},
  }
  setmetatable(session, Session)
  local ok, st = pcall(function()
    return ram.read()
  end)
  if ok then
    session.state = st
    session.frame = st.frame or 0
  else
    session.state = setmetatable({
      frame = 0,
      game_state = 0,
      room_id = 0,
      samus_x = 0,
      samus_y = 0,
      pose = 0,
    }, ram.State)
  end
  return session
end

local function apply_joyset(names)
  if not (input and input.joyset) then
    return
  end
  input.joyset(1, input_mod.joyset_table(names or {}))
end

-- Starts the coroutine. on_input resumes it and applies yielded names.
function runtime.run(session, fn)
  session._co = coroutine.create(function()
    fn(session)
  end)
  session._done = false
  session._error = nil

  local function on_input()
    session:refresh()
    if session._done then
      apply_joyset({})
      return
    end
    local ok, names, reason = coroutine.resume(session._co)
    if not ok then
      session._done = true
      session._error = names
      error(tostring(names), 0)
    end
    session.info = session.info or {}
    session.info.reason = reason
    if coroutine.status(session._co) == "dead" then
      session._done = true
    end
    apply_joyset(names or {})
  end

  if callback and callback.register then
    callback.register("input", on_input)
  else
    _G.on_input = on_input
  end
  return session
end

runtime.hold = function(session, n, names, reason)
  return session:hold(n, names, reason)
end

return runtime
