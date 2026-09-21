-- In-room takeoff windows. ready() is the DoorKinematicsRequirement subset:
-- x in range, x_sub in range, facing in set, abs(momentum_x) >= min_momentum.

local ram = require("ram")

local takeoff = {}

takeoff.DEFAULT_PUMP_PERIOD = 2

local SIDES = { LEFT = true, RIGHT = true }
local SHOULDER = { L = true, R = true }

local function facing_for_side(side)
  if side == "RIGHT" then
    return { [ram.FACING_RIGHT] = true }
  end
  if side == "LEFT" then
    return { [ram.FACING_LEFT] = true }
  end
  error("takeoff side must be LEFT or RIGHT, got " .. tostring(side))
end

local function range_pair(raw, k1, k2)
  if type(raw) ~= "table" then
    return nil
  end
  for _, key in ipairs({ k1, k2 }) do
    local val = raw[key]
    if type(val) == "table" and val[1] ~= nil and val[2] ~= nil then
      return { tonumber(val[1]), tonumber(val[2]) }
    end
  end
  if raw[1] ~= nil and raw[2] ~= nil and k1 == nil then
    return { tonumber(raw[1]), tonumber(raw[2]) }
  end
  return nil
end

local function facing_set_of(win)
  if win.facings ~= nil then
    local src = win.facings
    if src[1] ~= nil then
      local s = {}
      for i = 1, #src do
        s[tonumber(src[i])] = true
      end
      return s
    end
    return src
  end
  return facing_for_side(win.side)
end

local TakeoffWindow = {}
TakeoffWindow.__index = TakeoffWindow
takeoff.TakeoffWindow = TakeoffWindow

local function check_side(side)
  if SHOULDER[side] then
    error("takeoff side is D-pad LEFT/RIGHT, not shoulder L/R; got " .. tostring(side))
  end
  if not SIDES[side] then
    error("takeoff side must be LEFT or RIGHT, got " .. tostring(side))
  end
end

function takeoff.new_window(opts)
  opts = opts or {}
  local side = opts.side or "RIGHT"
  check_side(side)
  local x_range = opts.x_range or opts.xRange
  if type(x_range) ~= "table" or x_range[1] == nil then
    error("TakeoffWindow missing x range")
  end
  if x_range[1] > x_range[2] then
    error("takeoff x_range inverted")
  end
  local x_sub = opts.x_sub_range or opts.xSubRange or { 0, 65535 }
  local win = {
    x_range = { tonumber(x_range[1]), tonumber(x_range[2]) },
    side = side,
    x_sub_range = { tonumber(x_sub[1]) or 0, tonumber(x_sub[2]) or 65535 },
    min_momentum = tonumber(opts.min_momentum or opts.minMomentum or 1) or 1,
    pump = opts.pump,
    release_vy = tonumber(opts.release_vy or opts.releaseVy or 0) or 0,
    facings = opts.facings,
  }
  if win.pump == nil then
    win.pump = true
  end
  return setmetatable(win, TakeoffWindow)
end

function TakeoffWindow:facing_set()
  return facing_set_of(self)
end

function TakeoffWindow:ready(state)
  local x = tonumber(state.samus_x) or 0
  local x_sub = tonumber(state.samus_x_sub) or 0
  local facing = tonumber(state.facing) or 0
  local mx = tonumber(state.momentum_x) or 0
  if mx < 0 then
    mx = -mx
  end
  if x < self.x_range[1] or x > self.x_range[2] then
    return false
  end
  if x_sub < self.x_sub_range[1] or x_sub > self.x_sub_range[2] then
    return false
  end
  if not facing_set_of(self)[facing] then
    return false
  end
  return mx >= self.min_momentum
end

function TakeoffWindow:to_dict()
  local payload = {
    xRange = { self.x_range[1], self.x_range[2] },
    side = self.side,
    xSubRange = { self.x_sub_range[1], self.x_sub_range[2] },
    minMomentum = self.min_momentum,
    pump = self.pump and true or false,
    releaseVy = self.release_vy,
  }
  if self.facings ~= nil then
    payload.facings = self.facings
  end
  return payload
end

function takeoff.window_from_dict(raw)
  local x_range = range_pair(raw, "xRange", "x_range")
  if x_range == nil and raw.x_jump_lo ~= nil then
    x_range = { tonumber(raw.x_jump_lo), tonumber(raw.x_jump_hi) }
  end
  if x_range == nil then
    error("TakeoffWindow missing x range")
  end
  local x_sub = range_pair(raw, "xSubRange", "x_sub_range")
  if x_sub == nil and raw.x_sub_lo ~= nil then
    x_sub = { tonumber(raw.x_sub_lo or 0), tonumber(raw.x_sub_hi or 65535) }
  end
  return takeoff.new_window({
    x_range = x_range,
    side = raw.side or "RIGHT",
    x_sub_range = x_sub or { 0, 65535 },
    min_momentum = raw.minMomentum or raw.min_momentum or 1,
    pump = raw.pump,
    release_vy = raw.releaseVy or raw.release_vy or 0,
    facings = raw.facings,
  })
end

TakeoffWindow.from_dict = function(_, raw)
  return takeoff.window_from_dict(raw)
end

local PlatformHop = {}
PlatformHop.__index = PlatformHop
takeoff.PlatformHop = PlatformHop

function takeoff.new_hop(opts)
  opts = opts or {}
  local hop = {
    y = tonumber(opts.y) or 0,
    x_lo = tonumber(opts.x_lo or opts.xLo) or 0,
    x_hi = tonumber(opts.x_hi or opts.xHi) or 0,
    takeoff = opts.takeoff,
  }
  if getmetatable(hop.takeoff) ~= TakeoffWindow and type(hop.takeoff) == "table" then
    if hop.takeoff.x_range then
      hop.takeoff = takeoff.new_window(hop.takeoff)
    else
      hop.takeoff = takeoff.window_from_dict(hop.takeoff)
    end
  end
  hop.side = hop.takeoff and hop.takeoff.side
  return setmetatable(hop, PlatformHop)
end

-- Positional constructors used by ceres/geometry.lua.
function takeoff.takeoff_window(x_range, side, opts)
  opts = opts or {}
  local merged = {}
  for k, v in pairs(opts) do
    merged[k] = v
  end
  merged.x_range = x_range
  merged.side = side or merged.side
  return takeoff.new_window(merged)
end

function takeoff.platform_hop(y, x_lo, x_hi, window)
  return takeoff.new_hop({
    y = y,
    x_lo = x_lo,
    x_hi = x_hi,
    takeoff = window,
  })
end

function PlatformHop:ready(state)
  return self.takeoff:ready(state)
end

function PlatformHop:covers_y(y, slack)
  slack = slack or 16
  local d = (tonumber(y) or 0) - self.y
  if d < 0 then
    d = -d
  end
  return d <= slack
end

function PlatformHop:at_ledge_end(x, slack)
  slack = slack or 12
  x = tonumber(x) or 0
  if self.takeoff.side == "RIGHT" then
    return x >= self.x_hi - slack
  end
  return x <= self.x_lo + slack
end

function PlatformHop:with_takeoff(kwargs)
  local merged = {}
  for k, v in pairs(self.takeoff) do
    merged[k] = v
  end
  for k, v in pairs(kwargs or {}) do
    merged[k] = v
  end
  return takeoff.new_hop({
    y = self.y,
    x_lo = self.x_lo,
    x_hi = self.x_hi,
    takeoff = takeoff.new_window(merged),
  })
end

function PlatformHop:to_dict()
  return {
    y = self.y,
    xLo = self.x_lo,
    xHi = self.x_hi,
    takeoff = self.takeoff:to_dict(),
  }
end

function takeoff.hop_from_dict(raw)
  local nested = raw.takeoff
  local win
  if type(nested) == "table" then
    win = takeoff.window_from_dict(nested)
  else
    win = takeoff.window_from_dict(raw)
  end
  local x_lo = raw.xLo or raw.x_lo
  local x_hi = raw.xHi or raw.x_hi
  if x_lo == nil or x_hi == nil then
    error("PlatformHop missing x_lo/x_hi")
  end
  return takeoff.new_hop({
    y = raw.y,
    x_lo = x_lo,
    x_hi = x_hi,
    takeoff = win,
  })
end

PlatformHop.from_dict = function(_, raw)
  return takeoff.hop_from_dict(raw)
end

function takeoff.hop_for_y(y, hops, slack)
  slack = slack or 16
  for i = 1, #hops do
    if hops[i]:covers_y(y, slack) then
      return hops[i]
    end
  end
  return nil
end

function takeoff.next_hop_above(y, hops, slack)
  slack = slack or 16
  y = tonumber(y) or 0
  local best = nil
  for i = 1, #hops do
    local hop = hops[i]
    if hop.y < y - slack then
      if best == nil or hop.y > best.y then
        best = hop
      end
    end
  end
  return best
end

function takeoff.walk_toward_x(x, target, slack)
  slack = slack or 6
  x = tonumber(x) or 0
  target = tonumber(target) or 0
  if x > target + slack then
    return { "LEFT" }
  end
  if x < target - slack then
    return { "RIGHT" }
  end
  return {}
end

function takeoff.spin_jump(side)
  check_side(side)
  return { side, "B", "A" }
end

function takeoff.shoulder_pump_button(i, period)
  period = period or takeoff.DEFAULT_PUMP_PERIOD
  if period < 1 then
    period = 1
  end
  if math.floor((tonumber(i) or 0) / period) % 2 == 0 then
    return "L"
  end
  return "R"
end

function takeoff.approach_window(state, hop, pump_i, period)
  period = period or takeoff.DEFAULT_PUMP_PERIOD
  if period < 1 then
    period = 1
  end
  pump_i = tonumber(pump_i) or 0
  local side = hop.takeoff.side
  local facing = tonumber(state.facing) or 0
  if not facing_set_of(hop.takeoff)[facing] then
    return { side, "B" }, pump_i + 1
  end
  local mx = tonumber(state.momentum_x) or 0
  if mx < 0 then
    mx = -mx
  end
  local running = (tonumber(state.speed_flag) or 0) ~= 0 or mx >= 1
  if hop.takeoff.pump and running then
    return { side, "B", takeoff.shoulder_pump_button(pump_i, period) }, pump_i + 1
  end
  return { side, "B" }, pump_i + 1
end

function takeoff.should_release_over(state, nxt, release_vy, slack, x_pad)
  if nxt == nil then
    return false
  end
  release_vy = release_vy or 0
  slack = slack or 20
  x_pad = x_pad or 8
  local y = tonumber(state.samus_y) or 0
  local x = tonumber(state.samus_x) or 0
  local vy = tonumber(state.velocity_y) or 0
  local dy = y - nxt.y
  if dy < 0 then
    dy = -dy
  end
  return dy <= slack
    and (nxt.x_lo - x_pad) <= x
    and x <= (nxt.x_hi + x_pad)
    and vy >= release_vy
end

return takeoff
