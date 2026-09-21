-- Morph bomb-roll left with y-band / pit recovery.
-- Python: routes/kpdr/brinstar/morph_bomb_roll.py

local ctrl = require("brinstar.ctrl")

local num = ctrl.num
local hold = ctrl.hold

local M = {}

function M.bomb_roll_left_safe(session, target_x, opts)
  opts = opts or {}
  local max_y = opts.max_y or 415
  local pit_y = opts.pit_y or 430
  local max_frames = opts.max_frames or 280
  local cycle_len = opts.cycle_len or 38
  local elev_y = opts.elev_y or 400
  local log_every = opts.log_every or 0
  local stall_frames = opts.stall_frames or 0

  local frames = 0
  local last_progress_x = num(session.state.samus_x)
  local frames_since_progress = 0
  local deep_pit_y = 445
  local pit_recoveries = 0
  local phase = "ON_BAND"

  local function log(tag)
    if log_every <= 0 then
      return
    end
    if (frames % log_every) ~= 0 and tag == "tick" then
      return
    end
    local s = session.state
    print(string.format(
      "[bomb_roll %s f=%s phase=%s] x=%s y=%s pose=%s vy=%s vx=%s morph=%s pb=%s",
      tag, tostring(frames), phase,
      tostring(s.samus_x), tostring(s.samus_y), tostring(s.pose),
      tostring(s.velocity_y), tostring(s.velocity_x),
      tostring(ctrl.is_morph(s.pose)), tostring(s.max_power_bombs)
    ))
  end

  while num(session.state.samus_x) > target_x and frames < max_frames do
    local s = session.state
    if num(s.max_power_bombs) > 0 then
      phase = "DONE"
      return s
    end
    log("tick")

    local falling_hard = num(s.velocity_y) > 80
    local in_pit = num(s.samus_y) > pit_y or falling_hard
    local deep_pit = num(s.samus_y) > deep_pit_y

    if in_pit then
      phase = "IN_PIT"
      pit_recoveries = pit_recoveries + 1
      phase = "RECOVERING"
      if deep_pit then
        hold(session, 4, {"UP"}, "pit_unmorph")
        local i
        for i = 1, 18 do
          hold(session, 1, {"RIGHT"}, "pit_right")
          if num(session.state.samus_y) <= pit_y then
            break
          end
        end
        if not ctrl.is_morph(session.state.pose)
            or num(session.state.samus_y) <= pit_y + 5 then
          hold(session, 6, {"A", "RIGHT"}, "pit_jump")
          hold(session, 8, {}, "pit_settle")
        end
        if not ctrl.is_morph(session.state.pose) then
          pcall(ctrl.ensure_morph, session)
        end
        frames = frames + 36
      else
        hold(session, 6, {"UP"}, "pit_unmorph")
        hold(session, 8, {"A", "RIGHT"}, "pit_jump")
        hold(session, 10, {}, "pit_settle")
        pcall(ctrl.ensure_morph, session)
        frames = frames + 24
      end
      if deep_pit and pit_recoveries >= 3
          and num(session.state.samus_x) >= last_progress_x - 2 then
        log("deep_pit_stuck")
        phase = "DONE"
        return session.state
      end
      phase = "ON_BAND"
    elseif not ctrl.is_morph(session.state.pose) then
      ctrl.ensure_morph(session)
      frames = frames + 20
    elseif stall_frames > 0 and frames_since_progress >= stall_frames then
      phase = "STALLED"
      hold(session, 2, {"X"}, "safe_watchdog_bomb")
      hold(session, 10, {}, "safe_watchdog_pause")
      frames = frames + 12
      frames_since_progress = 0
      log("watchdog")
      phase = "ON_BAND"
    else
      phase = "ON_BAND"
      hold(session, 2, {"X"}, "safe_bomb")
      frames = frames + 2
      local step
      for step = 0, cycle_len - 1 do
        if frames >= max_frames then
          break
        end
        s = session.state
        if num(s.max_power_bombs) > 0 then
          phase = "DONE"
          return s
        end
        if num(s.samus_x) <= target_x and num(s.samus_y) <= max_y + 5 then
          phase = "DONE"
          return s
        end
        if num(s.samus_y) > pit_y or num(s.velocity_y) > 80 then
          break
        end
        if not ctrl.is_morph(s.pose) then
          break
        end
        if num(s.samus_y) < elev_y
            or (num(s.samus_y) <= max_y and step > math.floor(cycle_len / 2)) then
          hold(session, 1, {"LEFT"}, "safe_roll")
        elseif num(s.samus_y) <= max_y + 8 then
          if step < 8 then
            hold(session, 1, {}, "safe_bomb_wait")
          else
            hold(session, 1, {"LEFT"}, "safe_roll")
          end
        else
          hold(session, 1, {}, "safe_band_wait")
        end
        frames = frames + 1
        frames_since_progress = frames_since_progress + 1
      end

      if num(session.state.samus_x) < last_progress_x - 2 then
        last_progress_x = num(session.state.samus_x)
        frames_since_progress = 0
      else
        hold(session, 6, {}, "safe_stall_pause")
        frames = frames + 6
        frames_since_progress = frames_since_progress + 6
      end
    end
  end
  log("done")
  phase = "DONE"
  return session.state
end

return M
