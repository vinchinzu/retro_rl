-- Classic L↔R arm-pump helpers and knockback recovery for Ceres.
--
-- YT reference (Kentroid TFsGVxQReMw chunk k0_ceres) lives at
-- policies/early_game/ceres_kentroid_spans.json + gitignored
-- refs/yt_reference/.../chunks/k0_ceres/. Absolute Input Display replay
-- desyncs on elevator lag / magnet geometry — do not restore fixed product
-- open-loop when a later leg breaks. Speed every section; re-solve tails with
-- room / pose / y / knockback reads (same idea as K4 knockback skills).
--
-- Classic arm-pump: dir+B with L↔R angle spam (runway_dash period-2).

local ram = require("ram")
local takeoff = require("takeoff")
local knockback = require("skills.knockback")
local geom = require("ceres.geometry")

local GS_ORDINARY = ram.GS_ORDINARY or 8

local M = {}

-- Expand dir+B into classic L↔R arm-pump (runway_dash pattern).
function M.arm_pump_dash_spans(direction, frames, reason, period)
  period = math.max(1, period or geom._CERES_ARM_PUMP_PERIOD or 2)
  local out = {}
  local i = 0
  while i < frames do
    local ang = takeoff.shoulder_pump_button(i, period)
    local chunk = math.min(period, frames - i)
    out[#out + 1] = {
      names = {direction, "B", ang},
      frames = chunk,
      reason = reason,
    }
    i = i + chunk
  end
  return out
end

-- Spin-escape knockback using WRAM pose (no fixed open-loop restore).
-- 6f dir+B, then dir+B+A, while still in knockback.
function M.clear_knockback(session, direction, reason, max_frames)
  max_frames = max_frames or 40
  for i = 0, max_frames - 1 do
    if not knockback.is_knockback(session.state) then
      return
    end
    -- Short run then spin in travel direction.
    if i < 6 then
      session:step({direction, "B"}, reason .. "_kb_run")
    else
      session:step({direction, "B", "A"}, reason .. "_kb_spin")
    end
  end
end

function M.wait_ordinary(session, room_id, reason, timeout)
  timeout = timeout or 200
  session:wait_until(function(s)
    return s.room_id == room_id and s.game_state == GS_ORDINARY
  end, timeout, reason)
end

return M
