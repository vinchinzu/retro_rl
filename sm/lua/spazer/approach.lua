-- Below Spazer solid top → Super green door → Spazer Room.
-- Port of snes/super_metroid/routes/kpdr/spazer/approach.py.

local geo = require("red_tower.ctrl")
local g = require("spazer.geometry")
local climb = require("spazer.climb")
local scripts = require("spazer.scripts")

local M = {}

function M.approach_super_door_from_top(session)
  if g.on_super_door_approach(session.state) then
    return session.state
  end
  if g.is_lag_pose(session.state) then
    geo.break_rle_lag(session)
  end

  if session.state.samus_x < g.DOOR_X_MIN then
    geo.play_script(session, scripts.TOP_DOOR_APPROACH_RLE, {
      reason = "spazer_top_door_tunnel",
      room_id = geo.ROOM_BELOW_SPAZER,
      stop_when = g.on_super_door_approach,
      on_lag = "ignore",
    })
  end

  if not g.on_super_door_approach(session.state) then
    geo.unmorph(session)
    for _ = 1, 80 do
      if not g.in_below_spazer(session.state) then
        return session.state
      end
      if g.on_super_door_approach(session.state) then
        return session.state
      end
      geo.hold(session, 1, {"RIGHT", "B"}, "spazer_door_approach")
    end
  end
  return session.state
end

function M.play_below_spazer_to_spazer(session)
  geo.require_room(session, geo.ROOM_BELOW_SPAZER, "below_spazer_to_spazer")
  geo.unmorph(session)
  geo.break_rle_lag(session)

  if not (g.on_solid_top(session.state) or g.on_super_door_approach(session.state)) then
    climb.play_below_spazer_climb(session)
    geo.break_rle_lag(session)
    if not g.solid_ish_top(session.state) then
      error("below_spazer_to_spazer: climb failed solid top: "
        .. geo.fmt_state(session.state))
    end
  end

  M.approach_super_door_from_top(session)
  if not g.on_super_door_approach(session.state) then
    error("below_spazer_to_spazer: missed green-door lip: "
      .. geo.fmt_state(session.state))
  end

  geo.hold(session, 10, {}, "spazer_door_settle")
  geo.select_weapon(session, 2)
  geo.hold(session, 6, {}, "spazer_super_ready")
  geo.hold(session, 3, {"RIGHT"}, "spazer_face_door")
  geo.hold(session, 3, {}, "spazer_face_door_release")
  geo.hold(session, 8, {"X"}, "spazer_green_door_super")
  geo.hold(session, 50, {}, "spazer_green_door_fuse")
  local entered = false
  for _ = 1, 250 do
    local state = geo.hold(session, 1, {"RIGHT", "B"}, "spazer_enter")
    if state.room_id == geo.ROOM_SPAZER then
      entered = true
      break
    end
  end
  if not entered then
    error("below_spazer_to_spazer: green door did not open: "
      .. geo.fmt_state(session.state))
  end
  return geo.wait_ordinary_room(session, geo.ROOM_SPAZER, {
    settle_frames = 120,
    label = "below_spazer_to_spazer",
  })
end

return M
