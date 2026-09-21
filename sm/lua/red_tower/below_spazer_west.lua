-- Below Spazer floor/water → West Tunnel (RIGHT runner only).
-- Port of snes/super_metroid/routes/kpdr/red_tower/below_spazer_west.py.

local geo = require("red_tower.ctrl")

local M = {}

function M.play_below_spazer_floor_to_west(session)
  geo.require_room(session, geo.ROOM_BELOW_SPAZER, "below_spazer_floor_to_west")
  geo.hold(session, 6, {}, "below_spazer_entry_glide")
  geo.unmorph(session)
  geo.select_weapon(session, 0)
  local state
  local reached = false
  for frame = 0, 1999 do
    local names
    if frame % 35 < 10 then
      names = {"RIGHT", "B", "X"}
    else
      names = {"RIGHT", "B", "A"}
    end
    state = geo.hold(session, 1, names, "below_spazer_right")
    if state.room_id == geo.ROOM_WEST_TUNNEL then
      reached = true
      break
    end
  end
  if not reached then
    error("below_spazer_floor_to_west: West Tunnel not reached: "
      .. geo.fmt_state(session.state))
  end
  return geo.wait_ordinary_room(session, geo.ROOM_WEST_TUNNEL, {
    settle_frames = 260,
    label = "below_spazer_floor_to_west",
  })
end

return M
