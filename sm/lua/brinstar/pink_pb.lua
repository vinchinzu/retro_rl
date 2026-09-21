-- Mission Impossible / Pink PB maze wall open and collect.
-- Python: routes/kpdr/brinstar/pink_pb.py

local ctrl = require("brinstar.ctrl")
local roll = require("brinstar.morph_bomb_roll")

local rooms = {}
pcall(function()
  rooms = require("rooms")
end)

local num = ctrl.num
local hold = ctrl.hold
local brief = ctrl.brief

local ROOM_PINK_PB = rooms.ROOM_PINK_PB or 0x9E11

local M = {}
M.ROOM_PINK_PB = ROOM_PINK_PB

function M.play_pink_pb_break_maze_wall(session, max_frames)
  max_frames = max_frames or 640
  ctrl.require_room(session, ROOM_PINK_PB, "pink_pb_break_maze_wall")
  if num(session.state.max_power_bombs) > 0 then
    return session.state
  end
  if num(session.state.samus_x) <= 410 and num(session.state.samus_y) <= 415 then
    return session.state
  end
  local i
  for i = 1, 40 do
    if num(session.state.samus_x) <= 440 then
      break
    end
    hold(session, 1, {"LEFT"}, "pb_maze_to_wall")
  end
  ctrl.ensure_morph(session)
  local state = roll.bomb_roll_left_safe(session, 410, {
    max_y = 412,
    pit_y = 430,
    max_frames = max_frames,
  })
  if num(state.max_power_bombs) > 0 then
    return state
  end
  if num(state.samus_x) > 420 then
    error("pink_pb_break_maze_wall: still blocked: " .. brief(session.state))
  end
  return state
end

function M.play_pink_pb_morph_bomb_collect(session, max_frames)
  max_frames = max_frames or 400
  ctrl.require_room(session, ROOM_PINK_PB, "pink_pb_morph_collect")
  if num(session.state.max_power_bombs) > 0 then
    return session.state
  end
  ctrl.ensure_morph(session)
  local state = roll.bomb_roll_left_safe(session, 100, {
    max_y = 415,
    pit_y = 430,
    max_frames = max_frames,
  })
  if num(state.max_power_bombs) > 0 then
    return state
  end
  if num(state.samus_x) <= 225 then
    hold(session, 5, {"UP"}, "pb_collect_unmorph")
    hold(session, 10, {}, "pb_collect_unmorph")
    local i
    for i = 1, 100 do
      state = hold(session, 1, {"LEFT"}, "pb_collect_walk")
      if num(state.max_power_bombs) > 0 then
        return state
      end
    end
    for i = 1, 40 do
      state = hold(session, 1, {"RIGHT"}, "pb_collect_walk_back")
      if num(state.max_power_bombs) > 0 then
        return state
      end
    end
  end
  if num(session.state.max_power_bombs) <= 0 then
    error("pink_pb_morph_collect: still 0 PB capacity: " .. brief(session.state))
  end
  return session.state
end

function M.play_pink_pb_from_left_zone(session)
  ctrl.require_room(session, ROOM_PINK_PB, "from_left_zone")
  if num(session.state.max_power_bombs) > 0 then
    return session.state
  end
  if num(session.state.samus_y) < 385 and num(session.state.samus_x) <= 230 then
    local i
    for i = 0, 79 do
      local d
      if math.floor(i / 8) % 2 == 0 then
        d = "LEFT"
      else
        d = "RIGHT"
      end
      local state = hold(session, 1, {d}, "pb_leftzone_drop")
      if num(state.samus_y) >= 385 or num(state.max_power_bombs) > 0 then
        break
      end
      if num(state.samus_x) > 230 then
        hold(session, 5, {"LEFT"}, "pb_leftzone_back")
      end
    end
  end
  if num(session.state.max_power_bombs) > 0 then
    return session.state
  end
  return M.play_pink_pb_morph_bomb_collect(session)
end

function M.play_pink_pb_mid_maze_to_collect(session, max_frames, log_every)
  max_frames = max_frames or 500
  log_every = log_every or 0
  ctrl.require_room(session, ROOM_PINK_PB, "mid_maze")
  if num(session.state.max_power_bombs) > 0 then
    return session.state
  end
  if num(session.state.samus_x) <= 230 and num(session.state.samus_y) <= 420 then
    if num(session.state.samus_y) < 385 and num(session.state.samus_x) <= 220 then
      return M.play_pink_pb_from_left_zone(session)
    end
    return M.play_pink_pb_morph_bomb_collect(session)
  end
  ctrl.ensure_morph(session)
  local start_x = num(session.state.samus_x)
  local start_y = num(session.state.samus_y)
  roll.bomb_roll_left_safe(session, 225, {
    max_y = 412,
    pit_y = 420,
    max_frames = max_frames,
    elev_y = 400,
    log_every = log_every,
    stall_frames = 50,
  })
  local s = session.state
  if num(s.max_power_bombs) > 0 then
    return s
  end
  if num(s.samus_x) <= 230 and num(s.samus_y) <= 420 then
    return M.play_pink_pb_from_left_zone(session)
  end
  local pit_note = ""
  if num(s.samus_y) > 440 then
    pit_note = " deep-pit trap y≈457 (rollable under mid but ~2px headroom — no climb to item band y≈360–395);"
  end
  error(string.format(
    "pink_pb_mid_maze: no pure path yet (start=(%s,%s) → x=%s y=%s pose=%s);%s mid solid at band — need door→left-volume or top→crumble (see Mission Impossible Room)",
    tostring(start_x),
    tostring(start_y),
    tostring(s.samus_x),
    tostring(s.samus_y),
    tostring(s.pose),
    pit_note
  ))
end

return M
