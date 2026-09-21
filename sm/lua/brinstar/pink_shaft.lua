-- Big Pink crest, Super-block clear, morph tunnel, and main shaft.
-- Python: routes/kpdr/brinstar/pink_shaft.py

local ctrl = require("brinstar.ctrl")

local rooms = {}
pcall(function()
  rooms = require("rooms")
end)

local num = ctrl.num
local hold = ctrl.hold
local brief = ctrl.brief

local ROOM_BIG_PINK = rooms.ROOM_BIG_PINK or 0x9D19

local M = {}
M.ROOM_BIG_PINK = ROOM_BIG_PINK

function M.play_big_pink_crest_pocket(session)
  local state = session.state
  ctrl.require_room(session, ROOM_BIG_PINK, "crest_pocket")
  ctrl.unmorph(session)
  ctrl.try_select(session, 2)

  local i
  for i = 1, 40 do
    state = hold(session, 1, {"LEFT", "B"}, "big_pink_pocket_approach")
    if num(state.samus_x) <= 1160 then
      break
    end
  end
  hold(session, 25, {"RIGHT", "B"}, "big_pink_pocket_runup")

  for i = 1, 40 do
    state = hold(session, 12, {"LEFT", "A", "B"}, "big_pink_pocket_crest")
    state = hold(session, 6, {"LEFT", "B"}, "big_pink_pocket_crest")
    if num(state.samus_x) <= 1135 then
      break
    end
  end
  if num(state.samus_x) > 1135 then
    error(string.format(
      "Big Pink pocket crest failed (still x=%s, y=%s)",
      tostring(state.samus_x),
      tostring(state.samus_y)
    ))
  end
  hold(session, 12, {}, "big_pink_pocket_crest_settle")
  return session.state
end

function M.play_big_pink_clear_super_block(session)
  local state = session.state
  ctrl.require_room(session, ROOM_BIG_PINK, "clear_super_block")
  hold(session, 4, {"A"}, "big_pink_unspin")
  hold(session, 40, {}, "big_pink_unspin_settle")
  ctrl.try_select(session, 2)
  hold(session, 15, {"DOWN"}, "big_pink_crouch_super")
  local i
  for i = 1, 8 do
    hold(session, 3, {"LEFT", "X"}, "big_pink_crouch_super")
    state = hold(session, 18, {"LEFT", "DOWN"}, "big_pink_crouch_super")
  end
  hold(session, 10, {}, "big_pink_super_block_settle")
  return session.state
end

function M.play_big_pink_morph_to_tunnel(session)
  ctrl.require_room(session, ROOM_BIG_PINK, "morph_to_tunnel")
  hold(session, 12, {}, "big_pink_morph_stand")
  hold(session, 1, {"DOWN"}, "big_pink_morph_tap1")
  hold(session, 4, {}, "big_pink_morph_gap")
  hold(session, 1, {"DOWN"}, "big_pink_morph_tap2")
  local state = hold(session, 18, {"DOWN"}, "big_pink_morph_hold")
  local on_tunnel = 1395 <= num(state.samus_y) and num(state.samus_y) <= 1410
    and num(state.samus_x) <= 1155
  if not on_tunnel then
    hold(session, 8, {}, "big_pink_morph_retry_stand")
    hold(session, 2, {"DOWN"}, "big_pink_morph_retry_tap1")
    hold(session, 4, {}, "big_pink_morph_retry_gap")
    hold(session, 8, {"DOWN"}, "big_pink_morph_retry_tap2")
    state = hold(session, 20, {"DOWN"}, "big_pink_morph_retry_hold")
    on_tunnel = 1395 <= num(state.samus_y) and num(state.samus_y) <= 1410
      and num(state.samus_x) <= 1155
  end
  if not on_tunnel then
    error(string.format(
      "Big Pink morph-to-tunnel failed: (%s, %s) pose=%s; expected morph on raised floor y≈1401 after double-tap DOWN",
      tostring(state.samus_x),
      tostring(state.samus_y),
      tostring(state.pose)
    ))
  end
  return session.state
end

function M.play_big_pink_tunnel_west(session, target_x, max_frames)
  target_x = target_x or 750
  max_frames = max_frames or 400
  local state = session.state
  ctrl.require_room(session, ROOM_BIG_PINK, "tunnel_west")
  ctrl.try_select(session, 0)
  hold(session, 8, {"DOWN"}, "big_pink_tunnel_morph")

  local i
  for i = 0, max_frames - 1 do
    state = hold(session, 1, {"LEFT", "B"}, "big_pink_tunnel_roll")
    if (i % 18) == 5 then
      hold(session, 2, {"X"}, "big_pink_tunnel_bomb")
      hold(session, 50, {}, "big_pink_tunnel_bomb_wait")
    end
    if num(state.room_id) ~= ROOM_BIG_PINK then
      error(string.format(
        "tunnel west left Big Pink at frame %s: %s",
        tostring(session.frame),
        brief(state)
      ))
    end
    if num(state.samus_x) <= target_x then
      hold(session, 10, {}, "big_pink_tunnel_west_settle")
      return session.state
    end
  end
  error(string.format(
    "Big Pink tunnel west failed: x=%s, y=%s (target x≤%s)",
    tostring(state.samus_x),
    tostring(state.samus_y),
    tostring(target_x)
  ))
end

function M.play_big_pink_drop_to_pocket(session)
  ctrl.require_room(session, ROOM_BIG_PINK, "drop_to_pocket")
  hold(session, 10, {}, "big_pink_pocket_drop_settle")
  hold(session, 4, {"A"}, "big_pink_pocket_drop_unspin")
  hold(session, 25, {}, "big_pink_pocket_drop_unspin")
  hold(session, 40, {"RIGHT", "B"}, "big_pink_pocket_drop")
  hold(session, 50, {}, "big_pink_pocket_drop_land")
  ctrl.unmorph(session)
  return session.state
end

function M.play_big_pink_bomb_to_walkway_edge(session, fuse_frames, jump_frames)
  fuse_frames = fuse_frames or 15
  jump_frames = jump_frames or 14
  local state = session.state
  ctrl.require_room(session, ROOM_BIG_PINK, "bomb_to_edge")
  local i
  for i = 1, 60 do
    if num(state.samus_x) <= 1168 then
      break
    end
    state = hold(session, 1, {"LEFT", "B"}, "big_pink_edge_approach")
  end
  ctrl.try_select(session, 0)
  hold(session, 12, {"DOWN"}, "big_pink_edge_morph")
  hold(session, 3, {"LEFT"}, "big_pink_edge_press")
  hold(session, 2, {"X"}, "big_pink_edge_bomb")
  hold(session, fuse_frames, {"LEFT"}, "big_pink_edge_fuse")
  hold(session, jump_frames, {"LEFT", "A"}, "big_pink_edge_boost")
  for i = 1, 50 do
    state = hold(session, 1, {}, "big_pink_edge_land")
  end
  hold(session, 10, {}, "big_pink_edge_settle")
  return session.state
end

function M.play_big_pink_into_main_shaft(session, target_x)
  target_x = target_x or 750
  M.play_big_pink_crest_pocket(session)
  M.play_big_pink_clear_super_block(session)
  M.play_big_pink_morph_to_tunnel(session)
  return M.play_big_pink_tunnel_west(session, target_x)
end

return M
