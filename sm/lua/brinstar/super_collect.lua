-- Super Missile room collect and exit to farming / Big Pink.
-- Python: routes/kpdr/brinstar/super_collect.py

local ctrl = require("brinstar.ctrl")

local rooms = {}
pcall(function()
  rooms = require("rooms")
end)

local num = ctrl.num
local hold = ctrl.hold
local brief = ctrl.brief

local ROOM_SUPER = rooms.ROOM_SUPER or 0x9B5B
local ROOM_FARMING = rooms.ROOM_FARMING or 0xA0A4
local ROOM_BIG_PINK = rooms.ROOM_BIG_PINK or 0x9D19

local M = {}
M.ROOM_SUPER = ROOM_SUPER
M.ROOM_FARMING = ROOM_FARMING
M.ROOM_BIG_PINK = ROOM_BIG_PINK

function M.play_super_room_collect(session)
  local entry_frame = session.frame
  local state = session.state
  if num(state.room_id) ~= ROOM_SUPER then
    error(string.format(
      "Super collect entry: room 0x%04X != 0x%04X",
      num(state.room_id),
      ROOM_SUPER
    ))
  end
  if num(state.max_super_missiles) > 0 then
    error("Super collect entry: supers already collected")
  end

  local i
  for i = 1, 80 do
    state = hold(session, 1, {"RIGHT", "B"}, "super_shaft_approach")
    if num(state.samus_x) >= 140 then
      break
    end
  end

  for i = 1, 15 do
    hold(session, 3, {"DOWN", "X"}, "super_shaft_shot")
    hold(session, 2, {"DOWN"}, "super_shaft_shot")
  end
  hold(session, 10, {"DOWN"}, "super_shaft_morph")
  for i = 1, 8 do
    hold(session, 2, {"A"}, "super_shaft_bomb")
    hold(session, 30, {}, "super_shaft_bomb_wait")
  end

  for i = 1, 50 do
    state = hold(session, 2, {"RIGHT", "A", "B"}, "super_shaft_jump")
    if num(state.samus_x) > 250 then
      break
    end
  end

  for i = 0, 199 do
    local phase = i % 10
    if phase < 3 then
      state = hold(session, 4, {"RIGHT", "B", "X"}, "super_shaft_explore")
    elseif phase < 5 then
      state = hold(session, 4, {"RIGHT", "A", "B"}, "super_shaft_explore")
    elseif phase < 7 then
      state = hold(session, 3, {"DOWN", "X"}, "super_shaft_explore")
    elseif phase < 8 then
      hold(session, 8, {"DOWN"}, "super_shaft_explore")
      hold(session, 2, {"A"}, "super_shaft_explore")
      state = hold(session, 20, {}, "super_shaft_explore")
    else
      state = hold(session, 4, {"RIGHT", "B"}, "super_shaft_explore")
    end
    if num(state.samus_y) > 500 then
      break
    end
  end

  for i = 1, 800 do
    state = hold(session, 2, {}, "super_shaft_fall")
    if num(state.samus_y) > 2100 then
      break
    end
  end
  if num(state.samus_y) <= 2000 then
    error("Super shaft fall failed: " .. brief(state))
  end

  local collect_frame = nil
  for i = 0, 399 do
    if num(state.samus_x) < 412 then
      state = hold(session, 2, {"RIGHT", "B"}, "super_item_approach")
    elseif num(state.samus_x) > 428 then
      state = hold(session, 2, {"LEFT", "B"}, "super_item_approach")
    else
      state = hold(session, 2, {}, "super_item_approach")
    end
    if (i % 12) == 0 then
      state = hold(session, 4, {"X"}, "super_item_shoot")
    end
    if (i % 40) == 20 then
      state = hold(session, 6, {"A"}, "super_item_jump")
    end
    if num(state.max_super_missiles) > 0 then
      collect_frame = session.frame
      break
    end
  end
  if collect_frame == nil or num(state.max_super_missiles) <= 0 then
    error("Super Missile PLM never collected: " .. brief(state))
  end

  for i = 0, 299 do
    state = hold(session, 1, {}, "super_item_fanfare")
    if num(state.game_state) == 8
        and ctrl.is_ordinary(state)
        and num(state.max_super_missiles) > 0
        and i > 80 then
      break
    end
  end

  return {
    entry_frame = entry_frame,
    collect_frame = collect_frame,
    exit_frame = nil,
    max_super_missiles = num(state.max_super_missiles),
    final_room_id = num(state.room_id),
    samus_x = num(state.samus_x),
    samus_y = num(state.samus_y),
  }
end

function M.play_super_room_to_farming(session)
  local state = session.state
  ctrl.require_room(session, ROOM_SUPER, "to_farming")
  if num(state.max_super_missiles) <= 0 then
    error("to_farming requires Super Missiles")
  end
  ctrl.try_select(session, 1)

  local i
  for i = 1, 80 do
    state = hold(session, 2, {"LEFT", "B"}, "super_gate_approach")
    if num(state.samus_x) <= 320 then
      break
    end
  end

  hold(session, 12, {"DOWN"}, "super_gate_bomb")
  for i = 1, 15 do
    hold(session, 2, {"A"}, "super_gate_bomb")
    hold(session, 35, {}, "super_gate_bomb")
    state = hold(session, 4, {"LEFT"}, "super_gate_bomb")
    if num(state.samus_x) < 200 then
      break
    end
  end
  ctrl.unmorph(session)

  for i = 1, 120 do
    state = hold(session, 2, {"LEFT", "B"}, "super_door_approach")
    if num(state.samus_x) < 50 then
      break
    end
  end

  for i = 1, 50 do
    hold(session, 3, {"LEFT", "X"}, "super_door_shot")
    state = hold(session, 5, {"LEFT", "B"}, "super_door_enter")
    if num(state.room_id) == ROOM_FARMING then
      break
    end
  end
  for i = 1, 200 do
    state = hold(session, 1, {}, "farming_settle")
    if num(state.room_id) == ROOM_FARMING
        and num(state.game_state) == 8
        and ctrl.is_ordinary(state) then
      break
    end
  end
  ctrl.require_room(session, ROOM_FARMING, "farming entry")
  return state
end

function M.play_farming_to_big_pink(session)
  local state = session.state
  ctrl.require_room(session, ROOM_FARMING, "farming_to_pink")
  ctrl.try_select(session, 2)
  ctrl.unmorph(session)

  local i
  for i = 0, 499 do
    if session.state.pose == 39 or session.state.pose == 40
        or session.state.pose == 137 or session.state.pose == 138 then
      ctrl.unmorph(session)
    end
    state = hold(session, 3, {"LEFT", "A", "B"}, "farming_cross")
    if (i % 5) == 0 then
      hold(session, 2, {"LEFT", "X"}, "farming_super")
    end
    if (i % 25) == 12 then
      hold(session, 8, {"DOWN"}, "farming_bomb")
      hold(session, 2, {"A"}, "farming_bomb")
      hold(session, 30, {}, "farming_bomb")
      ctrl.unmorph(session)
    end
    if num(state.room_id) == ROOM_BIG_PINK then
      break
    end
  end
  for i = 1, 150 do
    state = hold(session, 1, {}, "big_pink_settle")
    if num(state.room_id) == ROOM_BIG_PINK
        and num(state.game_state) == 8
        and ctrl.is_ordinary(state) then
      break
    end
  end
  ctrl.require_room(session, ROOM_BIG_PINK, "big pink entry")
  return state
end

function M.play_post_spore_supers(session, opts)
  opts = opts or {}
  local continue_to_farming = opts.continue_to_farming
  if continue_to_farming == nil then
    continue_to_farming = true
  end
  local continue_to_big_pink = opts.continue_to_big_pink
  if continue_to_big_pink == nil then
    continue_to_big_pink = true
  end
  local continue_to_crest = opts.continue_to_crest or false
  local evidence = M.play_super_room_collect(session)
  if continue_to_farming then
    M.play_super_room_to_farming(session)
    if continue_to_big_pink then
      M.play_farming_to_big_pink(session)
      if continue_to_crest then
        local pink = require("brinstar.pink_shaft")
        pink.play_big_pink_crest_pocket(session)
      end
    end
    local state = session.state
    return {
      entry_frame = evidence.entry_frame,
      collect_frame = evidence.collect_frame,
      exit_frame = session.frame,
      max_super_missiles = num(state.max_super_missiles),
      final_room_id = num(state.room_id),
      samus_x = num(state.samus_x),
      samus_y = num(state.samus_y),
    }
  end
  return evidence
end

return M
