-- Green Hill Zone → Noob Bridge → Red Tower.
-- Python: routes/kpdr/brinstar/ghz_to_red.py

local ctrl = require("brinstar.ctrl")

local rooms = {}
pcall(function()
  rooms = require("rooms")
end)

local num = ctrl.num
local hold = ctrl.hold
local brief = ctrl.brief

local ROOM_GHZ = rooms.ROOM_GHZ or 0x9E52
local ROOM_NOOB = rooms.ROOM_NOOB or 0x9FBA
local ROOM_RED_TOWER = rooms.ROOM_RED_TOWER or 0xA253

local M = {}
M.ROOM_GHZ = ROOM_GHZ
M.ROOM_NOOB = ROOM_NOOB
M.ROOM_RED_TOWER = ROOM_RED_TOWER

function M.play_ghz_to_noob(session)
  ctrl.require_room(session, ROOM_GHZ, "ghz_to_noob")
  ctrl.try_select(session, 0)

  local reached_pillar = false
  local i
  for i = 1, 120 do
    local state = session.state
    if num(state.room_id) == ROOM_NOOB then
      break
    end
    if num(state.samus_x) >= 1445 and num(state.samus_y) >= 885 then
      reached_pillar = true
      break
    end
    if num(state.samus_y) < 700 then
      hold(session, 8, {"RIGHT", "B"}, "ghz_run")
      hold(session, 4, {"RIGHT", "B", "A"}, "ghz_hop")
      if num(state.samus_x) > 200 and num(state.samus_y) < 500 then
        hold(session, 6, {"RIGHT"}, "ghz_edge")
      end
    else
      hold(session, 10, {"RIGHT", "B"}, "ghz_low_run")
      hold(session, 3, {"RIGHT", "B", "X"}, "ghz_shoot")
      hold(session, 20, {"RIGHT", "B", "A"}, "ghz_spin")
    end
  end

  if num(session.state.room_id) ~= ROOM_NOOB and not reached_pillar then
    error("ghz_to_noob: did not reach bottom pillar: " .. brief(session.state))
  end

  if num(session.state.room_id) ~= ROOM_NOOB then
    for i = 1, 60 do
      local state = session.state
      if num(state.samus_y) >= 935 then
        break
      end
      hold(session, 1, {}, "ghz_pillar_settle")
    end
    hold(session, 20, {}, "ghz_pillar_stand")

    local gate_line = false
    local state
    for i = 1, 32 do
      state = hold(session, 1, {"A"}, "ghz_gate_jump")
      if 886 <= num(state.samus_y) and num(state.samus_y) <= 889 then
        gate_line = true
        break
      end
    end
    if not gate_line then
      error(string.format(
        "ghz_to_noob: missed blue-gate shot line at (%s,%s)",
        tostring(state.samus_x),
        tostring(state.samus_y)
      ))
    end
    hold(session, 3, {"RIGHT", "X"}, "ghz_gate_shot")
    hold(session, 60, {}, "ghz_gate_open")

    ctrl.vertical_hop(session, 24, "ghz_pillar_vertical_jump")
    local cleared = false
    for i = 1, 220 do
      state = hold(session, 1, {"RIGHT", "B", "A"}, "ghz_pillar_clear")
      if num(state.samus_x) >= 1700 then
        cleared = true
        break
      end
    end
    if not cleared and num(state.samus_x) < 1700 then
      error(string.format(
        "ghz_to_noob: blue gate/pillar clear failed at (%s,%s) pose=%s",
        tostring(state.samus_x),
        tostring(state.samus_y),
        tostring(state.pose)
      ))
    end

    local frame
    for frame = 0, 499 do
      local buttons
      if (frame % 24) < 6 then
        buttons = {"RIGHT", "B", "X"}
      elseif (frame % 40) >= 28 then
        buttons = {"RIGHT", "B", "A"}
      else
        buttons = {"RIGHT", "B"}
      end
      state = hold(session, 1, buttons, "ghz_exit_run")
      if num(state.room_id) == ROOM_NOOB then
        break
      end
    end
  end

  if num(session.state.room_id) ~= ROOM_NOOB then
    error("ghz_to_noob: still in GHZ: " .. brief(session.state))
  end
  return ctrl.wait_ordinary_room(session, ROOM_NOOB, 200, "ghz_to_noob")
end

function M.play_noob_to_red_tower(session)
  ctrl.require_room(session, ROOM_NOOB, "noob_to_red")
  ctrl.try_select(session, 2)

  local state
  if num(session.state.samus_x) < 1150 then
    local setup = false
    local frame
    for frame = 0, 149 do
      state = hold(session, 1, {"RIGHT", "B", "A"}, "noob_bridge_setup_hop")
      if frame > 45 and num(state.samus_x) >= 190 and num(state.samus_y) >= 155 then
        setup = true
        break
      end
    end
    if not setup then
      error(string.format(
        "noob_to_red: could not reach upper-bridge jump setup from (%s,%s)",
        tostring(state.samus_x),
        tostring(state.samus_y)
      ))
    end

    hold(session, 10, {"LEFT"}, "noob_bridge_brake")
    ctrl.vertical_hop(session, 24, "noob_bridge_vertical_jump")
    local dashed = false
    local i
    for i = 1, 330 do
      state = hold(session, 1, {"RIGHT", "B", "A"}, "noob_bridge_dash")
      if num(state.samus_x) >= 1200 then
        dashed = true
        break
      end
    end
    if not dashed and num(state.samus_x) < 1200 then
      error(string.format(
        "noob_to_red: failed upper pit-block bridge (samus=(%s,%s) pose=%s)",
        tostring(state.samus_x),
        tostring(state.samus_y),
        tostring(state.pose)
      ))
    end
  end

  local stuck = 0
  local last_x = num(session.state.samus_x)
  local i
  for i = 0, 1399 do
    state = session.state
    if num(state.room_id) == ROOM_RED_TOWER then
      break
    end

    local pose = num(state.pose)
    if pose == 39 or pose == 40 or pose == 137 or pose == 138 then
      hold(session, 2, {"UP"}, "noob_unmorph")
      hold(session, 4, {}, "noob_unmorph")
      hold(session, 2, {"A"}, "noob_unmorph")
      hold(session, 6, {}, "noob_unmorph")
    elseif num(state.samus_x) >= 1380 then
      hold(session, 2, {"RIGHT", "X"}, "noob_super")
      hold(session, 12, {}, "noob_fuse")
      hold(session, 18, {"RIGHT", "B", "A"}, "noob_spin")
      local j
      for j = 1, 30 do
        state = hold(session, 1, {"RIGHT", "B"}, "noob_push")
        if num(state.room_id) == ROOM_RED_TOWER then
          break
        end
      end
      if num(state.room_id) == ROOM_RED_TOWER then
        break
      end
    else
      local phase = i % 24
      if num(state.samus_y) > 250 then
        state = hold(session, 2, {"LEFT", "A"}, "noob_recover")
      elseif phase < 14 then
        state = hold(session, 1, {"RIGHT", "B"}, "noob_run")
      elseif phase < 18 then
        state = hold(session, 1, {"RIGHT", "B", "X"}, "noob_shoot")
      else
        state = hold(session, 1, {"RIGHT", "B", "A"}, "noob_hop")
      end

      if num(state.samus_x) > last_x + 0.5 then
        stuck = 0
        last_x = num(state.samus_x)
      else
        stuck = stuck + 1
      end

      if stuck > 25 and num(state.samus_x) >= 1050 then
        local k
        for k = 1, 15 do
          hold(session, 1, {"LEFT", "B"}, "noob_backup")
        end
        for k = 1, 18 do
          hold(session, 1, {"RIGHT", "B"}, "noob_runup")
        end
        for k = 1, 45 do
          state = hold(session, 1, {"RIGHT", "B", "A"}, "noob_longjump")
          if num(state.samus_x) >= 1150 then
            stuck = 0
            last_x = num(state.samus_x)
            break
          end
        end
        if num(state.samus_x) < 1150 and stuck > 50 then
          error(string.format(
            "noob_to_red: stalled before right corridor (samus=(%s,%s) pose=%s)",
            tostring(state.samus_x),
            tostring(state.samus_y),
            tostring(state.pose)
          ))
        end
      end
    end
  end

  if num(session.state.room_id) ~= ROOM_RED_TOWER then
    error("noob_to_red: " .. brief(session.state))
  end
  return ctrl.wait_ordinary_room(session, ROOM_RED_TOWER, 220, "noob_to_red")
end

return M
