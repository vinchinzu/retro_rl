-- Spazer Room collect + return to Below Spazer top handoff.
-- Port of snes/super_metroid/routes/kpdr/spazer/collect.py.

local geo = require("red_tower.ctrl")
local g = require("spazer.geometry")

local M = {}
M.SPAZER_BEAM_MASK = g.SPAZER_BEAM_MASK

local function finish_spazer_fanfare(session)
  if not g.has_spazer(session.state) then
    error("spazer_collect: Spazer bit missing: " .. geo.fmt_state(session.state))
  end
  for _ = 1, 500 do
    local state = geo.hold(session, 1, {}, "spazer_item_fanfare")
    if g.is_true_ground_pose(state) and g.has_spazer(state) then
      geo.hold(session, 16, {}, "spazer_item_settle")
      return session.state
    end
  end
  for _ = 1, 40 do
    geo.hold(session, 1, {"A"}, "spazer_item_unstick")
    if g.is_true_ground_pose(session.state) then
      break
    end
    geo.hold(session, 1, {"UP"}, "spazer_item_unstick")
  end
  geo.hold(session, 16, {}, "spazer_item_settle")
  if not g.has_spazer(session.state) then
    error("spazer_collect: Spazer bit missing after fanfare: "
      .. geo.fmt_state(session.state))
  end
  return session.state
end

function M.play_spazer_collect(session)
  geo.require_room(session, geo.ROOM_SPAZER, "spazer_collect")
  if g.has_spazer(session.state) then
    return session.state
  end

  geo.unmorph(session)
  geo.try_select_weapon(session, 0)
  geo.hold(session, 8, {}, "spazer_weapon_settle")

  for _ = 1, 200 do
    local state = geo.hold(session, 1, {"RIGHT", "B"}, "spazer_chozo_approach")
    if g.has_spazer(state) then
      return finish_spazer_fanfare(session)
    end
    if state.samus_x >= 158 then
      break
    end
  end
  geo.hold(session, 10, {}, "spazer_chozo_settle")

  for _ = 1, 20 do
    if geo.POSE_KNOCKBACK[session.state.pose] then
      geo.hold(session, 6, {"A"}, "spazer_lag_break")
      geo.hold(session, 8, {}, "spazer_lag_land")
    end
    geo.hold(session, 2, {"RIGHT"}, "spazer_face_chozo")
    geo.hold(session, 1, {"X"}, "spazer_chozo_shot")
    for _w = 1, 20 do
      local state = geo.hold(session, 1, {"RIGHT"}, "spazer_chozo_wait")
      if g.has_spazer(state) then
        return finish_spazer_fanfare(session)
      end
    end
    if g.has_spazer(session.state) then
      return finish_spazer_fanfare(session)
    end
  end
  error("spazer_collect: Spazer PLM not collected: " .. geo.fmt_state(session.state))
end

local function stand_after_collect(session)
  if g.is_true_ground_pose(session.state) then
    return
  end
  for _ = 1, 200 do
    geo.hold(session, 1, {}, "spazer_return_clear")
    if g.is_true_ground_pose(session.state) then
      return
    end
  end
  for _ = 1, 40 do
    geo.hold(session, 1, {"A"}, "spazer_return_unstick")
    if g.is_true_ground_pose(session.state) then
      return
    end
    geo.hold(session, 1, {"UP"}, "spazer_return_unstick")
  end
end

function M.play_spazer_return_to_below(session)
  geo.require_room(session, geo.ROOM_SPAZER, "spazer_return_to_below")
  if not g.has_spazer(session.state) then
    error("spazer_return_to_below: Spazer not collected (beams="
      .. geo.fmt_hex(session.state.collected_beams or 0) .. ")")
  end

  geo.unmorph(session)
  stand_after_collect(session)
  geo.hold(session, 12, {}, "spazer_return_stand")

  geo.hold(session, 5, {"LEFT", "B"}, "spazer_return_runup")
  geo.hold(session, 12, {"LEFT", "B", "A"}, "spazer_return_spin1")
  geo.hold(session, 40, {"LEFT", "B"}, "spazer_return_land1")
  geo.hold(session, 12, {"LEFT", "B", "A"}, "spazer_return_spin2")
  for _ = 1, 100 do
    local state = geo.hold(session, 1, {"LEFT", "B"}, "spazer_return_to_sill")
    if state.room_id == geo.ROOM_BELOW_SPAZER then
      break
    end
    if state.samus_x <= 50 and state.samus_y <= 155 then
      break
    end
  end

  if session.state.room_id ~= geo.ROOM_BELOW_SPAZER then
    for _ = 1, 30 do
      geo.hold(session, 1, {}, "spazer_return_stop")
      if session.state.pose == 1 or session.state.pose == 2 then
        break
      end
    end
    for _ = 1, 25 do
      if g.is_true_ground_pose(session.state) then
        break
      end
      geo.hold(session, 1, {"A"}, "spazer_return_lag_break")
      geo.hold(session, 2, {}, "spazer_return_lag_break")
    end
    geo.try_select_weapon(session, 0)
    geo.hold(session, 4, {"LEFT"}, "spazer_return_face")
    geo.hold(session, 3, {}, "spazer_return_face_rel")
    geo.hold(session, 8, {"X"}, "spazer_return_blue_shot")
    geo.hold(session, 45, {}, "spazer_return_blue_fuse")
    local entered = false
    for _ = 1, 200 do
      local state = geo.hold(session, 1, {"LEFT", "B"}, "spazer_return_enter")
      if state.room_id == geo.ROOM_BELOW_SPAZER then
        entered = true
        break
      end
      if g.is_lag_pose(state) then
        geo.hold(session, 8, {"A"}, "spazer_return_unstick")
      end
    end
    if not entered then
      error("spazer_return_to_below: failed to leave Spazer: "
        .. geo.fmt_state(session.state))
    end
  end

  geo.wait_ordinary_room(session, geo.ROOM_BELOW_SPAZER, {
    settle_frames = 160,
    label = "spazer_return_to_below",
  })
  for _ = 1, 50 do
    local state = geo.hold(session, 1, {"LEFT", "B"}, "spazer_return_clear_door")
    if state.samus_x <= g.HANDOFF_X_MAX and state.room_id == geo.ROOM_BELOW_SPAZER then
      break
    end
    if state.room_id ~= geo.ROOM_BELOW_SPAZER then
      error("spazer_return_to_below: left Below Spazer during clear: "
        .. geo.fmt_state(state))
    end
  end
  geo.hold(session, 16, {}, "spazer_return_handoff_settle")
  if session.state.room_id ~= geo.ROOM_BELOW_SPAZER then
    error("spazer_return_to_below: bad handoff room: " .. geo.fmt_state(session.state))
  end
  if session.state.samus_x > g.DOOR_TRAP_X_MAX then
    error("spazer_return_to_below: still door-trapped x="
      .. tostring(session.state.samus_x) .. ": " .. geo.fmt_state(session.state))
  end
  return session.state
end

return M
