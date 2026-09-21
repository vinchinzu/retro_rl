-- Big Pink main shaft → Green Hill Zone (with Charge Beam detour).
-- Python: routes/kpdr/brinstar/pink_to_ghz.py

local ctrl = require("brinstar.ctrl")
local charge = require("brinstar.charge_return")

local rooms = {}
pcall(function()
  rooms = require("rooms")
end)

local num = ctrl.num
local hold = ctrl.hold
local brief = ctrl.brief

local ROOM_BIG_PINK = rooms.ROOM_BIG_PINK or 0x9D19
local ROOM_GHZ = rooms.ROOM_GHZ or 0x9E52
local CHARGE_BEAM_MASK = charge.CHARGE_BEAM_MASK or 0x1000
local LIP_Y_MAX = 1635
local LIP_Y_MIN = 1585
local LIP_X_MIN = 690

local M = {}
M.ROOM_BIG_PINK = ROOM_BIG_PINK
M.ROOM_GHZ = ROOM_GHZ

local function on_tunnel_lip(state)
  local y = num(state.samus_y)
  return num(state.samus_x) >= LIP_X_MIN and LIP_Y_MIN <= y and y <= LIP_Y_MAX
end

local function climb_charge_pit_to_mass(session)
  ctrl.unmorph(session)
  local best_y = num(session.state.samus_y)
  local cycle
  for cycle = 1, 30 do
    if num(session.state.samus_y) <= 1700 and num(session.state.samus_x) >= 700 then
      hold(session, 10, {}, "big_pink_mass_band_settle")
      return session.state
    end
    hold(session, 6, {"RIGHT", "B"}, "big_pink_pit_runup")
    hold(session, 1, {"RIGHT", "A"}, "big_pink_pit_jump")
    local i
    for i = 1, 28 do
      local state = hold(session, 1, {"RIGHT", "A"}, "big_pink_pit_air")
      if num(state.samus_y) < best_y then
        best_y = num(state.samus_y)
      end
      if num(state.samus_y) <= 1700 then
        break
      end
    end
    for i = 1, 16 do
      hold(session, 1, {}, "big_pink_pit_land")
    end
  end
  error(string.format(
    "big_pink_to_ghz: charge-pit climb stalled best_y=%s: %s",
    tostring(best_y),
    brief(session.state)
  ))
end

local function escape_knockback_spin(session, prefer_dir, run_frames, spin_frames, label)
  prefer_dir = prefer_dir or "LEFT"
  run_frames = run_frames or 3
  spin_frames = spin_frames or 12
  local ok, knockback = pcall(require, "skills.knockback")
  if ok and knockback and knockback.escape_knockback_spin then
    local called = pcall(
      knockback.escape_knockback_spin,
      session,
      prefer_dir,
      run_frames,
      spin_frames,
      label
    )
    if called then
      return session.state
    end
  end
  local i
  for i = 1, run_frames do
    hold(session, 1, {prefer_dir, "B"}, label .. "_kb_run")
  end
  for i = 1, spin_frames do
    hold(session, 1, {prefer_dir, "B", "A"}, label .. "_kb_spin")
  end
  return session.state
end

local function break_pose_lag(session, label)
  ctrl.unmorph(session)
  if not ctrl.POSE_KNOCKBACK[num(session.state.pose)] then
    hold(session, 8, {}, label .. "_stand_settle")
    return session.state
  end
  hold(session, 12, {"A"}, label .. "_lag_a")
  local i
  for i = 1, 24 do
    local state = hold(session, 1, {}, label .. "_lag_land")
    if not ctrl.POSE_KNOCKBACK[num(state.pose)] then
      break
    end
  end
  if ctrl.POSE_KNOCKBACK[num(session.state.pose)] then
    escape_knockback_spin(session, "LEFT", 3, 12, label)
    for i = 1, 20 do
      hold(session, 1, {}, label .. "_kb_idle")
      if not ctrl.POSE_KNOCKBACK[num(session.state.pose)] then
        break
      end
    end
  end
  hold(session, 10, {}, label .. "_stand_settle")
  return session.state
end

local function mass_to_tunnel_lip(session)
  break_pose_lag(session, "big_pink_lip")

  if num(session.state.samus_y) > 1780 then
    climb_charge_pit_to_mass(session)
    break_pose_lag(session, "big_pink_lip_post_pit")
  end

  local attempt
  for attempt = 1, 4 do
    if num(session.state.samus_y) <= 1690
        and 690 <= num(session.state.samus_x)
        and num(session.state.samus_x) <= 725 then
      break
    end
    hold(session, 12, {"A"}, "big_pink_shelf_prep")
    local i
    for i = 1, 45 do
      local state = hold(session, 1, {"LEFT", "A"}, "big_pink_shelf_hop")
      if num(state.samus_y) <= 1690 and num(state.samus_x) <= 720 then
        break
      end
    end
    for i = 1, 20 do
      hold(session, 1, {}, "big_pink_shelf_land")
    end
    break_pose_lag(session, "big_pink_shelf")
  end

  hold(session, 6, {"DOWN"}, "big_pink_shelf_crouch")
  hold(session, 4, {}, "big_pink_shelf_crouch")
  hold(session, 6, {"DOWN"}, "big_pink_shelf_crouch")
  local i
  for i = 1, 50 do
    if num(session.state.samus_x) <= 680 then
      break
    end
    hold(session, 1, {"LEFT"}, "big_pink_mass_left")
  end
  hold(session, 10, {}, "big_pink_mass_left_settle")

  for attempt = 1, 6 do
    if on_tunnel_lip(session.state) then
      hold(session, 16, {}, "big_pink_tunnel_lip_settle")
      return session.state
    end
    break_pose_lag(session, "big_pink_lip_try")
    hold(session, 8, {"UP", "RIGHT"}, "big_pink_lip_aim")
    hold(session, 10, {"A"}, "big_pink_lip_prep")
    for i = 1, 40 do
      local state = hold(session, 1, {"RIGHT", "A"}, "big_pink_lip_jump")
      if on_tunnel_lip(state) then
        hold(session, 16, {}, "big_pink_tunnel_lip_settle")
        return session.state
      end
    end
    for i = 1, 16 do
      hold(session, 1, {}, "big_pink_lip_land")
    end
    if num(session.state.samus_x) > 730 then
      for i = 1, 25 do
        hold(session, 1, {"LEFT"}, "big_pink_lip_recenter")
      end
      hold(session, 6, {"DOWN"}, "big_pink_lip_recrouch")
    end
  end

  hold(session, 20, {"RIGHT", "B"}, "big_pink_mass_run")
  hold(session, 10, {}, "big_pink_mass_settle")
  hold(session, 12, {"LEFT"}, "big_pink_mass_brake")
  hold(session, 8, {"A"}, "big_pink_mass_vertical")
  for i = 1, 200 do
    local state = hold(session, 1, {"RIGHT", "A"}, "big_pink_tunnel_mount")
    if on_tunnel_lip(state) then
      hold(session, 40, {}, "big_pink_tunnel_lip_settle")
      return session.state
    end
  end
  error("big_pink_to_ghz: missed morph-tunnel lip: " .. brief(session.state))
end

local function mass_to_ghz(session)
  mass_to_tunnel_lip(session)

  local state = session.state
  if not (num(state.samus_x) >= 680 and num(state.samus_y) <= 1650) then
    error("big_pink_to_ghz: not on tunnel lip: " .. brief(session.state))
  end

  ctrl.ensure_morph(session)
  local rolled = false
  local frame
  for frame = 0, 599 do
    local buttons
    if (frame % 45) < 3 then
      buttons = {"RIGHT", "X"}
    else
      buttons = {"RIGHT"}
    end
    state = hold(session, 1, buttons, "big_pink_bomb_roll")
    if num(state.samus_x) >= 900 then
      rolled = true
      break
    end
  end
  if not rolled then
    error("big_pink_to_ghz: lower bomb-roll stalled: " .. brief(session.state))
  end

  ctrl.unmorph(session)
  local pocket = false
  local i
  for i = 1, 160 do
    state = hold(session, 1, {"RIGHT"}, "big_pink_door_approach")
    if num(state.samus_x) >= 930 and num(state.samus_y) >= 1660 then
      pocket = true
      break
    end
  end
  if not pocket then
    error("big_pink_to_ghz: missed green-door pocket: " .. brief(session.state))
  end

  hold(session, 12, {}, "big_pink_door_settle")
  if not ctrl.select_weapon(session, 2) then
    error("could not select weapon 2, still " .. tostring(session.state.selected_item))
  end
  hold(session, 6, {}, "big_pink_super_ready")
  hold(session, 3, {"RIGHT"}, "big_pink_face_door")
  hold(session, 3, {}, "big_pink_face_door_release")
  hold(session, 8, {"X"}, "big_pink_green_door_super")
  hold(session, 50, {}, "big_pink_green_door_fuse")
  local entered = false
  for i = 1, 300 do
    state = hold(session, 1, {"RIGHT", "B"}, "big_pink_enter_ghz")
    if num(state.room_id) == ROOM_GHZ then
      entered = true
      break
    end
  end
  if not entered then
    error("big_pink_to_ghz: green door did not open: " .. brief(session.state))
  end
  return ctrl.wait_ordinary_room(session, ROOM_GHZ, 240, "big_pink_to_ghz")
end

function M.play_big_pink_to_ghz(session)
  ctrl.require_room(session, ROOM_BIG_PINK, "big_pink_to_ghz")
  if ctrl.band(num(session.state.collected_beams), CHARGE_BEAM_MASK) ~= 0 then
    charge._descend_main_to_mass(session)
    return mass_to_ghz(session)
  end
  charge.play_charge_beam_collect(session)
  charge.play_charge_beam_return(session)
  return mass_to_ghz(session)
end

return M
