-- Post-Varia return hops (Varia → Kraid → Eye).
-- Port of snes/super_metroid/routes/kpdr/kraid/varia_return.py.

local geo = require("red_tower.ctrl")

local M = {}

local STANDING = geo.set(1, 2, 5, 6, 9, 10)
local BAD_AIR = geo.set(81, 164, 0, 155)

local function recover_standing(session, timeout)
  timeout = timeout or 400
  geo.unmorph(session)
  for index = 0, timeout - 1 do
    local st = session.state
    if STANDING[st.pose] and (st.velocity_y or 0) == 0 and st.samus_y >= 130 then
      return
    end
    local phase = index % 24
    if BAD_AIR[st.pose] or st.pose == 137 or st.pose == 138 then
      if phase < 4 then
        geo.hold(session, 1, {"A"}, "varia_recover_jump")
      elseif phase < 12 then
        geo.hold(session, 1, {"RIGHT"}, "varia_recover_nudge")
      else
        geo.hold(session, 1, {}, "varia_recover_idle")
      end
    elseif phase < 3 then
      geo.hold(session, 1, {"A"}, "varia_recover_jump")
    elseif phase < 10 then
      geo.hold(session, 1, {"RIGHT"}, "varia_recover_nudge")
    else
      geo.hold(session, 1, {}, "varia_recover_idle")
    end
  end
  error("varia_to_kraid: recover standing failed: " .. geo.fmt_state(session.state))
end

function M.play_varia_to_kraid(session)
  geo.require_room(session, geo.ROOM_VARIA, "varia_to_kraid")
  if not geo.has_varia(session.state) then
    error("varia_to_kraid: Varia not collected; " .. geo.fmt_state(session.state))
  end
  recover_standing(session)

  for _ = 1, 50 do
    local x = session.state.samus_x
    if x >= 95 and x <= 130 then
      break
    end
    local dir = (x < 95) and "RIGHT" or "LEFT"
    geo.hold(session, 1, {dir}, "varia_center")
  end
  geo.hold(session, 8, {}, "varia_center_settle")
  geo.try_select_weapon(session, 0)

  geo.hold(session, 8, {"LEFT"}, "varia_face_left")
  geo.hold(session, 6, {}, "varia_face_release")
  for _ = 1, 4 do
    geo.hold(session, 4, {"X"}, "varia_door_shot")
    geo.hold(session, 18, {}, "varia_door_fuse")
  end

  local entered = false
  local state
  for _ = 1, 360 do
    state = geo.hold(session, 1, {"LEFT", "B", "A"}, "varia_exit_spin")
    if state.room_id == geo.ROOM_KRAID then
      entered = true
      break
    end
    if (state.pose == 137 or state.pose == 138) and state.samus_x <= 60 then
      geo.hold(session, 4, {}, "varia_lip_release")
      geo.hold(session, 3, {"RIGHT"}, "varia_lip_backoff")
      geo.hold(session, 4, {"X"}, "varia_lip_reshot")
      geo.hold(session, 12, {}, "varia_lip_fuse")
    end
  end
  if not entered then
    error("varia_to_kraid: did not reach " .. geo.fmt_hex(geo.ROOM_KRAID)
      .. ": " .. geo.fmt_state(session.state))
  end

  return geo.wait_ordinary_room(session, geo.ROOM_KRAID, {
    settle_frames = 240,
    label = "varia_to_kraid",
    x_range = {350, 560},
    y_range = {300, 450},
    min_settle_frame = 12,
  })
end

function M.play_kraid_to_eye_return(session)
  geo.require_room(session, geo.ROOM_KRAID, "kraid_to_eye_return")
  geo.select_weapon(session, 0)

  local approached = false
  for _ = 1, 150 do
    local state = geo.hold(session, 1, {"LEFT"}, "kraid_return_approach")
    if state.samus_x <= 160 then
      approached = true
      break
    end
  end
  if not approached then
    error("kraid_to_eye_return: left door approach timed out: "
      .. geo.fmt_state(session.state))
  end
  geo.hold(session, 12, {}, "kraid_return_approach_settle")
  geo.hold(session, 10, {"RIGHT"}, "kraid_return_lip_backoff")
  geo.unmorph(session)
  geo.hold(session, 8, {"LEFT"}, "kraid_return_face_left")
  geo.hold(session, 6, {}, "kraid_return_release")
  for _ = 1, 6 do
    geo.hold(session, 4, {"X"}, "kraid_return_door_shot")
    geo.hold(session, 14, {}, "kraid_return_door_fuse")
  end

  local entered = false
  local state
  for index = 0, 899 do
    local phase = index % 30
    if phase < 4 then
      state = geo.hold(session, 1, {"LEFT", "A"}, "kraid_return_jump")
    elseif phase < 10 then
      state = geo.hold(session, 1, {"LEFT", "A", "B"}, "kraid_return_jump_spin")
    elseif phase < 14 then
      state = geo.hold(session, 1, {"X"}, "kraid_return_reshot")
    else
      state = geo.hold(session, 1, {"LEFT", "B"}, "kraid_return_run")
    end
    if state.room_id == geo.ROOM_KRAID_EYE then
      entered = true
      break
    end
    if state.door_transition and state.door_transition ~= 0 then
      for _ = 1, 80 do
        state = geo.hold(session, 1, {}, "kraid_return_transition")
        if state.room_id == geo.ROOM_KRAID_EYE and (state.door_transition or 0) == 0 then
          break
        end
      end
      if state.room_id == geo.ROOM_KRAID_EYE then
        entered = true
        break
      end
    end
    if (state.pose == 137 or state.pose == 138) and state.samus_x <= 80 then
      geo.hold(session, 4, {}, "kraid_return_lip_release")
      geo.hold(session, 4, {"RIGHT"}, "kraid_return_lip_backoff")
      geo.hold(session, 4, {"X"}, "kraid_return_lip_reshot")
      geo.hold(session, 12, {}, "kraid_return_lip_fuse")
    end
  end
  if not entered then
    error("kraid_to_eye_return: left eye-door exit timed out: "
      .. geo.fmt_state(session.state))
  end
  return geo.wait_ordinary_room(session, geo.ROOM_KRAID_EYE, {
    settle_frames = 340,
    label = "kraid_to_eye_return",
    x_range = {300, 560},
    y_range = {300, 450},
    min_settle_frame = 12,
  })
end

return M
