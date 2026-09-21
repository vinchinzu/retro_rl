-- Kraid return reverse hops (Eye → Baby → Kihunter → Zeela → Warehouse).
-- Port of snes/super_metroid/routes/kpdr/kraid/from_kraid.py.

local geo = require("red_tower.ctrl")
local kr = require("kraid.kraid_return")

local M = {}

local LAND_POSES = geo.set(1, 2, 5, 6, 9, 10, 137, 138)

function M.play_eye_to_baby_return(session)
  geo.require_room(session, geo.ROOM_KRAID_EYE, "eye_to_baby_return")
  geo.select_weapon(session, 0)
  kr.eye_mid_room_approach(session, "eye_to_baby_return")
  kr.lip_stage(session, {
    label = "eye_to_baby",
    backoff = "RIGHT",
    face = "LEFT",
    backoff_frames = 8,
    face_frames = 8,
    release_frames = 6,
    settle_frames = 8,
  })
  kr.beam_open_door(session, {
    label = "eye_to_baby",
    shots = 6,
    shot_frames = 4,
    fuse_frames = 14,
  })
  kr.jump_enter_exit(session, geo.ROOM_BABY_KRAID, {
    label = "eye_to_baby",
    direction = "LEFT",
    max_frames = 700,
    transition_drain = 80,
  })
  return geo.wait_ordinary_room(session, geo.ROOM_BABY_KRAID, {
    settle_frames = 320,
    label = "eye_to_baby_return",
  })
end

function M.play_baby_to_kihunter_return(session)
  local to = require("kraid.to_kraid")
  geo.require_room(session, geo.ROOM_BABY_KRAID, "baby_to_kihunter_return")
  geo.select_weapon(session, 2)
  for _ = 1, 30 do
    local state = geo.hold(session, 1, {}, "baby_return_land")
    if (state.velocity_y or 0) == 0 and LAND_POSES[state.pose] then
      break
    end
  end
  to.baby_kraid_sweep(session, "LEFT", 80, 1700, "baby_return_clear_left")
  if (session.state.enemies_killed or 0) < (session.state.num_enemies or 0) then
    to.baby_kraid_sweep(session, "RIGHT", 1490, 1900, "baby_return_clear_right")
  end
  if (session.state.enemies_killed or 0) < (session.state.num_enemies or 0) then
    to.baby_kraid_sweep(session, "LEFT", 80, 1900, "baby_return_clear_left2")
  end
  geo.select_weapon(session, 0)
  for _ = 1, 200 do
    local state = geo.hold(session, 1, {"LEFT", "B"}, "baby_return_door_approach")
    if state.samus_x <= 120 then
      break
    end
  end
  kr.lip_stage(session, {
    label = "baby_return",
    backoff = "RIGHT",
    face = "LEFT",
    backoff_frames = 10,
    face_frames = 8,
    release_frames = 6,
  })
  kr.beam_open_door(session, {
    label = "baby_return",
    shots = 6,
    shot_frames = 4,
    fuse_frames = 14,
  })
  kr.jump_enter_exit(session, geo.ROOM_WAREHOUSE_KIHUNTER, {
    label = "baby_return",
    direction = "LEFT",
    max_frames = 700,
    transition_drain = 80,
  })
  return geo.wait_ordinary_room(session, geo.ROOM_WAREHOUSE_KIHUNTER, {
    settle_frames = 320,
    label = "baby_to_kihunter_return",
  })
end

function M.play_kihunter_to_zeela_return(session)
  local label = "kihunter_to_zeela_return"
  geo.require_room(session, geo.ROOM_WAREHOUSE_KIHUNTER, label)
  geo.select_weapon(session, 0)
  geo.unmorph(session)
  local best_min_y = {session.state.samus_y}
  kr.kihunter_wall_plant(session, label)
  kr.kihunter_mid_ledge(session, label, best_min_y)
  kr.kihunter_bomb_hole(session, label, best_min_y)
  kr.kihunter_upper_to_zeela_window(session, label)
  return geo.wait_ordinary_room(session, geo.ROOM_ZEELA, {
    settle_frames = 320,
    label = label,
    y_range = {385, 410},
  })
end

function M.play_zeela_to_warehouse_return(session)
  local label = "zeela_to_warehouse_return"
  geo.require_room(session, geo.ROOM_ZEELA, label)
  geo.select_weapon(session, 0)
  kr.zeela_bottom_roll(session, label)
  kr.zeela_mid_platform(session, label)
  kr.zeela_below_platform_lip(session, label)
  kr.zeela_wall_plant(session, label)
  local early = kr.zeela_shotblock_wall_climb(session, label)
  if early ~= nil then
    return early
  end
  kr.zeela_warehouse_door_exit(session, label)
  return geo.wait_ordinary_room(session, geo.ROOM_WAREHOUSE, {
    settle_frames = 320,
    label = label,
  })
end

return M
