-- Post-Torizo Parlor → Terminator → Green Pirates → Mushrooms → Main Shaft
-- → Dachora → Big Pink → Kihunters → Spore (Clean fight) → Super room.
-- Python: routes/kpdr/brinstar/spore_spawn.py + crateria/alcatraz_escape.py

local ctrl = require("brinstar.ctrl")
local combat = require("combat.spore_spawn")
local tape = require("brinstar.spore_climb_tape")

local rooms = {}
pcall(function()
  rooms = require("rooms")
end)

local num = ctrl.num
local hold = ctrl.hold
local brief = ctrl.brief

local ROOM_PARLOR = rooms.ROOM_PARLOR or 0x92FD
local ROOM_TERMINATOR = rooms.ROOM_TERMINATOR or 0x990D
local ROOM_GREEN_PIRATES = rooms.ROOM_GREEN_PIRATES or 0x99BD
local ROOM_LOWER_MUSHROOMS = rooms.ROOM_LOWER_MUSHROOMS or 0x9969
local ROOM_GREEN_ELEVATOR = rooms.ROOM_GREEN_ELEVATOR or 0x9938
local ROOM_GREEN_MAIN_SHAFT = rooms.ROOM_GREEN_MAIN_SHAFT or 0x9AD9
local ROOM_DACHORA = rooms.ROOM_DACHORA or 0x9CB3
local ROOM_BIG_PINK = rooms.ROOM_BIG_PINK or 0x9D19
local ROOM_SPORE_KIHUNTER = rooms.ROOM_SPORE_KIHUNTER or 0x9D9C
local ROOM_SPORE_SPAWN = rooms.ROOM_SPORE_SPAWN or combat.ROOM_SPORE_SPAWN or 0x9DC7
local ROOM_SUPER = rooms.ROOM_SUPER or 0x9B5B

local SHAFT_LIP_Y = 210
local ROLLOUT_MAX_X = 760
local ROLLOUT_MAX_Y = 230
local LEFT_WALL_X = {795, 820}
local LEFT_WALL_Y = {535, 550}
local MID_LEDGE_X = {820, 838}
local MID_LEDGE_Y = {450, 470}
local MID_GRAB_X = {818, 848}
local MID_GRAB_Y_MAX = 478
local LOWER_WJ_X = {795, 815}
local LOWER_WJ_Y = {350, 375}
local GROUNDED = {
  [1] = true, [2] = true, [5] = true, [6] = true,
  [7] = true, [8] = true, [9] = true, [10] = true,
}
local PROBE_MORPH = {
  [29] = true, [30] = true, [31] = true, [32] = true,
  [49] = true, [50] = true, [65] = true, [66] = true,
  [165] = true, [166] = true, [167] = true,
}
local LIP_LAND = {
  [1] = true, [2] = true, [5] = true, [6] = true,
  [7] = true, [8] = true, [9] = true, [10] = true,
  [81] = true, [82] = true, [108] = true, [163] = true,
  [164] = true, [165] = true, [166] = true, [167] = true, [229] = true,
}
local BOMBS_MASK = 0x1000

local M = {}

local function in_xy(state, xr, yr)
  return xr[1] <= num(state.samus_x) and num(state.samus_x) <= xr[2]
    and yr[1] <= num(state.samus_y) and num(state.samus_y) <= yr[2]
end

local function at_mid_ledge(state)
  return num(state.room_id) == ROOM_PARLOR
    and in_xy(state, MID_LEDGE_X, MID_LEDGE_Y)
    and GROUNDED[num(state.pose)] == true
end

local function mid_ledge_landed(state)
  if at_mid_ledge(state) then
    return true
  end
  local x, y = num(state.samus_x), num(state.samus_y)
  local on_lip = (MID_LEDGE_X[1] - 4) <= x and x <= (MID_LEDGE_X[2] + 12)
    and (MID_LEDGE_Y[1] - 4) <= y and y <= (MID_LEDGE_Y[2] + 4)
    and LIP_LAND[num(state.pose)] == true
  if on_lip then
    return true
  end
  return y <= 470 and GROUNDED[num(state.pose)] == true
end

local function at_alcatraz_rollout(state)
  return num(state.room_id) == ROOM_PARLOR
    and num(state.samus_x) <= ROLLOUT_MAX_X
    and num(state.samus_y) <= ROLLOUT_MAX_Y
    and PROBE_MORPH[num(state.pose)] == true
end

local function require_geometry(session, label, xr, yr)
  local state = session.state
  if not (num(state.room_id) == ROOM_PARLOR and in_xy(state, xr, yr)) then
    error(string.format(
      "Alcatraz %s missed: room=0x%04X xy=(%s,%s) pose=%s frame=%s",
      label,
      num(state.room_id),
      tostring(state.samus_x),
      tostring(state.samus_y),
      tostring(state.pose),
      tostring(session.frame)
    ))
  end
end

local function unmorph_probe_pose(session)
  if PROBE_MORPH[num(session.state.pose)] ~= true then
    return
  end
  hold(session, 8, {"UP"}, "alcatraz_base_unmorph")
  hold(session, 6, {}, "alcatraz_base_unmorph_settle")
end

local function land_left_wall_base(session)
  hold(session, 2, {"LEFT"}, "alcatraz_base_face")
  hold(session, 30, {"LEFT", "B"}, "alcatraz_base_run")
  hold(session, 18, {"LEFT", "A"}, "alcatraz_base_hop")
  hold(session, 16, {}, "alcatraz_base_land")
  unmorph_probe_pose(session)
  require_geometry(session, "left-wall base", LEFT_WALL_X, LEFT_WALL_Y)
  return session.frame
end

local function reach_mid_ledge(session)
  hold(session, 2, {"RIGHT"}, "alcatraz_ledge_face")
  local _
  for _ = 1, 3 do
    hold(session, 40, {"RIGHT", "A"}, "alcatraz_ledge_cross")
    hold(session, 2, {"LEFT"}, "alcatraz_ledge_turn")
    hold(session, 28, {"LEFT", "A"}, "alcatraz_ledge_latch")
    if at_mid_ledge(session.state)
        or (num(session.state.samus_y) <= 470
          and GROUNDED[num(session.state.pose)] == true) then
      break
    end
  end
  hold(session, 16, {}, "alcatraz_ledge_settle")
  unmorph_probe_pose(session)
  hold(session, 8, {}, "alcatraz_ledge_stand")
  require_geometry(session, "mid ledge", MID_LEDGE_X, MID_LEDGE_Y)
  return session.frame
end

local function play_walljump_pulse(session, away, turn_frames, jump_frames, reason)
  hold(session, turn_frames, {away}, reason .. "_turn")
  hold(session, jump_frames, {away, "A"}, reason .. "_jump")
end

local function instant_morph_rollout(session)
  hold(session, 1, {"DOWN", "A"}, "alcatraz_instant_morph")
  local morph_frame = session.frame
  local _
  for _ = 1, 80 do
    hold(session, 1, {"LEFT"}, "alcatraz_escape")
    if at_alcatraz_rollout(session.state) then
      return morph_frame
    end
  end
  error("Alcatraz Morph opening missed: " .. brief(session.state))
end

local function climb_chimney(session)
  hold(session, 3, {"LEFT"}, "alcatraz_chimney_face")
  hold(session, 22, {"LEFT", "B", "A"}, "alcatraz_chimney_left_wall")
  hold(session, 4, {"LEFT"}, "alcatraz_chimney_contact")
  play_walljump_pulse(session, "RIGHT", 2, 40, "alcatraz_walljump_1")
  play_walljump_pulse(session, "LEFT", 2, 30, "alcatraz_walljump_2")
  require_geometry(session, "lower walljumps", LOWER_WJ_X, LOWER_WJ_Y)
  play_walljump_pulse(session, "RIGHT", 2, 34, "alcatraz_walljump_3")
  hold(session, 5, {"LEFT"}, "alcatraz_walljump_4_turn")
  local walljump_frame
  local _
  for _ = 1, 90 do
    hold(session, 1, {"LEFT", "A"}, "alcatraz_walljump_4_jump")
    if num(session.state.samus_y) <= 225
        and num(session.state.pose) == ctrl.POSE_WALL_LATCH then
      walljump_frame = session.frame
      break
    end
  end
  if walljump_frame == nil then
    error("Alcatraz final walljump missed: " .. brief(session.state))
  end
  local morph_frame = instant_morph_rollout(session)
  return walljump_frame, morph_frame
end

function M.play_alcatraz_escape(session)
  local state = session.state
  local morph_ball = state.morph_ball
  if morph_ball == nil then
    morph_ball = ctrl.band(num(state.collected_items), 0x0004) ~= 0
  end
  if not (
      num(state.room_id) == ROOM_PARLOR
      and ctrl.is_ordinary(state)
      and num(state.game_state) == 8
      and num(state.door_transition) == 0
      and morph_ball
      and num(state.samus_x) == 968
      and num(state.samus_y) == 651
      and num(state.pose) == 2
    ) then
    error("Alcatraz natural entry mismatch: " .. brief(state))
  end
  local entry_frame = session.frame
  local base_frame = land_left_wall_base(session)
  local ledge_frame = reach_mid_ledge(session)
  local walljump_frame, morph_frame = climb_chimney(session)
  state = session.state
  return {
    entry_frame = entry_frame,
    base_frame = base_frame,
    ledge_frame = ledge_frame,
    walljump_frame = walljump_frame,
    morph_frame = morph_frame,
    exit_frame = session.frame,
    exit_x = num(state.samus_x),
    exit_y = num(state.samus_y),
  }
end

function M.play_parlor_to_main_shaft(session)
  ctrl.require_ordinary_room(session, ROOM_PARLOR, "post-Torizo entry")
  local st = session.state
  local bombs = st.bombs
  if bombs == nil then
    bombs = ctrl.band(num(st.collected_items), BOMBS_MASK) ~= 0
  end
  if not bombs or num(st.max_missiles) < 10 then
    error("post-Torizo capabilities missing: " .. brief(st))
  end
  M.play_alcatraz_escape(session)
  local i
  for i = 1, 9 do
    hold(session, 50, {"LEFT", "A", "B", "X"}, "parlor_terminator_exit")
    hold(session, 10, {"LEFT", "B", "X"}, "parlor_terminator_exit")
  end
  hold(session, 100, {}, "terminator_entry_settle")
  hold(session, 2, {"DOWN"}, "terminator_morph")
  hold(session, 3, {}, "terminator_morph")
  hold(session, 2, {"DOWN"}, "terminator_morph")
  hold(session, 10, {}, "terminator_morph")
  for i = 1, 8 do
    hold(session, 45, {"LEFT", "X"}, "terminator_bomb_tunnel")
    hold(session, 15, {"LEFT"}, "terminator_bomb_tunnel")
  end
  ctrl.wait_ordinary_room(session, ROOM_TERMINATOR, 240, "terminator_traversal")

  for i = 1, 7 do
    hold(session, 50, {"LEFT", "A", "B", "X"}, "terminator_energy_tank")
    hold(session, 10, {"LEFT", "B", "X"}, "terminator_energy_tank")
  end
  for i = 1, 30 do
    hold(session, 10, {"LEFT"}, "collect_terminator_energy_tank")
    if num(session.state.max_health) >= 199 then
      break
    end
  end
  if num(session.state.max_health) < 199 then
    error("Terminator Energy Tank was not collected: " .. brief(session.state))
  end
  ctrl.wait_until(session, function(state)
    return num(state.room_id) == ROOM_GREEN_PIRATES
  end, 600, "exit_terminator", {"LEFT", "A", "B", "X"})
  hold(session, 180, {}, "green_pirates_entry_settle")
  ctrl.require_ordinary_room(session, ROOM_GREEN_PIRATES, "Green Pirates entry")

  hold(session, 100, {"LEFT", "B", "X"}, "green_pirates_descent")
  local dirs = {"RIGHT", "LEFT", "RIGHT", "LEFT", "RIGHT", "LEFT", "RIGHT", "LEFT"}
  for i = 1, #dirs do
    hold(session, 80, {dirs[i], "B", "X"}, "green_pirates_descent")
  end
  hold(session, 180, {}, "lower_mushrooms_entry_settle")
  ctrl.require_ordinary_room(session, ROOM_LOWER_MUSHROOMS, "Lower Mushrooms entry")

  for i = 1, 13 do
    hold(session, 60, {"LEFT", "A", "B", "X"}, "lower_mushrooms")
  end
  hold(session, 240, {}, "green_elevator_entry_settle")
  ctrl.require_ordinary_room(session, ROOM_GREEN_ELEVATOR, "Green elevator entry")
  local centered = false
  for i = 1, 30 do
    local state = session.state
    if 118 <= num(state.samus_x) and num(state.samus_x) <= 126
        and num(state.velocity_x) == 0 then
      centered = true
      break
    end
    local direction
    if num(state.samus_x) < 118 then
      direction = "RIGHT"
    elseif num(state.samus_x) > 126 then
      direction = "LEFT"
    elseif num(state.velocity_x) < 0 then
      direction = "RIGHT"
    else
      direction = "LEFT"
    end
    hold(session, 10, {direction}, "green_elevator_center")
    hold(session, 10, {}, "green_elevator_center")
  end
  if not centered then
    local state = session.state
    if not (118 <= num(state.samus_x) and num(state.samus_x) <= 126
        and num(state.velocity_x) == 0) then
      error("Could not center on Green Brinstar elevator: " .. brief(state))
    end
  end
  hold(session, 10, {"DOWN"}, "green_elevator_descend")
  hold(session, 1000, {}, "green_elevator_descent_settle")
  ctrl.require_ordinary_room(session, ROOM_GREEN_MAIN_SHAFT, "Green Brinstar Main Shaft landing")
end

function M.play_main_shaft_to_spore_spawn(session)
  ctrl.require_ordinary_room(session, ROOM_GREEN_MAIN_SHAFT, "Main Shaft route entry")
  hold(session, 1000, {}, "main_shaft_entry_settle")
  local descent = {
    {"RIGHT", "B"}, {"LEFT", "B"}, {"RIGHT", "B"}, {"LEFT", "B"},
  }
  local i
  for i = 1, #descent do
    hold(session, 60, descent[i], "main_shaft_descent")
  end
  hold(session, 50, {}, "main_shaft_descent_settle")
  local dachora_level = {
    {"RIGHT", "B"}, {"LEFT", "B"}, {"RIGHT", "B"}, {"LEFT", "B"}, {"RIGHT", "B"},
  }
  for i = 1, #dachora_level do
    hold(session, 80, dachora_level[i], "main_shaft_dachora_level")
  end
  hold(session, 30, {}, "main_shaft_dachora_door_settle")
  hold(session, 1, {"SELECT"}, "select_missiles")
  hold(session, 10, {}, "select_missiles_settle")
  for i = 1, 15 do
    hold(session, 2, {"X"}, "open_dachora_red_door")
    hold(session, 15, {}, "open_dachora_red_door")
  end
  hold(session, 100, {"RIGHT", "B"}, "enter_dachora")
  hold(session, 250, {}, "dachora_entry_settle")
  ctrl.require_ordinary_room(session, ROOM_DACHORA, "Dachora entry")

  hold(session, 350, {"RIGHT", "A", "B", "X"}, "cross_dachora")
  hold(session, 2, {"DOWN"}, "dachora_tunnel_morph")
  hold(session, 3, {}, "dachora_tunnel_morph")
  hold(session, 2, {"DOWN"}, "dachora_tunnel_morph")
  hold(session, 10, {}, "dachora_tunnel_morph")
  for i = 1, 15 do
    hold(session, 45, {"RIGHT", "X"}, "bomb_dachora_tunnel")
    hold(session, 15, {"RIGHT"}, "bomb_dachora_tunnel")
  end
  hold(session, 160, {"RIGHT", "A", "B", "X"}, "exit_dachora")
  hold(session, 300, {}, "big_pink_entry_settle")
  ctrl.require_ordinary_room(session, ROOM_BIG_PINK, "Big Pink entry")

  hold(session, 2, {"UP"}, "unmorph_big_pink")
  hold(session, 10, {}, "unmorph_big_pink")
  hold(session, 180, {"RIGHT", "A", "B", "X"}, "big_pink_climb")
  hold(session, 80, {"LEFT", "A", "B", "X"}, "big_pink_climb")
  ctrl.play_tape(session, tape.big_pink_climb, "big_pink_map_guided_climb", tape.hold_frames)
  if not (num(session.state.samus_y) <= 150 and num(session.state.samus_x) >= 780) then
    error("Big Pink climb missed upper-right door: " .. brief(session.state))
  end
  hold(session, 100, {"RIGHT", "B", "X"}, "big_pink_red_door_approach")
  for i = 1, 15 do
    hold(session, 2, {"X"}, "open_kihunter_red_door")
    hold(session, 15, {}, "open_kihunter_red_door")
  end
  hold(session, 150, {"RIGHT", "B"}, "enter_kihunter")
  hold(session, 300, {}, "kihunter_entry_settle")
  ctrl.require_ordinary_room(session, ROOM_SPORE_KIHUNTER, "Spore Kihunter entry")

  for i = 0, 7 do
    local direction
    if (i % 2) == 0 then
      direction = "RIGHT"
    else
      direction = "LEFT"
    end
    hold(session, 180, {direction, "A", "B", "X"}, "clear_spore_kihunters")
  end
  local aim = {
    {"UP", "X"},
    {"LEFT", "UP", "X"},
    {"RIGHT", "UP", "X"},
    {"LEFT", "X"},
    {"RIGHT", "X"},
  }
  for i = 0, 239 do
    local names = aim[(i % 5) + 1]
    hold(session, 2, names, "aim_at_spore_kihunters")
    local rest = {}
    local n
    for n = 1, #names do
      if names[n] ~= "X" then
        rest[#rest + 1] = names[n]
      end
    end
    hold(session, 8, rest, "aim_at_spore_kihunters")
  end
  hold(session, 300, {}, "kihunter_clear_settle")
  if num(session.state.enemies_killed) < 3 then
    error("Spore Kihunters did not clear naturally: " .. brief(session.state))
  end
  hold(session, 80, {"RIGHT", "B"}, "kihunter_boss_door_runway")
  hold(session, 100, {"RIGHT", "A", "B", "X"}, "kihunter_boss_door_jump")
  hold(session, 10, {}, "release_kihunter_jump")
  hold(session, 80, {"RIGHT", "A", "B", "X"}, "align_spore_spawn_door")
  hold(session, 10, {}, "release_kihunter_door_align")
  hold(session, 30, {"LEFT", "B"}, "center_under_spore_spawn_door")
  hold(session, 60, {}, "center_under_spore_spawn_door")
  for i = 1, 15 do
    hold(session, 2, {"UP", "X"}, "open_spore_spawn_door")
    hold(session, 10, {"UP"}, "open_spore_spawn_door")
  end
  hold(session, 10, {}, "release_spore_spawn_door_shot")
  hold(session, 120, {"UP", "A", "B"}, "enter_spore_spawn")
  hold(session, 300, {}, "spore_spawn_entry_settle")
  ctrl.require_ordinary_room(session, ROOM_SPORE_SPAWN, "Spore Spawn entry")

  local bits_before = ctrl.area_boss_bits(session.state, 1)
  local fight = combat.play_spore_spawn_fight(session)
  if fight.outcome ~= "spore_spawn_defeated" then
    error("Spore Spawn fight failed: " .. tostring(fight.outcome)
      .. " " .. brief(session.state))
  end

  hold(session, 600, {}, "spore_spawn_death_settle")
  ctrl.play_tape(session, tape.spore_exit_climb, "spore_exit_map_guided_climb", tape.hold_frames)
  if not (num(session.state.samus_y) <= 150 and num(session.state.samus_x) >= 170) then
    error("Spore exit climb missed upper-right door: " .. brief(session.state))
  end
  for i = 1, 20 do
    hold(session, 2, {"RIGHT", "X"}, "open_spore_exit_door")
    hold(session, 8, {"RIGHT"}, "open_spore_exit_door")
  end
  hold(session, 300, {}, "spore_spawn_exit_settle")
  ctrl.require_ordinary_room(session, ROOM_SUPER, "Spore Spawn natural exit")

  return {
    entry_frame = fight.start_frame,
    activation_frame = fight.start_frame,
    defeat_frame = fight.defeat_frame or session.frame,
    exit_frame = session.frame,
    peak_hp = fight.peak_hp,
    observed_hp = {
      fight.peak_hp, fight.min_enemy_hp, fight.final_enemy_hp,
    },
    brinstar_boss_bits_before = bits_before,
    brinstar_boss_bits_after = ctrl.area_boss_bits(session.state, 1),
    vulnerable_spritemaps = fight.vulnerable_spritemaps,
    outcome = fight.outcome,
  }
end

function M.play_post_torizo_to_spore_spawn(session)
  M.play_parlor_to_main_shaft(session)
  return M.play_main_shaft_to_spore_spawn(session)
end

M.ROOM_PARLOR = ROOM_PARLOR
M.ROOM_GREEN_MAIN_SHAFT = ROOM_GREEN_MAIN_SHAFT
M.ROOM_SPORE_SPAWN = ROOM_SPORE_SPAWN
M.ROOM_SUPER = ROOM_SUPER
M.SHAFT_LIP_Y = SHAFT_LIP_Y
M.mid_ledge_landed = mid_ledge_landed

return M
