-- Bubble Mountain → Bat Cave hop policy. Skills must not hardcode these.

local rooms = require("rooms")
local controller = require("skills.controller")
local geometry = require("skills.geometry")

local pol = {}

pol.ROOM_ID = rooms.ROOM_BUBBLE
pol.EXIT_ROOM_ID = rooms.ROOM_BAT_CAVE

pol.LOWER_FRAMES = 3500
pol.MID_REPIN_FRAMES = 900
pol.MID_FRAMES = 5500
pol.DOOR_FRAMES = 1200
pol.TO_BAT_SETTLE_FRAMES = 320
pol.DOOR_SUPER_X = 420
pol.DOOR_SUPER_Y = 160
pol.DOOR_WJ_PERIOD = 10
pol.DOOR_WJ_INTO = 3
pol.DOOR_WJ_BOUNCE = 2
pol.DOOR_X_CAP = 480
pol.DOOR_OUTER_X = 400
pol.DOOR_CROUCH_FRAMES = 0
pol.MID_Y = 400
pol.TOP_Y = 200
pol.TOP_X = 300

pol.TRUE_GROUND = geometry.TRUE_GROUND
pol.STAND_PIN = geometry.STAND_PIN
pol.STANDING_POSES = geometry.STANDING_POSES

pol.CAVITY_X_MAX = 395
pol.MID_STAND_X = { 77, 160 }

pol.FLOOR_SHELF_X = 108
pol.LOWER_SHELVES = {
  { 120, 560 },
  { 110, 515 },
  { 100, 475 },
  { 90, 450 },
  { 70, 420 },
  { 50, 395 },
}

pol.LIP_X = { 65, 100 }
pol.LIP_Y = { 410, 450 }

pol.HEIGHT_CLASS_Y = 280
pol.MID_RESEAT_Y = 320
pol.RIGHT_SHELF_X = 300
pol.RIGHT_SHELF_Y = 390
pol.RIGHT_WJ_PERIOD = 8
pol.RIGHT_WJ_INTO = 2
pol.RIGHT_WJ_BOUNCE = 2
pol.MIDHIGH_Y = 450

pol.LIP_CHARGE = 12
pol.LIP_SPIN = 44
pol.LIP_EXTEND = 70

pol.SAVE_RUNWAY_X = { 25, 90 }
pol.SAVE_RUNWAY_Y = { 380, 430 }
pol.SAVE_RUNWAY_FIRE_X = { 25, 60 }
pol.SAVE_CLEAR_X_FRAMES = 12
pol.SAVE_EDGE_LEFT_FRAMES = 40
pol.SAVE_STATIONARY_FACE = 6
pol.SAVE_STATIONARY_X = 28
pol.SAVE_HUMAN_SEAT_X = { 25, 30 }
pol.SAVE_RUN_FRAMES = 21
pol.SAVE_DASH_MAX_FRAMES = 32
pol.SAVE_ARM_PUMP = false
pol.SAVE_ARM_PUMP_PERIOD = 2
pol.SAVE_CROUCH_FRAMES = 2
pol.SAVE_SPIN_FRAMES = 83
pol.SAVE_APPROACH_BA = 4
pol.SAVE_APPROACH_IDLE = 2
pol.SAVE_APPROACH_TURN = 2
pol.SAVE_WJ_LEFT_A = 20
pol.SAVE_WJ_AMID = 4
pol.SAVE_WJ_RIGHT_A = 8
pol.SAVE_WJ2_LEFT_A = 14
pol.SAVE_WJ2_AMID = 2
pol.SAVE_WJ2_RIGHT_A = 6
pol.SAVE_WJ_FOLLOW = 40
pol.WJ_INTO_X = 250
pol.WJ_LATCH_TIMEOUT = 36
pol.WJ_APPROACH_X = { 230, 290 }
pol.WJ_APPROACH_Y = { 200, 340 }
pol.WJ2_LEFT_X = 220
pol.WJ2_LEFT_Y = 200
pol.WJ2_LEFT_SEEK = 28
pol.WJ2_LEFT_INTO = 8
pol.WJ2_LEFT_FLIP = 16
pol.DMG_BOOST_HOLD_FRAMES = 8

pol.FIRE_PHASE_MAX_WAIT = 280
pol.FIRE_PHASE_A_E4 = { 117, 125, 270, 276 }
pol.FIRE_PHASE_A_E6 = { 190, 198, 158, 172 }
pol.FIRE_PHASE_B_E4 = { 158, 165, 272, 276 }
pol.FIRE_PHASE_B_E6 = { 175, 182, 184, 190 }
pol.FIRE_PHASE_GERUTA_ID = 0xD63F
pol.FIRE_PHASE_SLOTS = { 4, 6 }

pol.FLOOR_RECLIMB_Y = 480
pol.FLOOR_RUNWAY_X = { 270, 310 }
pol.FLOOR_RUNWAY_Y = 500
pol.FLOOR_RECLIMB_CHARGE = 12
pol.FLOOR_RECLIMB_SPIN = 44

pol.PHASE_C_X_MIN = 300
pol.PHASE_C_Y_MAX = 430
pol.PHASE_C_Y_MIN = 200
pol.PHASE_D_X = pol.TOP_X
pol.PHASE_D_Y = pol.TOP_Y

pol.BUBBLE_PHASE_C_X_MIN = pol.PHASE_C_X_MIN
pol.BUBBLE_PHASE_C_Y_MAX = pol.PHASE_C_Y_MAX
pol.BUBBLE_PHASE_C_Y_MIN = pol.PHASE_C_Y_MIN
pol.BUBBLE_PHASE_D_X = pol.PHASE_D_X
pol.BUBBLE_PHASE_D_Y = pol.PHASE_D_Y

pol.HIJUMP_WALLJUMP_VY0 = 5.33
pol.REGULAR_WALLJUMP_VY0 = 4.41
pol.DAMAGE_BOOST_HX = 5.25

pol.POSE_KNOCKBACK = geometry.POSE_KNOCKBACK
pol.POSE_STAND_LEFT = geometry.POSE_STAND_LEFT
pol.POSE_STAND_RIGHT = geometry.POSE_STAND_RIGHT

pol.DOOR_WJ_POSES = {
  [81] = true,
  [82] = true,
  [83] = true,
  [84] = true,
  [132] = true,
}
pol.DOOR_FALL_Y = 220

pol.R15_WJ1 = controller.WallJumpTiming({
  into = "LEFT",
  flip = "RIGHT",
  into_frames = pol.SAVE_WJ_LEFT_A,
  amid_frames = pol.SAVE_WJ_AMID,
  flip_frames = pol.SAVE_WJ_RIGHT_A,
  delay_into_frames = 0,
})
pol.R15_WJ2 = controller.WallJumpTiming({
  into = "LEFT",
  flip = "RIGHT",
  into_frames = pol.SAVE_WJ2_LEFT_A,
  amid_frames = pol.SAVE_WJ2_AMID,
  flip_frames = pol.SAVE_WJ2_RIGHT_A,
  delay_into_frames = 0,
})
pol.R15_DOUBLE = { pol.R15_WJ1, pol.R15_WJ2 }

pol.DOOR_WJ = controller.WallJumpTiming({
  into = "LEFT",
  flip = "RIGHT",
  into_frames = pol.DOOR_WJ_INTO,
  amid_frames = 1,
  flip_frames = pol.DOOR_WJ_BOUNCE + 2,
})

return pol
