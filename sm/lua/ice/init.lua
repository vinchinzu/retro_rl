-- K4 Ice stack: Business Super → Gate → Acid → Snake → Ice PLM (+ return).

local G = require("ice.geometry")
local business_to_gate = require("ice.business_to_gate")
local gate_to_acid = require("ice.gate_to_acid")
local acid_to_snake = require("ice.acid_to_snake")
local snake_to_ice = require("ice.snake_to_ice")
local ice_to_snake = require("ice.ice_to_snake")
local snake_to_tutorial = require("ice.snake_to_tutorial")
local tutorial_to_gate = require("ice.tutorial_to_gate")
local gate_to_business = require("ice.gate_to_business")
local spine = require("ice.spine")

return {
  ICE_BEAM_MASK = G.ICE_BEAM_MASK,
  ICE_SUPER_DOOR_X = G.ICE_SUPER_DOOR_X,
  ICE_SUPER_LIP_X_MAX = G.ICE_SUPER_LIP_X_MAX,
  ICE_SUPER_Y_MAX = G.ICE_SUPER_Y_MAX,
  ICE_SUPER_Y_MIN = G.ICE_SUPER_Y_MIN,
  ROOM_ICE = G.ROOM_ICE,
  ROOM_ICE_ACID = G.ROOM_ICE_ACID,
  ROOM_ICE_GATE = G.ROOM_ICE_GATE,
  ROOM_ICE_SNAKE = G.ROOM_ICE_SNAKE,
  ROOM_ICE_TUTORIAL = G.ROOM_ICE_TUTORIAL,
  on_ice_super_lip = G.on_ice_super_lip,
  has_ice = G.has_ice,
  play_business_to_ice_gate = business_to_gate.play_business_to_ice_gate,
  play_ice_acid_to_snake = acid_to_snake.play_ice_acid_to_snake,
  play_ice_gate_to_acid = gate_to_acid.play_ice_gate_to_acid,
  play_ice_gate_to_business = gate_to_business.play_ice_gate_to_business,
  play_ice_snake_to_ice = snake_to_ice.play_ice_snake_to_ice,
  play_ice_snake_to_tutorial = snake_to_tutorial.play_ice_snake_to_tutorial,
  play_ice_tutorial_to_gate = tutorial_to_gate.play_ice_tutorial_to_gate,
  play_ice_to_snake = ice_to_snake.play_ice_to_snake,
  play_business_to_ice = spine.play_business_to_ice,
  ICE_COLLECT_LEAVE = spine.ICE_COLLECT_LEAVE,
}
