-- K4 Wave branch: Bubble → Single → Double → Wave (+ return).

local G = require("wave.geometry")
local bubble_to_single = require("wave.bubble_to_single")
local single_to_double = require("wave.single_to_double")
local double_to_wave = require("wave.double_to_wave")
local wave_to_double = require("wave.wave_to_double")
local double_to_single = require("wave.double_to_single")
local single_to_bubble = require("wave.single_to_bubble")
local bubble_to_farm = require("wave.bubble_to_farm")
local farm_to_speedway = require("wave.farm_to_speedway")
local speedway_to_frog = require("wave.speedway_to_frog")
local frog_to_business = require("wave.frog_to_business")

return {
  WAVE_BEAM_MASK = G.WAVE_BEAM_MASK,
  play_bubble_to_single_chamber = bubble_to_single.play_bubble_to_single_chamber,
  play_single_to_double_chamber = single_to_double.play_single_to_double_chamber,
  play_double_chamber_to_wave = double_to_wave.play_double_chamber_to_wave,
  play_wave_to_double_chamber = wave_to_double.play_wave_to_double_chamber,
  play_double_to_single_chamber = double_to_single.play_double_to_single_chamber,
  play_single_to_bubble = single_to_bubble.play_single_to_bubble,
  play_bubble_to_farm = bubble_to_farm.play_bubble_to_farm,
  play_farm_to_speedway = farm_to_speedway.play_farm_to_speedway,
  play_speedway_to_frog_save = speedway_to_frog.play_speedway_to_frog_save,
  play_frog_save_to_business = frog_to_business.play_frog_save_to_business,
  ROOM_BUBBLE = G.ROOM_BUBBLE,
  ROOM_SINGLE_CHAMBER = G.ROOM_SINGLE_CHAMBER,
  ROOM_DOUBLE_CHAMBER = G.ROOM_DOUBLE_CHAMBER,
  ROOM_WAVE = G.ROOM_WAVE,
}
