-- Early Spazer Beam — mainline K2.2 (Below climb → collect → return → West).

local approach = require("spazer.approach")
local climb = require("spazer.climb")
local collect = require("spazer.collect")
local detour = require("spazer.detour")
local drop = require("spazer.drop")
local g = require("spazer.geometry")
local geo = require("red_tower.ctrl")

local M = {
  ROOM_SPAZER = geo.ROOM_SPAZER,
  SPAZER_BEAM_MASK = g.SPAZER_BEAM_MASK,
  play_below_spazer_climb = climb.play_below_spazer_climb,
  play_below_spazer_floor_to_mid = climb.play_below_spazer_floor_to_mid,
  play_below_spazer_mid_to_top = climb.play_below_spazer_mid_to_top,
  play_below_spazer_to_spazer = approach.play_below_spazer_to_spazer,
  play_spazer_collect = collect.play_spazer_collect,
  play_spazer_detour = detour.play_spazer_detour,
  play_spazer_return_to_below = collect.play_spazer_return_to_below,
  play_spazer_top_to_mid = drop.play_spazer_top_to_mid,
  play_spazer_top_to_west = drop.play_spazer_top_to_west,
}

return M
