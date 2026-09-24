"""Gleeok arms for ``ab_stage.sh`` (stage_replay over offsets x pins). Scratch."""
from zelda_i.level6.gleeok18 import Level6Gleeok18Controller
from zelda_i.level8.gleeok import Level8FourHeadGleeokController


def l8_dy(stand_dy: int):
    return lambda: Level8FourHeadGleeokController(stand_dy=stand_dy)


def l6_dy(stand_dy: int):
    return lambda: Level6Gleeok18Controller(stand_dy=stand_dy)


l8_dy22, l8_dy30, l8_dy38 = l8_dy(22), l8_dy(30), l8_dy(38)
l6_dy22, l6_dy30, l6_dy38 = l6_dy(22), l6_dy(30), l6_dy(38)
