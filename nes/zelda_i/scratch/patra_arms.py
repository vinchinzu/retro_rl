"""Patra arms for ``ab_stage.sh``: module knobs patched per process. Scratch."""
import zelda_i.level9.patra as pa
from zelda_i.level9.natural_path import NaturalFinalPatraController
from zelda_i.level9.prefix import make_stairs_61_controller


def arm(clear: int, dist: int, cd: int, room: str):
    def make():
        pa.PATRA_MIN_CLEAR = clear
        pa.PATRA_ATTACK_COOLDOWN = cd
        if room == "52":
            return NaturalFinalPatraController(stand_dy=dist)
        ctl = make_stairs_61_controller()
        ctl.patra_stand_dy = dist
        return ctl
    return make


p52_c44_d64, p52_c56_d84_cd8, p52_c44_d84 = arm(44, 64, 12, "52"), arm(56, 84, 8, "52"), arm(44, 84, 12, "52")
p61_c44_d64, p61_c56_d84_cd8, p61_c44_d84 = arm(44, 64, 12, "61"), arm(56, 84, 8, "61"), arm(44, 84, 12, "61")
p52_cd8, p61_cd8, p52_cd6, p61_cd6 = arm(56, 84, 8, "52"), arm(56, 84, 8, "61"), arm(56, 84, 6, "52"), arm(56, 84, 6, "61")
