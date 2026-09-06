"""From Level7Interior1AReconFixture: fight with the CENTER-COLUMN goriyas
(slots that spawn at x~128 and will descend into the sealed cross) targeted
first, before they get trapped.  Report whether the room clears (rad != 0) and
the 0x68 then pushes UP.  Repeat N trials for flakiness.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \
        nes/zelda_i/scratch/probe_l7_1a_center_first.py --tag 1a_cf_v1 --trials 3
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.combat import manhattan
from zelda_i.dungeon.behaviors import EnemyKind, engagement_hint
from zelda_i.level7.path import live_goriyas
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.runner import make_assist

ROOM = 0x1A
CELLAR_MODES = {9, 10, 11}


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, a, btn, f):
    env.step(nes_idle_action() if btn is None
             else nes_action(*btn) if isinstance(btn, tuple) else nes_action(btn))
    if a:
        a.apply_env(env, frame=f)


def _danger(o):
    """Goriya heading for / inside the sealed center cross."""
    return 104 <= int(o.x) <= 152 and 100 <= int(o.y) <= 176


def _g(s):
    return [[int(o.x), int(o.y), int(o.hp)] for o in live_goriyas(s)]


def _priority_target(s):
    live = live_goriyas(s)
    if not live:
        return None
    danger = [o for o in live if _danger(o)]
    pool = danger or live
    return min(pool, key=lambda o: manhattan(s.link_x, s.link_y, o.x, o.y))


def _trial(env, a):
    reset_obs(env)
    for _ in range(2):
        env.step(nes_idle_action())
    f = 0
    out = {"start_g": _g(_s(env))}
    for i in range(9000):
        s = _s(env)
        if int(s.screen) != ROOM or int(s.mode) in CELLAR_MODES:
            break
        tgt = _priority_target(s)
        if tgt is None:
            out["clear_f"] = f
            break
        hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
        d = manhattan(s.link_x, s.link_y, tgt.x, tgt.y)
        btn = (hint.face, "A") if d <= 24 and (i % 6) < 3 else hint.face
        _step(env, a, btn, f)
        f += 1
    s = _s(env)
    out["after_fight"] = {"f": f, "rad": int(s.room_all_dead), "g": _g(s),
                          "xy": [int(s.link_x), int(s.link_y)]}
    # try push
    for _ in range(400):
        s = _s(env)
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - 96) <= 3 and abs(y - 162) <= 4:
            break
        if abs(y - 162) > 4:
            _step(env, a, "UP" if y > 162 else "DOWN", f)
        else:
            _step(env, a, "LEFT" if x > 96 else "RIGHT", f)
        f += 1
    pushed = False
    for _ in range(240):
        s = _s(env)
        if int(s.mode) in CELLAR_MODES:
            break
        blk = [(int(o.x), int(o.y)) for o in s.objects
               if int(o.type_id) == 0x68 and 1 <= int(o.slot) <= 12]
        if blk and blk[0][1] <= 132:
            pushed = True
            break
        _step(env, a, "UP", f)
        f += 1
    s = _s(env)
    out["push"] = {"pushed": pushed, "mode": int(s.mode),
                   "scr": f"0x{int(s.screen):02x}", "rad": int(s.room_all_dead),
                   "g": _g(s)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1a_cf_v1")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, "Level7Interior1AReconFixture", GAME_DIR,
                   render_mode="rgb_array")
    results = []
    try:
        for t in range(args.trials):
            r = _trial(env, a)
            results.append(r)
            print(f"TRIAL {t}: clear_f={r.get('clear_f')} "
                  f"after={r['after_fight']} push={r['push']}")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(
            json.dumps(results, indent=1))
    finally:
        env.close()


if __name__ == "__main__":
    main()
