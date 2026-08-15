"""Diagnostic: L2-close bomb on 0x31, watch state/HP, then sword after blast."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

STATE = "Level5North56"
TYPE_31 = 0x31
FACE_E, FACE_W, FACE_S, FACE_N = 0x01, 0x02, 0x04, 0x08

def live(snap):
    if snap.mode != PLAY_MODE:
        return []
    return [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == TYPE_31 and o.hp > 0]

def mouth(d):
    f = int(d.facing)
    if f & FACE_E:
        return d.x + 12, d.y, 'LEFT'
    if f & FACE_W:
        return d.x - 12, d.y, 'RIGHT'
    if f & FACE_S:
        return d.x, d.y + 12, 'UP'
    if f & FACE_N:
        return d.x, d.y - 12, 'DOWN'
    return d.x, d.y, 'UP'

def goto(snap, tx, ty, tol=6):
    if abs(snap.link_x - tx) > tol:
        return nes_action('RIGHT' if snap.link_x < tx else 'LEFT')
    if abs(snap.link_y - ty) > tol:
        return nes_action('DOWN' if snap.link_y < ty else 'UP')
    return nes_idle_action()

def main():
    configure_headless()
    env = make_env(GAME, STATE, GAME_DIR, render_mode='rgb_array')
    assist = UnlimitedHealthAssist(enabled=True)
    reset_obs(env)
    env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    events = []
    bombs_used = 0
    phase = 'approach'
    place_cd = 0
    last_n = 3
    for f in range(2500):
        assist.apply_env(env, frame=f)
        snap = read_snapshot(env.get_ram())
        if snap.mode != PLAY_MODE:
            env.step(nes_idle_action())
            continue
        L = live(snap)
        if len(L) != last_n:
            events.append({'event': 'n_change', 'f': f, 'n': len(L), 'bombs': snap.bombs, 'hps': [o.hp for o in L], 'states': [o.state for o in L]})
            last_n = len(L)
        if f % 40 == 0:
            events.append({'f': f, 'phase': phase, 'n': len(L), 'bombs': snap.bombs, 'xy': [snap.link_x, snap.link_y], 'objs': [[o.slot, o.x, o.y, o.facing, o.hp, o.state] for o in L]})
        if not L:
            events.append({'f': f, 'event': 'all_dead', 'bombs': snap.bombs})
            break
        if snap.bombs <= 0 and place_cd <= 0 and phase != 'sword':
            phase = 'sword'
        if phase == 'sword':
            d = min(L, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
            dx, dy = d.x - snap.link_x, d.y - snap.link_y
            face = 'RIGHT' if abs(dx) >= abs(dy) and dx >= 0 else ('LEFT' if abs(dx) >= abs(dy) else ('DOWN' if dy >= 0 else 'UP'))
            env.step(nes_action(face, 'A'))
            continue
        if place_cd > 0:
            place_cd -= 1
            d = min(L, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
            _t, _u, face = mouth(d)
            retreat = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}[face]
            if place_cd > 50:
                env.step(nes_action(retreat))
            elif place_cd > 20:
                env.step(nes_action(face, 'A'))
            else:
                env.step(nes_idle_action())
            if place_cd == 0:
                phase = 'approach'
            continue
        d = min(L, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
        tx, ty, face = mouth(d)
        dist = abs(snap.link_x - d.x) + abs(snap.link_y - d.y)
        at_mouth = abs(snap.link_x - tx) <= 12 and abs(snap.link_y - ty) <= 12
        if snap.bombs > 0 and (at_mouth or dist <= 24):
            if dist > 14:
                env.step(goto(snap, d.x, d.y, tol=8))
                continue
            env.step(nes_action(face))
            env.step(nes_action(face, 'B'))
            bombs_used += 1
            place_cd = 95
            phase = 'watch'
            events.append({'event': 'placed', 'f': f, 'face': face, 'dist': dist, 'link': [snap.link_x, snap.link_y], 'dodo': [d.x, d.y, d.facing, d.hp, d.state], 'bombs': snap.bombs})
            continue
        env.step(goto(snap, tx, ty, tol=6))
    snap = read_snapshot(env.get_ram())
    L = live(snap)
    report = {'bombs_used': bombs_used, 'bombs_left': snap.bombs, 'live': len(L), 'hps': [o.hp for o in L], 'events': events}
    write_json_report(RECORDINGS_DIR / 'l5_56_diag.json', report)
    print('used', bombs_used, 'left', snap.bombs, 'live', len(L), 'hps', [o.hp for o in L])
    for e in events:
        if e.get('event'):
            print(e)
    env.close()

if __name__ == '__main__':
    main()
