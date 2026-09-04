"""Recon: 0x0D fine colliding_tile sweep (poke-read only, no state advance).

Prints an ASCII map of y=0x55..0x88 across x=0x20..0xCF: '.'=floor,
'#'=diamond 0xB0-0xB3, 'S'=stair 0x70-0x73, 'W'=warp/mode-change.
Pass --push to run the verified 0x68 RIGHT push first.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=.:nes:snes uv run python \
        nes/zelda_i/scratch/probe_l7_room0d_tilesweep.py [--push]

Finding (2026-09-03): the y~101-115 band is solid + non-bombable across the
whole room width; the NE staircase pocket has no walk-on route.
"""

import sys
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action, nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot, ADDR_LINK_X, ADDR_LINK_Y
from zelda_i.runner import make_assist

configure_headless()
a = make_assist(True)
env = make_env(GAME, "Level7Interior0DClearedReconFixture", GAME_DIR, render_mode="rgb_array")
reset_obs(env)
for _ in range(2): env.step(nes_idle_action())
mem = env.unwrapped.data.memory
push = "--push" in sys.argv
def s(): return read_snapshot(env.get_ram())
def reach(tx,ty,budget=400,tol=2):
    for _ in range(budget):
        st=s(); x,y=int(st.link_x),int(st.link_y)
        if int(st.screen)!=0x0d or int(st.mode) in (9,10,11,16): return
        if abs(x-tx)<=tol and abs(y-ty)<=tol: return
        if abs(y-ty)>tol: b="UP" if y>ty else "DOWN"
        else: b="LEFT" if x>tx else "RIGHT"
        env.step(nes_action(b)); a.apply_env(env,frame=0)
if push:
    bl=[o for o in s().objects if int(o.type_id)==0x68]
    bx,by=int(bl[0].x),int(bl[0].y)
    for wx,wy in ((160,141),(176,144),(bx-16,by)): reach(wx,wy)
    for _ in range(30):
        st=s()
        if abs(int(st.link_y)-by)<=1: break
        env.step(nes_action("DOWN" if int(st.link_y)<by else "UP")); a.apply_env(env,frame=0)
    started=False
    for _ in range(120):
        st=s(); cb=[o for o in st.objects if int(o.type_id)==0x68]
        cbx=int(cb[0].x) if cb else bx; cby=int(cb[0].y) if cb else by
        if not started and (cbx!=bx or cby!=by): started=True
        if started and (abs(cbx-bx)>=16 or cby!=by): break
        env.step(nes_action("RIGHT") if not started else nes_idle_action()); a.apply_env(env,frame=0)
    for _ in range(24): env.step(nes_idle_action()); a.apply_env(env,frame=0)
    print("block now", [(int(o.x),int(o.y)) for o in s().objects if int(o.type_id)==0x68])
# fine sweep x 0x20..0xCF, y 0x55..0x88
rows={}
for py in range(0x55,0x89,2):
    line=[]
    for px in range(0x20,0xD0,4):
        mem.assign(int(ADDR_LINK_X),"|u1",px); mem.assign(int(ADDR_LINK_Y),"|u1",py)
        env.step(nes_idle_action())
        st=s()
        t=int(st.colliding_tile)
        mark = "." if t in (116,117,118,119) else ("#" if t in (176,177,178,179) else ("S" if 112<=t<=115 else "?"))
        if int(st.mode) in (9,10,11,16) or int(st.screen)!=0x0d: mark="W"
        line.append(mark)
    print(f"y={py:3d} " + "".join(line))
env.close()
