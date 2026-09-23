"""Pre-L1 coast walk under an RNG offset (idle frames after boot) and an arm.

One tape cannot score a combat change: a change reshuffles every later frame.
Run this over many offsets and compare arms on shop reach, hits and causes.
Arms: ``old`` (no lattice), ``geo`` (lattice, no shot model), ``new`` (shot
model without rocks), anything else is the committed default. Scratch, not STATUS.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/eval_coast_offsets.py \
        --offset 13 --arm rocks --out /tmp/e_13.json
"""
import argparse, json
from pathlib import Path
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.overworld.gathering import SHOP_P7_WALK_MAX_FRAMES, SWORD_MAX, SwordCaveController, make_shop_p7_walk_controller
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.route.chain import boot_to_ready, run_controller_stage
ap=argparse.ArgumentParser(); ap.add_argument('--offset',type=int,default=0); ap.add_argument('--arm',default='new'); ap.add_argument('--out')
a=ap.parse_args()
configure_headless()
env=make_env(GAME,"NONE",GAME_DIR)
obs,_=reset_obs(env)
obs,boot=boot_to_ready(env, first_playthrough=True, assist=None)
for _ in range(a.offset): obs,*_=env.step(nes_idle_action())
sword=SwordCaveController()
obs,sr=run_controller_stage(env,obs,name='sword',controller=sword,max_frames=SWORD_MAX,assist=None,frame_base=boot+a.offset)
walk=make_shop_p7_walk_controller()
if a.arm=='old': walk.geo=False
elif a.arm=='geo': walk.shot_model=False
elif a.arm=='new': walk.shot_model_rocks=False
import collections
inner=walk.step; hist=collections.deque(maxlen=16); hitlog=[]; prev={'if':0}
def traced(snap):
    act=inner(snap)
    hist.append((int(snap.link_x),int(snap.link_y),act.reason))
    if prev['if']==0 and int(snap.link_iframes)>0 and snap.mode==5:
        hitlog.append({'screen':hex(snap.screen),'x':int(snap.link_x),'y':int(snap.link_y),'hist':list(hist)[::2],
          'objs':[(hex(o.type_id),int(o.x),int(o.y),int(o.state)) for o in snap.objects[1:] if o.type_id and abs(int(o.x)-int(snap.link_x))<40 and abs(int(o.y)-int(snap.link_y))<40]})
    prev['if']=int(snap.link_iframes)
    return act
walk.step=traced
obs,wr=run_controller_stage(env,obs,name='walk',controller=walk,max_frames=SHOP_P7_WALK_MAX_FRAMES,assist=None,frame_base=sr.end_frame)
s=read_snapshot(env.get_ram())
rep=walk.report(); h=rep.get('hunt') or {}
out={'offset':a.offset,'arm':a.arm,'ok':bool(walk.success),'end_screen':hex(s.screen),'mode':int(s.mode),'hearts':float(s.whole_hearts), 'partial':int(s.heart_partial),'rupees':int(s.rupees),'frames':int(walk.frames),
 'screens':[{k:sc[k] for k in ('screen','frames','hearts_in','hearts_out','hits','hits_by_cause','kills','rupees','heal_hearts')} for sc in h.get('screens',[])],
 'evade':rep.get('evade_reasons'),'hitlog':hitlog, 'notes':rep.get('notes')}
Path(a.out).write_text(json.dumps(out,indent=1))
print(json.dumps({k:out[k] for k in ('offset','arm','ok','end_screen','mode','hearts','rupees','frames')}))
