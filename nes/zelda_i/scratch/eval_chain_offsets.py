"""Gather chain from ``PreL1BombLeave`` under an RNG offset and an arm.

``--assist last`` is the spine's last-heart refill, ``none`` is rung 3.
``--save-at <stage>`` writes ``S1Dbg_<stage>`` before that stage. Scratch.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/eval_chain_offsets.py \
        --offset 7 --arm noevade --assist none --out /tmp/c_7.json
"""
import argparse, json
from pathlib import Path
from retro_harness.nes import nes_idle_action
from zelda_i.assist import LastHeartAssist
from zelda_i.overworld.gather_segments import chain_stages, CHAIN_FROM
from zelda_i.overworld.gather_segments import HopWalkController, GatherWhiteController
from zelda_i.overworld.gather_run import _stopped
from zelda_i.ram import read_snapshot
from zelda_i.runner import open_env
ap=argparse.ArgumentParser(); ap.add_argument('--offset',type=int,default=0); ap.add_argument('--arm',default='walk'); ap.add_argument('--assist',default='last'); ap.add_argument('--out'); ap.add_argument('--save-at'); ap.add_argument('--stop-after')
a=ap.parse_args()
stages=chain_stages()
for _,c in stages:
    if a.arm=='noevade' and isinstance(c,(HopWalkController,GatherWhiteController)): c.evade=False
assist=LastHeartAssist(enabled=(a.assist=='last'))
env=open_env(from_state=CHAIN_FROM)
for _ in range(a.offset): env.step(nes_idle_action())  # in the cave: nothing hurts
total=0; rows=[]; died=None
for name,c in stages:
    if a.save_at==name:
        from retro_harness.env import save_state
        from zelda_i.paths import GAME, GAME_DIR
        print('saved', save_state(env, GAME_DIR, GAME, 'S1Dbg_'+name), flush=True)
    if hasattr(c,'bind_env'): c.bind_env(env)
    limit=int(getattr(c,'max_frames',8000) or 8000)
    for f in range(1,limit+1):
        s=read_snapshot(env.get_ram()); act=c.step(s); env.step(act.action); total+=1
        assist.apply_env(env, frame=total)
        s=read_snapshot(env.get_ram())
        if s.mode==17 or (s.health&0x0F)==0 and s.heart_partial==0: died=name; break
        if _stopped(c): break
    s=read_snapshot(env.get_ram())
    ok=bool(getattr(c,'success',False)) and died is None
    rows.append({'stage':name,'ok':ok,'frames':f,'hearts':float(s.whole_hearts),'containers':int(s.heart_containers),'screen':hex(s.screen)})
    if not ok or a.stop_after==name: break
rep=assist.report()
out={'offset':a.offset,'arm':a.arm,'assist':a.assist,'ok':all(r['ok'] for r in rows) and len(rows)==len(stages),'last':rows[-1]['stage'],'died':died,'frames':total,
     'writes':(rep.get('health') or {}).get('writes'),'restored':(rep.get('health') or {}).get('restored'),'dmg':rep.get('total_damage'),'by':rep.get('damage_by_location'),'rows':rows}
Path(a.out).write_text(json.dumps(out,indent=1))
print(json.dumps({k:out[k] for k in ('offset','arm','assist','ok','last','died','frames','writes','dmg')}))
