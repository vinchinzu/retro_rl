import sys, json
from pathlib import Path
import numpy as np
from PIL import Image
from alttp.startup import build_boot_env, build_blank_sram, inject_sram_into_env, action_for, no_action
from alttp.scripts.audit_power_on import Survival, Recorder
from alttp.spine.tape import LiveRun, ram_crc
from alttp.ram import read_snapshot, snapshot_to_diag, read_inventory, read_u16
from alttp.primitives import move_to, Waypoint, active_sprites, settle_control
from alttp.opening_route.room_engine import run_room_edge
from alttp.dungeon.tilemap import read_room_tiles
from retro_harness.env import write_state_bytes
from alttp.paths import INTEGRATION_DIR
raw=build_boot_env(state_name='NONE')
raw.reset()
inject_sram_into_env(raw,build_blank_sram())
survival=Survival(True)
rec=Recorder(raw,survival)
live=LiveRun(rec.attach(),provider='sanctuary_development')
e=live.env
rec.live=e
out=Path('snes/alttp/recordings/sanctuary_20261001')
def glance():
 s=read_snapshot(e.get_ram()); print(json.dumps(snapshot_to_diag(s))); print('layer',int(e.get_ram()[0xEE]),'floor',int(e.get_ram()[0xA4]),'bigkey',read_inventory(e.get_ram()).big_key,'frames',len(rec.buttons)); print([(q.slot,hex(q.sprite_type),q.x,q.y,q.hp) for q in active_sprites(e)])
def press(btn,n):
 for _ in range(n): e.step(action_for(*btn.split('+')) if btn else no_action())
 glance()
def go(x,y,tol=3):
 r=move_to(e,Waypoint(x,y,tolerance=tol),max_frames=1000); print(r.ok,r.reason,r.frames); glance()
def shot(name='now'):
 Image.fromarray(raw.render()).save(out/(name+'.png'))
def save(name):
 checkpoints[name]=e.em.get_state(); shot(name)
def restore(name):
 e.em.set_state(checkpoints[name]); e.data.update_ram(); glance()
checkpoints={}
with np.load('snes/alttp/recordings/audit_power_on_20261001/power_on_controller.tape.npz',allow_pickle=False) as tape:
 for i,a in enumerate(tape['buttons']):
  e.step(a)
  if ram_crc(e.get_ram()) != int(tape['frame_crc'][i]): raise RuntimeError(f'desync {i+1}')
save('room81_entry'); glance()
