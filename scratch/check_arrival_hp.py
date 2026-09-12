from retro_harness.env import make_env
from retro_harness.segment_runner import configure_headless
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.level4.spine import run_level4_entrance_tf

configure_headless()
env = make_env(GAME, "Level4Entrance", GAME_DIR, render_mode="rgb_array")
env.reset()
print("Starting run through level4-gleeok13...")
res = run_level4_entrance_tf(env, through="level4-gleeok13")
leftover = res.get("leftover", {})
print(f"Result: ok={res.get('ok')} failed={res.get('failed_stage')} frames={res.get('frames')}")
print(f"Arrival in 0x13: room=0x{int(leftover.get('room', 0)):02x} xy={leftover.get('xy')} hp={leftover.get('health')} deaths={leftover.get('deaths')}")
for stage in res.get("stages", []):
    print(f"  stage {stage.get('segment', stage.get('name'))}: frames={stage.get('frames')} ok={stage.get('success')}")
env.close()
