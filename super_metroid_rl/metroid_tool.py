import argparse
import os
import time
import numpy as np
import cv2
import torch
import stable_retro as retro
from stable_baselines3 import PPO

# Import generic env logic
from metroid_env import make_metroid_env, ActionHoldRepeat, MetroidConfig, MODEL_DIR

# Default to latest best manual PPO
DEFAULT_MODEL = os.path.join(MODEL_DIR, "boss_ppo_final.zip")
FPS = 60.0
SCALE = 2

def find_wrapper(env, wrapper_class):
    """Traverse wrapper stack to find specific wrapper type."""
    current = env
    while hasattr(current, 'env'):
        if isinstance(current, wrapper_class):
            return current
        current = current.env
    return None

def record_video(args):
    print(f"Recording {args.episodes} episodes to {args.output}...")
    print(f"State: {args.state}, Scenario: {args.scenario}")
    
    # 1. Create Env
    # Use delayed termination for BOSS scenario to capture explosion
    delayed_term = (args.scenario == 'boss')
    
    env = make_metroid_env(
        state=args.state, 
        scenario=args.scenario,
        render_mode='rgb_array', 
        delayed_termination=delayed_term
    )
    
    # 2. Hook VideoWriter into ActionHoldRepeat
    ahr = find_wrapper(env, ActionHoldRepeat)
    if not ahr:
        print("CRITICAL: ActionHoldRepeat wrapper not found! Video will be fast/choppy.")
        return
    
    video_writer = [None] # Mutable container
    
    def render_hook(obs, info):
        frame = obs
        if video_writer[0] is None:
            h, w, c = frame.shape
            out_h, out_w = h * SCALE, w * SCALE
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer[0] = cv2.VideoWriter(args.output, fourcc, FPS, (out_w, out_h))
            print(f"Initialized VideoWriter: {out_w}x{out_h} @ {FPS}fps")

        if video_writer[0]:
            h, w, c = frame.shape
            # Resize
            resized = cv2.resize(frame, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
            
            # Overlay (Generic)
            font = cv2.FONT_HERSHEY_SIMPLEX
            samus_hp = info.get('health', 0)
            boss_hp = info.get('expected_boss_hp', info.get('boss_hp', 0))
            
            # Samus HP
            cv2.putText(resized, f"HPV: {samus_hp}", (10, h*SCALE - 20), font, 0.7, (255, 255, 255), 2)
            
            # Boss HP (Only show if relevant)
            if args.scenario == 'boss':
                 cv2.putText(resized, f"BOSS: {boss_hp}", (w*SCALE - 150, h*SCALE - 20), font, 0.7, (0, 0, 255), 2)
            else:
                 # Show X pos for Nav
                 samus_x = info.get('samus_x', 0)
                 cv2.putText(resized, f"X: {samus_x}", (w*SCALE - 150, h*SCALE - 20), font, 0.7, (0, 255, 0), 2)

            frame_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            video_writer[0].write(frame_bgr)

    ahr.render_fn = render_hook
    
    # 3. Load Model (Optional - if None, random actions? Or requiring model?)
    if args.model and os.path.exists(args.model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {args.model}...")
        try:
             # Hack for loading models with custom objects if pickle issues arise
            custom_objects = {
                'learning_rate': 0.0,
                'lr_schedule': lambda _: 0.0,
                'clip_range': lambda _: 0.0,
            }
            model = PPO.load(args.model, env=env, device=device, custom_objects=custom_objects)
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print("No valid model provided. Running random actions? (Or crashing if you expected a model)")
        model = None 

    # 4. Run Loop
    wins = 0
    total_episodes = args.episodes
    
    for ep in range(total_episodes):
        print(f"Starting Episode {ep+1}...")
        obs, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            if model:
                obs_batch = np.expand_dims(obs, axis=0) # (1, C, H, W)
                action, _ = model.predict(obs_batch, deterministic=False)
                action = action[0]
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            
        # End of episode stats
        if args.scenario == 'boss':
            if info.get('total_damage_dealt', 0) >= 800 or info.get('boss_hp', 100) <= 0:
                print(f"Episode {ep+1}: WIN")
                wins += 1
            else:
                print(f"Episode {ep+1}: LOSS")
        else:
             print(f"Episode {ep+1}: Finished. X_Pos: {info.get('samus_x')}")
            
    if video_writer[0]:
        video_writer[0].release()
        
    env.close()
    if args.scenario == 'boss':
        print(f"Done. Saved to {args.output}. Win Rate: {wins}/{total_episodes}")
    else:
        print(f"Done. Saved to {args.output}.")

def evaluate_model(args):
    # Simplified eval
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metroid RL Tool")
    subparsers = parser.add_subparsers(dest="command")
    
    # Record
    rec_parser = subparsers.add_parser("record", help="Record gameplay video")
    rec_parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    rec_parser.add_argument("--output", type=str, default="metroid_run.mp4", help="Output MP4 file")
    rec_parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model checkpoint path")
    rec_parser.add_argument("--state", type=str, default="BossTorizo", help="Game State (e.g. BossTorizo, LandingSite)")
    rec_parser.add_argument("--scenario", type=str, default="boss", choices=['boss', 'nav'], help="Scenario type")
    
    args = parser.parse_args()
    
    if args.command == "record":
        record_video(args)
    else:
        parser.print_help()
