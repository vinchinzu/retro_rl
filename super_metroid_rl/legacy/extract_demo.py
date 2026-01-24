#!/usr/bin/env python3
import stable_retro as retro
import numpy as np
import argparse
import os

def extract_demo(demo_path, output_path):
    print(f"Extracting demo: {demo_path}")
    
    if not os.path.exists(demo_path):
        print(f"Error: {demo_path} not found")
        return

    movie = retro.Movie(demo_path)
    movie.step()
    
    game = "SuperMetroid-Snes" # Fixed for this task
    
    # Must use Actions.ALL to replay correctly
    env = retro.make(
        game=game,
        state=retro.State.NONE, 
        use_restricted_actions=retro.Actions.ALL
    )
    
    env.initial_state = movie.get_state()
    env.reset()
    
    obs_list = []
    act_list = []
    
    print("Processing frames...")
    count = 0
    while movie.step():
        # Get action from movie
        keys = []
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, 0))
            
        # Step env to get NEXT observation (or current? usually obs, act, next_obs)
        # For BC, we map Obs_t -> Action_t
        # So we need Obs BEFORE the step.
        
        # Get current observation
        # retro.make resets to initial state, so env.get_screen() gives initial frame
        
        # Capture current frame
        img = env.get_screen()
        
        # Store (Obs, Act)
        # Resize to 84x84 grayscale to save space? Or keep full?
        # Let's keep full for now, but maybe downsample to 128x128 to fit in RAM
        # SNES is 256x224.
        
        # Use simple grayscale conversion to save space: 0.299R + 0.587G + 0.114B
        gray = np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        
        # Downsample by 2
        gray_small = gray[::2, ::2]
        
        obs_list.append(gray_small)
        act_list.append(keys)
        
        # Execute step
        _obs, _rew, _term, _trunc, _info = env.step(keys)
        count += 1
        
        if count % 1000 == 0:
            print(f"  Processed {count} frames...")

    print(f"Extraction complete. Total frames: {count}")
    
    # Save to .npz
    print(f"Saving to {output_path}...")
    np.savez_compressed(output_path, obs=np.array(obs_list), acts=np.array(act_list))
    print("Done.")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('demo_path', help='Path to .bk2 file')
    parser.add_argument('output_path', help='Path to output .npz file')
    args = parser.parse_args()
    
    extract_demo(args.demo_path, args.output_path)
