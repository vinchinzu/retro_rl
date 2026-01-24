#!/usr/bin/env python3
import stable_retro as retro
import os
import argparse
import time

# Use Actions.ALL to ensure Start/Select are replayed correctly
def replay_demo(demo_path):
    print(f"Replaying demo: {demo_path}")
    
    if not os.path.exists(demo_path):
        print("Demo file not found!")
        return

    movie = retro.Movie(demo_path)
    movie.step()
    
    game = movie.get_game()
    state = movie.get_state()
    
    # We must use Actions.ALL if the demo was recorded with it (or contained Start/Select)
    env = retro.make(
        game=game,
        state=retro.State.NONE, # State comes from movie
        use_restricted_actions=retro.Actions.ALL,
        render_mode='human'
    )
    
    env.initial_state = state
    env.reset()
    
    # Track playback
    framerate = 0
    time_start = time.time()
    
    while movie.step():
        keys = []
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, 0))
            
        _obs, _rew, _term, _trunc, _info = env.step(keys)
        
        # Render
        env.render()
        
        # Rate limit
        time_current = time.time()
        dt = time_current - time_start
        if dt < 1.0 / 60.0:
            time.sleep(1.0 / 60.0 - dt)
        time_start = time.time()

    print("Replay finished.")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('demo_path', help='Path to .bk2 file')
    args = parser.parse_args()
    
    replay_demo(args.demo_path)
