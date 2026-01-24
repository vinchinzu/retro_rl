
import stable_retro as retro
import os

# Force software rendering to avoid conflicts
os.environ['SDL_VIDEODRIVER'] = 'dummy'

def test_direction(direction_name):
    print(f"Testing direction: {direction_name}")
    env = retro.make(game="SuperMetroid-Snes", state="ZebesStart", render_mode='rgb_array')
    obs = env.reset()
    
    # SNES Buttons: B, Y, SEL, STA, U, D, L, R, A, X, L, R
    # Right = 7, Left = 6
    action_idx = 7 if direction_name == "RIGHT" else 6
    
    start_room = 0
    changed_rooms = False
    max_x = -99999
    min_x = 99999
    
    try:
        obs, info = env.reset()
        if len(info) == 0:
            # Step once to get info
            obs, reward, term, trunc, info = env.step([0]*12)
            
        start_room = info.get('room_id', 0)
        start_x = info.get('samus_x', 0)
        print(f"Start Room: {start_room:x}, Start X: {start_x}")
        
        for i in range(1000):
            action = [0] * 12
            action[action_idx] = 1 # Move
            action[0] = 1 # Dash (B) 
            
            # SHOOT CONSTANTLY (Button 9 = X)
            # Spam shoot (every other frame or 5 frames) to break door
            if i % 10 < 5:
                action[9] = 1 
            
            # Aim Angle? Default is straight.
            # Maybe Angle Up (4) or Down (5)? Blue doors are usually floor level.
            
            # Jump randomly to get over bumps
            if i % 60 > 40:
                 action[8] = 1 # Jump (A)
            
            obs, reward, term, trunc, info = env.step(action)
            
            curr_room = info.get('room_id', 0)
            curr_x = info.get('samus_x', 0)
            
            max_x = max(max_x, curr_x)
            min_x = min(min_x, curr_x)
            
            if curr_room != start_room and start_room != 0:
                print(f"SUCCESS! Changed room to {curr_room:x} at frame {i}")
                changed_rooms = True
                break
                
        print(f"Finished {direction_name}. X Range: {min_x} to {max_x}. Changed Room: {changed_rooms}\n")
        return changed_rooms
        
    finally:
        env.close()

if __name__ == "__main__":
    right_success = test_direction("RIGHT")
    left_success = test_direction("LEFT")
    
    if right_success and not left_success:
        print("CONCLUSION: GO RIGHT")
    elif left_success and not right_success:
        print("CONCLUSION: GO LEFT")
    else:
        print("CONCLUSION: INCONCLUSIVE (Both or Neither worked)")
