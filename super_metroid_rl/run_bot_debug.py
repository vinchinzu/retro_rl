#!/usr/bin/env python3
import stable_retro as retro
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import pygame
import shutil

# Make sure window works on X11
os.environ['SDL_VIDEODRIVER'] = 'x11'

class NavigationPolicy(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(NavigationPolicy, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_shape[0], input_shape[1])
            self.feature_size = self.features(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x.float() / 255.0
        x = self.features(x)
        x = self.fc(x)
        return x

def run_bot_debug(model_path, state="ZebesStart", max_frames=800):
    print(f"Loading model: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Init Env
    state_path = os.path.abspath(f'super_metroid_rl/custom_integrations/SuperMetroid-Snes/{state}.state')
    if not os.path.exists(state_path):
        state_path = state
        
    print(f"Loading state from: {state_path}")
    env = retro.make(
        game="SuperMetroid-Snes", 
        state=state_path, 
        use_restricted_actions=retro.Actions.ALL,
        render_mode='rgb_array'
    )
    
    # Init Pygame for rendering
    pygame.init()
    obs, info = env.reset()
    screen = pygame.display.set_mode((obs.shape[1]*2, obs.shape[0]*2))
    pygame.display.set_caption("Super Metroid Bot - DEBUG")
    clock = pygame.time.Clock()
    
    # Load Model
    gray = np.dot(obs[...,:3], [0.299, 0.587, 0.114])
    gray_small = gray[::2, ::2]
    input_shape = (gray_small.shape[0], gray_small.shape[1])
    
    policy = NavigationPolicy(input_shape, 12).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    
    running = True
    total_reward = 0
    frame = 0
    
    # Create debug directory
    debug_dir = os.path.abspath("debug_frames")
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)
    print(f"Saving debug frames to {debug_dir}")
    
    print("Bot started...")
    start_room = info.get('room_id', 0)
    print(f"Starting Room: {start_room:#06x}")
    
    btn_names = ['B','Y','Sel','Sta','U','D','L','R','A','X','L','R']

    while running and frame < max_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                
        # Preprocess Obs
        gray = np.dot(obs[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        gray_small = gray[::2, ::2]
        
        # Prepare Tensor
        t_obs = torch.from_numpy(gray_small).unsqueeze(0).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            logits = policy(t_obs)
            probs = torch.sigmoid(logits)
            action_mask = (probs > 0.5).cpu().numpy()[0].astype(int)
            
        # ALWAYS FIRE (User Request)
        action_mask[1] = 1 # Y (Shoot)
        action_mask[9] = 1 # X (Special)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action_mask)
        total_reward += reward
        frame += 1
        
        # Info logging
        x = info.get('samus_x', 0)
        y = info.get('samus_y', 0)
        room = info.get('room_id', 0)
        
        # Stuck Detection
        if 'last_x' not in locals():
            last_x = x
            stuck_counter = 0
            
        # Only count as stuck if we are trying to move AND position isn't changing
        # Movement keys: U(4), D(5), L(6), R(7)
        trying_to_move = any(action_mask[i] == 1 for i in range(4, 9)) # Include Jump(8) too
        
        # Check if position changed
        if abs(x - last_x) == 0:
            if trying_to_move:
                stuck_counter += 1
        else:
            stuck_counter = 0
            last_x = x
            
        # SPECIAL: Landing Site Door Logic (Room 0x91F8, Left side)
        # If we are stuck here, we MUST open the door.
        is_landing_site_door = (room == 0x91f8 and x < 200)
        
        if is_landing_site_door and stuck_counter > 30:
             # Door Breaching Protocol
             door_phase = (stuck_counter - 30) // 30
             phase_cycle = door_phase % 4
             
             if phase_cycle == 0: # Back up (Right)
                 print(f"DOOR STUCK: Backing up")
                 action_mask[:] = 0 # Clear ALL (Aiming prevents movement)
                 action_mask[7] = 1 # Right
                 
             elif phase_cycle == 1: # Stop and Face Left?
                 print(f"DOOR STUCK: Turning Left")
                 action_mask[:] = 0
                 action_mask[6] = 1 # Left (briefly)
                 
             elif phase_cycle == 2: # Stand clear and FIRE
                 print(f"DOOR STUCK: FIRE (X+Y)")
                 action_mask[:] = 0
                 action_mask[1] = 1 # Y (Shoot)
                 action_mask[9] = 1 # X (User requested "fire x weapon")
                 
             elif phase_cycle == 3: # Try to enter
                 print(f"DOOR STUCK: Entering")
                 action_mask[:] = 0
                 action_mask[6] = 1 # Left
                 
        elif stuck_counter > 60: # General Stuck Logic
            phase = (stuck_counter - 60) // 60
            phase_cycle = phase % 3
            
            if phase_cycle == 0: # Pulse Shoot
                 if (stuck_counter // 5) % 2 == 0:
                    # Keep movement? No, maybe aiming is problem here too.
                    # But generic stuck might be terrain.
                    # Let's just add Shoot to existing.
                    action_mask[1] = 1
                    action_mask[9] = 1
                    
            elif phase_cycle == 1: # Back up
                 # Determine back up direction? 
                 # We want to clear aiming here too!
                 aim_held = action_mask[10] or action_mask[11]
                 if aim_held:
                     action_mask[10] = 0
                     action_mask[11] = 0
                     
                 if action_mask[6]: 
                     action_mask[6] = 0
                     action_mask[7] = 1
                 elif action_mask[7]:
                     action_mask[7] = 0
                     action_mask[6] = 1
                     
            elif phase_cycle == 2: # Stop and Shoot
                 action_mask[:] = 0
                 action_mask[1] = 1
                 action_mask[9] = 1

        pressed = [btn_names[i] for i, v in enumerate(action_mask) if v]
        btn_str = ' '.join(pressed)
        
        if frame % 20 == 0:
            print(f"Frame {frame}: Room={room:#06x} Pos=({x},{y}) Btns={btn_str} Stuck={stuck_counter}")
            # Save screenshot
            fname = os.path.join(debug_dir, f"frame_{frame:04d}.png")
            surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
            pygame.image.save(surf, fname)

        # Render to window
        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        scaled = pygame.transform.scale(surf, (obs.shape[1]*2, obs.shape[0]*2))
        screen.blit(scaled, (0, 0))
        
        # Debug Text overlay on window
        font = pygame.font.SysFont('monospace', 14)
        text = font.render(f"{frame} {btn_str}", True, (0, 255, 0))
        screen.blit(text, (10, 10))
        
        pygame.display.flip()
        clock.tick(60) # Limit to 60 FPS actual time if possible
        
        if terminated or truncated:
            print("Episode Restart (Terminated/Truncated)")
            break

    env.close()
    pygame.quit()
    print("Debug run finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to .pth model')
    parser.add_argument('--state', default='ZebesStart', help='State to start from')
    args = parser.parse_args()
    
    run_bot_debug(args.model_path, args.state)
