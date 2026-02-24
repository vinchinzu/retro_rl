#!/usr/bin/env python3
import stable_retro as retro
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import time
import pygame
from enum import Enum

# Make sure window works on X11
os.environ['SDL_VIDEODRIVER'] = 'x11'

# Import Policy Architecture
from metroid_rewards import MetroidProgressReward

# =============================================================================
# MISSION PHASES
# =============================================================================
class MissionPhase(Enum):
    DESCENT = 1      # Going down to get Morph Ball
    RETURN = 2       # Going back up to Torizo after getting Morph Ball
    COMPLETE = 3     # Reached Torizo

# =============================================================================
# ROOM WAYPOINTS - Key rooms and navigation hints
# =============================================================================
ROOM_WAYPOINTS = {
    # Crateria rooms (DESCENT: go left/down, RETURN: go right/up)
    0x91F8: {"name": "Landing Site", "descent": "left", "return": "right"},
    0x92FD: {"name": "Crateria Pipe", "descent": "down", "return": "up", "has_vertical": True},
    0x93AA: {"name": "Crateria Tunnel", "descent": "left", "return": "right"},
    0x93D5: {"name": "Crateria Save Room", "descent": "left", "return": "right"},
    0x93FE: {"name": "Crateria Pre-Elevator", "descent": "down", "return": "up"},
    0x94CC: {"name": "Crateria Elevator", "descent": "elevator_down", "return": "elevator_up"},

    # Brinstar rooms
    0x96BA: {"name": "Brinstar Elevator", "descent": "down", "return": "elevator_up"},
    0x9AD9: {"name": "Brinstar Shaft", "descent": "down", "return": "up", "has_vertical": True},
    0x9B5B: {"name": "Brinstar Morph Tunnel", "descent": "left", "return": "right"},
    0x9E9F: {"name": "Morph Ball Room", "descent": "collect", "return": "right"},

    # Torizo area
    0x9804: {"name": "Bomb Torizo Room", "descent": None, "return": "boss"},
}

# Morph Ball item location (approximate)
MORPH_BALL_ROOM = 0x9E9F
MORPH_BALL_X_RANGE = (100, 300)  # X range where morph ball is

# Torizo room
TORIZO_ROOM = 0x9804

# =============================================================================
# POLICY NETWORK
# =============================================================================
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

def get_phase_direction(room, phase):
    """Get the navigation direction for a room based on current mission phase."""
    waypoint = ROOM_WAYPOINTS.get(room)
    if waypoint:
        if phase == MissionPhase.DESCENT:
            return waypoint.get("descent")
        elif phase == MissionPhase.RETURN:
            return waypoint.get("return")
    return None

def apply_direction_bias(action_mask, direction, frame):
    """Apply directional bias to action based on waypoint direction."""
    # Button indices: [0:B, 1:Y, 2:Select, 3:Start, 4:Up, 5:Down, 6:Left, 7:Right, 8:A, 9:X, 10:L, 11:R]
    if direction == "left":
        action_mask[6] = 1  # Left
        action_mask[7] = 0  # Clear Right
    elif direction == "right":
        action_mask[7] = 1  # Right
        action_mask[6] = 0  # Clear Left
    elif direction == "up":
        action_mask[4] = 1  # Up
        action_mask[8] = 1  # Jump
    elif direction == "down":
        action_mask[5] = 1  # Down
    return action_mask

def run_bot(model_path, state="ZebesStart"):
    print(f"Loading model: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init Env
    state_path = os.path.abspath(f'super_metroid_rl/custom_integrations/SuperMetroid-Snes/{state}.state')
    if not os.path.exists(state_path):
        state_path = state

    env = retro.make(
        game="SuperMetroid-Snes",
        state=state_path,
        use_restricted_actions=retro.Actions.ALL,
        render_mode='rgb_array',
        record="recordings"
    )
    env = MetroidProgressReward(env)

    # Init Pygame
    pygame.init()
    obs, info = env.reset()
    screen = pygame.display.set_mode((obs.shape[1]*2, obs.shape[0]*2))
    pygame.display.set_caption("Super Metroid Bot - Morph Ball Mission")
    clock = pygame.time.Clock()

    # Load Model
    gray = np.dot(obs[...,:3], [0.299, 0.587, 0.114])
    gray_small = gray[::2, ::2]
    input_shape = (gray_small.shape[0], gray_small.shape[1])

    policy = NavigationPolicy(input_shape, 12).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    # === STATE VARIABLES ===
    running = True
    total_reward = 0
    frame = 0
    jump_hold_timer = 0
    wiggle_timer = 0
    wiggle_btn = None

    # Stuck detection
    anchor_x = 0
    anchor_y = 0
    anchor_frame = 0
    stuck_counter = 0

    # Mission phase tracking
    mission_phase = MissionPhase.DESCENT
    has_morph_ball = False
    prev_room = None

    print("="*60)
    print("MISSION: Get Morph Ball and return to Bomb Torizo")
    print(f"Phase: {mission_phase.name}")
    print("="*60)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                
        # Preprocess Obs
        gray = np.dot(obs[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        gray_small = gray[::2, ::2]
        
        # Prepare Tensor
        t_obs = torch.from_numpy(gray_small).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, H, W)
        
        # Predict
        with torch.no_grad():
            logits = policy(t_obs)
            probs = torch.sigmoid(logits)
            
            # Threshold
            action_mask = (probs > 0.5).cpu().numpy()[0].astype(int)
            
            # Optional: Sample instead of threshold?
            # action_mask = (torch.rand_like(probs) < probs).cpu().numpy()[0].astype(int)
            
        # === GET CURRENT STATE ===
        x = info.get('samus_x', 0)
        y = info.get('samus_y', 0)
        room = info.get('room_id', 0)
        items = info.get('items', 0)

        # === CHECK FOR MORPH BALL ACQUISITION ===
        morph_ball_acquired = (items & 0x1) != 0
        if morph_ball_acquired and not has_morph_ball:
            has_morph_ball = True
            mission_phase = MissionPhase.RETURN
            print("="*60)
            print("*** MORPH BALL ACQUIRED! ***")
            print("Switching to RETURN phase - heading back to Torizo")
            print("="*60)

        # === CHECK IF REACHED TORIZO ===
        if room == TORIZO_ROOM and mission_phase == MissionPhase.RETURN:
            mission_phase = MissionPhase.COMPLETE
            print("="*60)
            print("*** MISSION COMPLETE - REACHED TORIZO! ***")
            print("="*60)

        # === ROOM TRANSITION LOGGING ===
        if room != prev_room:
            waypoint = ROOM_WAYPOINTS.get(room)
            room_name = waypoint["name"] if waypoint else "Unknown"
            direction = get_phase_direction(room, mission_phase)
            print(f"[ROOM] {room:04X} - {room_name} | Phase: {mission_phase.name} | Dir: {direction}")
            prev_room = room

        # === PULSED FIRING (prevent animation lock) ===
        if frame % 10 == 0:
            action_mask[1] = 1  # Y (Shoot)
            action_mask[9] = 1  # X (Special)
        else:
            action_mask[1] = 0
            action_mask[9] = 0

        # === PHASE-AWARE DIRECTION BIAS ===
        # Skip direction bias for rooms that need special navigation
        special_nav_rooms = [0x92fd, 0x94cc, 0x96ba, MORPH_BALL_ROOM]
        direction = get_phase_direction(room, mission_phase)
        if direction and direction not in ["elevator_down", "elevator_up", "collect", "boss"]:
            if room not in special_nav_rooms:
                apply_direction_bias(action_mask, direction, frame)

        # === JUMP HOLD LOGIC ===
        if action_mask[8] == 1 and jump_hold_timer == 0:
            jump_hold_timer = 35
        if jump_hold_timer > 0:
            action_mask[8] = 1
            jump_hold_timer -= 1

        # === STUCK DETECTION ===
        if frame == 0:
            anchor_x, anchor_y = x, y
            anchor_frame = 0
            stuck_counter = 0

        if frame - anchor_frame >= 60:
            dist = abs(x - anchor_x) + abs(y - anchor_y)
            if dist < 12:
                stuck_counter += 60
            else:
                stuck_counter = 0
            anchor_x, anchor_y = x, y
            anchor_frame = frame

        if abs(x - anchor_x) <= 2 and abs(y - anchor_y) <= 2:
            stuck_counter += 1

        # Debug print
        if frame % 120 == 0:
            print(f"[{room:04x}] Pos=({x},{y}) Stuck={stuck_counter} Phase={mission_phase.name}")

        # === RECOVERY LOGIC ===
        recovery_active = False

        # --- ELEVATOR HANDLING (Phase-aware) ---
        # Crateria Elevator (0x94CC)
        if room == 0x94cc:
            target_x = 426
            if mission_phase == MissionPhase.DESCENT:
                # Going DOWN to Brinstar
                if abs(x - target_x) < 30 and y > 100:
                    recovery_active = True
                    action_mask[:] = 0
                    if abs(x - target_x) > 3:
                        if x < target_x: action_mask[7] = 1
                        else: action_mask[6] = 1
                    else:
                        action_mask[5] = 1  # DOWN
                    if frame % 30 == 0:
                        print(f"ELEVATOR [0x94CC]: Going DOWN at ({x},{y})")
            elif mission_phase == MissionPhase.RETURN:
                # Going UP to Crateria - elevator is at bottom, need to ride up
                if abs(x - target_x) < 30 and y > 400:
                    recovery_active = True
                    action_mask[:] = 0
                    if abs(x - target_x) > 3:
                        if x < target_x: action_mask[7] = 1
                        else: action_mask[6] = 1
                    else:
                        action_mask[4] = 1  # UP
                    if frame % 30 == 0:
                        print(f"ELEVATOR [0x94CC]: Going UP at ({x},{y})")

        # Brinstar Elevator (0x96BA)
        if room == 0x96ba and not recovery_active:
            target_x = 426
            if mission_phase == MissionPhase.DESCENT:
                # Just arrived from Crateria, walk off platform
                if y < 150:
                    recovery_active = True
                    action_mask[:] = 0
                    action_mask[6] = 1  # Walk LEFT off platform
                    if frame % 30 == 0:
                        print(f"ELEVATOR [0x96BA]: Walking off platform at ({x},{y})")
            elif mission_phase == MissionPhase.RETURN:
                # Need to go UP - center and hold UP
                if abs(x - target_x) < 30 and y > 300:
                    recovery_active = True
                    action_mask[:] = 0
                    if abs(x - target_x) > 3:
                        if x < target_x: action_mask[7] = 1
                        else: action_mask[6] = 1
                    else:
                        action_mask[4] = 1  # UP
                    if frame % 30 == 0:
                        print(f"ELEVATOR [0x96BA]: Going UP at ({x},{y})")

        # --- MORPH BALL ROOM SPECIAL HANDLING ---
        if room == MORPH_BALL_ROOM and not recovery_active:
            if mission_phase == MissionPhase.DESCENT and not has_morph_ball:
                # Navigate to the morph ball pedestal
                if x > MORPH_BALL_X_RANGE[1]:
                    action_mask[:] = 0
                    action_mask[6] = 1  # Go LEFT towards item
                    recovery_active = True
                elif MORPH_BALL_X_RANGE[0] <= x <= MORPH_BALL_X_RANGE[1]:
                    # At the item, wait for pickup
                    action_mask[:] = 0
                    recovery_active = True
                    if frame % 30 == 0:
                        print(f"MORPH BALL ROOM: Waiting for pickup at ({x},{y})")

        # --- DOOR HANDLING (phase-aware) ---
        is_at_left_door = x < 200
        is_at_right_door = x > 700  # Adjust based on room width

        if not recovery_active and stuck_counter > 30:
            if mission_phase == MissionPhase.DESCENT and is_at_left_door:
                # Going left, stuck at left door
                recovery_active = True
                door_phase = (stuck_counter - 30) // 30 % 4
                action_mask[:] = 0
                if door_phase == 0: action_mask[7] = 1  # Back up right
                elif door_phase == 1: action_mask[6] = 1  # Turn left
                elif door_phase == 2:
                    action_mask[1] = 1  # Shoot
                    action_mask[9] = 1
                elif door_phase == 3: action_mask[6] = 1  # Enter left
                if stuck_counter % 60 == 0:
                    print(f"DOOR RECOVERY (LEFT): Phase {door_phase} at ({x},{y})")

            elif mission_phase == MissionPhase.RETURN and is_at_right_door:
                # Going right, stuck at right door
                recovery_active = True
                door_phase = (stuck_counter - 30) // 30 % 4
                action_mask[:] = 0
                if door_phase == 0: action_mask[6] = 1  # Back up left
                elif door_phase == 1: action_mask[7] = 1  # Turn right
                elif door_phase == 2:
                    action_mask[1] = 1  # Shoot
                    action_mask[9] = 1
                elif door_phase == 3: action_mask[7] = 1  # Enter right
                if stuck_counter % 60 == 0:
                    print(f"DOOR RECOVERY (RIGHT): Phase {door_phase} at ({x},{y})")

        # --- PIPE ROOM (0x92FD) - Vertical shaft navigation ---
        # This room requires special handling - it's a tall vertical shaft
        # Entry from Landing Site is at TOP RIGHT (x~1240, y~139)
        # Need to go LEFT first to get into the shaft, then DOWN
        if room == 0x92fd:
            if mission_phase == MissionPhase.DESCENT:
                recovery_active = True
                action_mask[:] = 0  # Clear all first

                # Zone-based navigation with stuck recovery
                if x > 800:
                    # Near right entrance - need to get LEFT into the shaft
                    if stuck_counter < 120:
                        # First: try walking left
                        action_mask[6] = 1  # LEFT
                        action_mask[0] = 1  # B (run)
                    elif stuck_counter < 240:
                        # Stuck - try jumping left
                        action_mask[8] = 1  # JUMP
                        action_mask[6] = 1  # LEFT
                    elif stuck_counter < 360:
                        # Still stuck - try shooting then left (break blocks)
                        if (stuck_counter // 15) % 2 == 0:
                            action_mask[1] = 1  # SHOOT
                        else:
                            action_mask[6] = 1  # LEFT
                    elif stuck_counter < 480:
                        # Try drop down
                        action_mask[5] = 1  # DOWN
                        action_mask[6] = 1  # LEFT
                    else:
                        # Cycle through all options
                        phase = (stuck_counter // 60) % 5
                        if phase == 0:
                            action_mask[6] = 1; action_mask[8] = 1  # jump left
                        elif phase == 1:
                            action_mask[5] = 1; action_mask[6] = 1  # down left
                        elif phase == 2:
                            action_mask[1] = 1; action_mask[6] = 1  # shoot left
                        elif phase == 3:
                            action_mask[7] = 1  # try going right briefly
                        else:
                            action_mask[8] = 1; action_mask[7] = 1  # jump right

                    if frame % 120 == 0:
                        print(f"PIPE ROOM: Zone RIGHT at ({x},{y}) stuck={stuck_counter}")

                elif x > 400:
                    # Middle of room - continue LEFT and DOWN
                    if stuck_counter < 60:
                        action_mask[6] = 1  # LEFT
                        action_mask[5] = 1  # DOWN
                    else:
                        # Stuck in middle - try jumping
                        phase = (stuck_counter // 45) % 4
                        if phase == 0:
                            action_mask[8] = 1; action_mask[6] = 1
                        elif phase == 1:
                            action_mask[5] = 1; action_mask[6] = 1
                        elif phase == 2:
                            action_mask[1] = 1  # shoot
                        else:
                            action_mask[8] = 1; action_mask[5] = 1  # jump + down

                    if frame % 120 == 0:
                        print(f"PIPE ROOM: Zone MIDDLE at ({x},{y}) stuck={stuck_counter}")

                else:
                    # Left side of room - navigate down the shaft OR exit at bottom
                    if y > 1000:
                        # In the lower area - need to find exit or continue down
                        if stuck_counter < 60:
                            # First: try going LEFT
                            action_mask[6] = 1  # LEFT
                            action_mask[0] = 1  # B (run)
                        elif stuck_counter < 180:
                            # Stuck on left - try going RIGHT to find drop point
                            action_mask[7] = 1  # RIGHT
                            action_mask[5] = 1  # DOWN (drop through platforms)
                        elif stuck_counter < 300:
                            # Try jumping to find alternate route
                            phase = (stuck_counter // 30) % 3
                            if phase == 0:
                                action_mask[8] = 1  # JUMP
                                action_mask[7] = 1  # RIGHT
                            elif phase == 1:
                                action_mask[8] = 1  # JUMP
                                action_mask[6] = 1  # LEFT
                            else:
                                action_mask[5] = 1  # DOWN
                                action_mask[7] = 1  # RIGHT
                        else:
                            # Cycle all options
                            phase = (stuck_counter // 45) % 5
                            if phase == 0:
                                action_mask[6] = 1; action_mask[1] = 1  # left + shoot
                            elif phase == 1:
                                action_mask[7] = 1; action_mask[5] = 1  # right + down
                            elif phase == 2:
                                action_mask[8] = 1; action_mask[6] = 1  # jump left
                            elif phase == 3:
                                action_mask[8] = 1; action_mask[7] = 1  # jump right
                            else:
                                action_mask[5] = 1  # down only

                        if frame % 120 == 0:
                            print(f"PIPE ROOM: LOWER AREA at ({x},{y}) stuck={stuck_counter}")
                    else:
                        # Still in the shaft - navigate down
                        action_mask[5] = 1  # DOWN (drop through platforms)

                        # Wiggle to find drop points
                        if (frame // 45) % 3 == 0:
                            action_mask[6] = 1  # LEFT
                        elif (frame // 45) % 3 == 1:
                            action_mask[7] = 1  # RIGHT

                        if stuck_counter > 90:
                            phase = (stuck_counter // 45) % 4
                            if phase == 0:
                                action_mask[8] = 1  # JUMP
                                action_mask[5] = 0
                            elif phase == 1:
                                action_mask[1] = 1  # shoot

                        if frame % 120 == 0:
                            print(f"PIPE ROOM: In SHAFT at ({x},{y}) stuck={stuck_counter}")

            elif mission_phase == MissionPhase.RETURN:
                # Going UP - need wall jumps
                recovery_active = True
                phase_in_pipe = (frame // 40) % 4

                if phase_in_pipe == 0:
                    action_mask[8] = 1  # JUMP
                    action_mask[6] = 1  # LEFT (toward wall)
                elif phase_in_pipe == 1:
                    action_mask[8] = 1  # JUMP
                    action_mask[7] = 1  # RIGHT (wall jump off)
                elif phase_in_pipe == 2:
                    action_mask[7] = 1  # Walk right
                    action_mask[8] = 0
                elif phase_in_pipe == 3:
                    action_mask[6] = 1  # Walk left
                    action_mask[8] = 0

                if frame % 120 == 0:
                    print(f"PIPE ROOM (RETURN): Phase {phase_in_pipe} at ({x},{y})")

        # --- GENERAL STUCK RECOVERY (phase-aware) ---
        if not recovery_active and stuck_counter > 60:
            recovery_active = True
            phase_duration = 60
            phase = (stuck_counter - 60) // phase_duration
            phase_cycle = phase % 5

            if stuck_counter % phase_duration == 0:
                phases = ["PULSE SHOOT", "WALK BACK", "WALK FORWARD", "JUMP & MOVE", "WIGGLE"]
                print(f"STUCK RECOVERY: {phases[phase_cycle]} at ({x},{y}) Phase={mission_phase.name}")

            if phase_cycle == 0:  # Pulse Shoot
                if (stuck_counter // 5) % 2 == 0:
                    action_mask[1] = 1
                    action_mask[9] = 1

            elif phase_cycle == 1:  # Walk BACK (opposite of mission direction)
                action_mask[:] = 0
                if mission_phase == MissionPhase.DESCENT:
                    action_mask[7] = 1  # Right (back)
                else:
                    action_mask[6] = 1  # Left (back)

            elif phase_cycle == 2:  # Walk FORWARD (mission direction)
                action_mask[:] = 0
                if mission_phase == MissionPhase.DESCENT:
                    action_mask[6] = 1  # Left (forward)
                else:
                    action_mask[7] = 1  # Right (forward)

            elif phase_cycle == 3:  # JUMP & MOVE
                action_mask[1] = 0
                action_mask[9] = 0
                action_mask[8] = 1
                if mission_phase == MissionPhase.DESCENT:
                    if (stuck_counter // 30) % 2 == 0: action_mask[6] = 1
                    else: action_mask[7] = 1
                else:
                    if (stuck_counter // 30) % 2 == 0: action_mask[7] = 1
                    else: action_mask[6] = 1

            elif phase_cycle == 4:  # WIGGLE
                if wiggle_timer <= 0:
                    wiggle_timer = 30
                    wiggle_btn = np.random.choice([4, 5, 6, 7, 8])
                action_mask[:] = 0
                action_mask[wiggle_btn] = 1
                wiggle_timer -= 1
        else:
            wiggle_timer = 0

        # === DEBUG SCREENSHOTS ===
        if (stuck_counter > 60 and stuck_counter % 30 == 0) or (frame % 600 == 0):
            os.makedirs("debug_screens", exist_ok=True)
            fname = f"debug_screens/frame_{frame:05d}_r{room:x}_{mission_phase.name}.png"
            surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
            pygame.image.save(surf, fname)
             # print(f"Saved screenshot: {fname}")

        # Step
        obs, reward, terminated, truncated, info = env.step(action_mask)
        total_reward += reward
        frame += 1
        
        # Render
        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        scaled = pygame.transform.scale(surf, (obs.shape[1]*2, obs.shape[0]*2))
        screen.blit(scaled, (0, 0))

        # Debug Text
        font = pygame.font.SysFont('monospace', 14)
        btn_names = ['B','Y','Sel','Sta','U','D','L','R','A','X','L','R']
        pressed = [btn_names[i] for i, v in enumerate(action_mask) if v]
        btn_str = ' '.join(pressed)

        # Line 1: Buttons
        text = font.render(f"Btns: {btn_str}", True, (0, 255, 0))
        screen.blit(text, (10, 10))

        # Line 2: Mission phase and morph ball status
        phase_color = (255, 255, 0) if mission_phase == MissionPhase.DESCENT else (0, 255, 255)
        if mission_phase == MissionPhase.COMPLETE:
            phase_color = (0, 255, 0)
        morph_str = "MORPH BALL: YES" if has_morph_ball else "MORPH BALL: NO"
        text2 = font.render(f"Phase: {mission_phase.name} | {morph_str}", True, phase_color)
        screen.blit(text2, (10, 28))

        # Line 3: Room info
        waypoint = ROOM_WAYPOINTS.get(room)
        room_name = waypoint["name"] if waypoint else "Unknown"
        text3 = font.render(f"Room: {room:04X} - {room_name}", True, (200, 200, 200))
        screen.blit(text3, (10, 46))

        pygame.display.flip()
        clock.tick(60)

        if terminated or truncated:
            print("Episode Restart")
            obs, info = env.reset()
            frame = 0
            stuck_counter = 0
            anchor_x, anchor_y = 0, 0
            anchor_frame = 0

    env.close()
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to .pth model')
    parser.add_argument('--state', default='ZebesStart', help='State to start from')
    args = parser.parse_args()
    
    run_bot(args.model_path, args.state)
