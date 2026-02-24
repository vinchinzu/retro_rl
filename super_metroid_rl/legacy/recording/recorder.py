import os
import time
import stable_retro as retro
import pygame
import warnings
from typing import Tuple, Optional

# Suppress pkg_resources warning from pygame
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

class DemoRecorder:
    FINALIZE_WAIT_SECONDS = 6.0
    POLL_SLEEP_SECONDS = 1.5
    MAX_POLL_ATTEMPTS = 60

    def __init__(self, recording_dir, state_dir):
        self.recording_dir = recording_dir
        self.state_dir = state_dir
        
        # Setup Pygame
        os.environ['SDL_VIDEODRIVER'] = 'x11'
        pygame.init()
        pygame.font.init()
        pygame.joystick.init()
        self.font = pygame.font.SysFont('Arial', 16)
        
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

    def resolve_state_path(self, state_name):
        # 1. Custom Dir
        custom = os.path.join(self.state_dir, f"{state_name}.state")
        if os.path.exists(custom): return custom
        # 2. Retro built-in
        if state_name in retro.data.list_states("SuperMetroid-Snes"): return state_name
        # 3. Already a path
        if os.path.exists(state_name): return state_name
        return None

    def draw_hud(self, screen, info, frame_count, state_name):
        hud_surface = pygame.Surface((350, 80), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 180))
        screen.blit(hud_surface, (5, 5))

        col = (255, 255, 255)
        title = self.font.render(f"{state_name}", True, (255, 200, 0))
        screen.blit(title, (10, 10))
        
        hp = info.get('health', 0)
        hp_txt = self.font.render(f"HP: {hp} | Frame: {frame_count}", True, (0, 255, 0) if hp > 30 else (255, 0, 0))
        screen.blit(hp_txt, (10, 30))
        
        help_txt = self.font.render("Start+Sel+L+R: Reset | ESC: Quit", True, (200, 200, 200))
        screen.blit(help_txt, (10, 50))
        
        if (frame_count // 30) % 2 == 0:
            pygame.draw.circle(screen, (255, 0, 0), (330, 20), 8)

    def get_input(self):
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        action = [0] * 12
        quit_requested = False
        reset_requested = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_requested = True
        
        # Keyboard mapping
        if keys[pygame.K_RIGHT]: action[7] = 1
        if keys[pygame.K_LEFT]:  action[6] = 1
        if keys[pygame.K_DOWN]:  action[5] = 1
        if keys[pygame.K_UP]:    action[4] = 1
        if keys[pygame.K_z]:     action[0] = 1 
        if keys[pygame.K_a]:     action[1] = 1 
        if keys[pygame.K_LSHIFT]: action[2] = 1 
        if keys[pygame.K_RETURN]: action[3] = 1 
        if keys[pygame.K_x]:     action[8] = 1 
        if keys[pygame.K_s]:     action[9] = 1 
        if keys[pygame.K_q]:     action[10]= 1
        if keys[pygame.K_w]:     action[11]= 1
        
        if self.joystick:
            ax0 = self.joystick.get_axis(0)
            ax1 = self.joystick.get_axis(1)
            if ax0 > 0.4: action[7] = 1
            elif ax0 < -0.4: action[6] = 1
            if ax1 > 0.4: action[5] = 1
            elif ax1 < -0.4: action[4] = 1
            
            if self.joystick.get_button(0): action[0] = 1 
            if self.joystick.get_button(1): action[8] = 1 
            if self.joystick.get_button(2): action[1] = 1 
            if self.joystick.get_button(3): action[9] = 1 
            
            if self.joystick.get_button(4): action[10] = 1 
            if self.joystick.get_button(5): action[11] = 1 
            if self.joystick.get_button(6): action[2] = 1  
            if self.joystick.get_button(7): action[3] = 1 
            
            # COMBO: Start+Select+L+R
            if (self.joystick.get_button(7) and self.joystick.get_button(6) and 
                self.joystick.get_button(4) and self.joystick.get_button(5)):
                reset_requested = True
                quit_requested = True # Stop this session

        return action, quit_requested, reset_requested

    def record_session(self, start_state) -> Tuple[Optional[str], str]:
        """
        Runs a SINGLE recording session.
        Returns: (saved_file_path, next_action)
        """
        resolved = self.resolve_state_path(start_state)
        if not resolved:
            print(f"ERROR: State '{start_state}' not found")
            return None, "QUIT"

        print(f"\n>>> Loading State: {start_state} <<<")
        session_start_time = time.time()

        # Create a unique temp directory for this recording session
        temp_dir = os.path.join(self.recording_dir, f"temp_{int(session_start_time)}")
        os.makedirs(temp_dir, exist_ok=True)

        env = retro.make(
            game="SuperMetroid-Snes",
            state=resolved,
            render_mode='rgb_array',
            record=temp_dir,
            use_restricted_actions=retro.Actions.ALL
        )

        obs, info = env.reset()
        scale = 3
        h, w, c = obs.shape
        screen = pygame.display.set_mode((w*scale, h*scale))
        pygame.display.set_caption(f"Recording: {start_state}")
        
        clock = pygame.time.Clock()
        running = True
        reset_triggered = False
        frame_count = 0
        
        try:
            while running:
                action, quit_req, reset_req = self.get_input()
                if quit_req:
                    running = False
                if reset_req:
                    reset_triggered = True

                obs, _, term, trunc, info = env.step(action)
                frame_count += 1
                
                surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
                scaled = pygame.transform.scale(surf, (w*scale, h*scale))
                screen.blit(scaled, (0, 0))
                self.draw_hud(screen, info, frame_count, start_state)
                
                pygame.display.flip()
                clock.tick(60)

                if term or trunc:
                    reset_triggered = True
                    running = False 
        finally:
            env.close()
            # Give stable_retro time to finalize the recording file
            print("Waiting for recording to finalize...")
            time.sleep(self.FINALIZE_WAIT_SECONDS)

        if reset_triggered:
            # Clean up temp directory
            import shutil
            temp_dir = os.path.join(self.recording_dir, f"temp_{int(session_start_time)}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return None, "RESET"

        return self._find_latest_recording(session_start_time, start_state), "QUIT"

    def _find_latest_recording(self, session_start_time, state_name):
        # Look in the temp directory we created
        temp_dir = os.path.join(self.recording_dir, f"temp_{int(session_start_time)}")

        print(f"Searching {temp_dir} for recording files...")
        for i in range(self.MAX_POLL_ATTEMPTS):
            if i > 0:
                time.sleep(self.POLL_SLEEP_SECONDS)

            if not os.path.exists(temp_dir):
                print(f"DEBUG: Temp directory doesn't exist yet")
                continue

            candidates = []
            for f in os.listdir(temp_dir):
                if f.endswith('.bk2') and "verify" not in f:
                    full_p = os.path.join(temp_dir, f)
                    candidates.append((full_p, os.path.getmtime(full_p)))
                    print(f"DEBUG: Found candidate {f}")

            if candidates:
                # Pick the latest one
                candidates.sort(key=lambda x: x[1], reverse=True)
                target_file = candidates[0][0]

                # Create clean filename: {StateName}-{UnixTimestamp}.bk2
                timestamp = int(time.time())
                new_name = f"{state_name}-{timestamp}.bk2"
                new_path = os.path.join(self.recording_dir, new_name)

                try:
                    import shutil
                    # Move from temp dir to main recording dir
                    shutil.move(target_file, new_path)
                    # Clean up temp directory
                    shutil.rmtree(temp_dir)
                    print(f"Saved recording as: {new_name}")
                    return new_path
                except OSError as e:
                    print(f"Error moving file: {e}")
                    # Try to return original location
                    return target_file
                
            print(f"Polling {i+1}/{self.MAX_POLL_ATTEMPTS}...")
        
        print("WARNING: No new .bk2 file found.")
        print(f"Leaving temp dir for recovery: {temp_dir}")
        return None

    def close(self):
        pygame.quit()
