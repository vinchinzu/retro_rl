"""Stage policies for Mega Man 2 (NES).

M3 isolated segments (Clean Bronze):

- **Air screen ≥ 1** from ``Level1``: periodic jump-run (legacy ``AirScreen1Policy``).
- **Air screen ≥ 2** from ``Level1`` or ``AirLanded``: multi-phase ``AirManPolicy``
  (~521f from Level1, ~225f from AirLanded; 3/3; verified 2026-08-08).
- **Air screen ≥ 3 / ≥ 4** from ``AirScreen2``: late-stage ``AirManPolicy(start=screen2)``
  (~241f → s3 HP20; ~502f → s4 HP16; 3/3; verified 2026-08-09).
- **Heat screen ≥ 1** from ``Heat1``: ``HeatManPolicy`` period 50/12 (~243f; 2026-08-10).
- **Heat screen ≥ 2** from ``HeatScreen1``: same early recipe (~194f; 2026-08-10).
- **Heat screen ≥ 3** from ``HeatScreen2``: mid 60/14 → late 25/12 (~351f grounded; 2026-08-10).
- **Heat screen ≥ 4** from ``HeatScreen3``: pillars 25/10 phase 10 (~181f grounded; 2026-08-10).
- **Heat screen ≥ 5** from ``HeatScreen4``: late 20/12 phase 4 (~131f cam / ~320f grounded; 2026-08-10).
- **Heat screen ≥ 7 pre-boss** from ``HeatScreen5Ground``: ``start=screen5`` A-edge
  j1/LEFT/j2 + hop 9/gw3 (~293f cam7; 3/3; 2026-08-10).
- **Heat cam ≥ 8 Sniper shaft** from ``HeatScreen7Mid``: ``start=screen7`` high-path
  past sx152 wall → ladder → scroll_down (~587f; 3/3; 2026-08-10).
- **HeatScreen8 Yoku room → cam ≥ 9** from ``HeatScreen8``: ``start=screen8``
  wait no-ceiling phase → first Yoku → catch upper → D → left ladder
  scroll_down (~680f cam9; 2026-08-29). Residual: E columns / F lava Yoku /
  G Sniper → boss door + Item-1 (rr-k1ea / rr-809 PARTIAL).

Level1 recipe (0-based frame index ``i``):

1. ``i < 180``: RIGHT + jump period 50 / hold 12 (+ shoot pulse)
2. ``216 ≤ i < 230``: long land jump onto screen-1 platforms
3. After land (``i ≥ 301`` at grounded AirLanded pose):
   - relative ``r = i - 301``: period 50/12 until ``r = 142``
   - gap jump ``142 ≤ r < 156``
4. Hold clear when ``camera_x_screen ≥ target``

From ``AirLanded``, only phase 3 applies with ``r = i``.

AirScreen2 late recipe (0-based ``i``):

1. ``48 ≤ i < 145``: approach period 45 / hold 16
2. ``145 ≤ i < 180``: continuous jump (fan / tall-gap boost)
3. ``i ≥ 180``: late period 40 / hold 16 through screens 3–4
4. Hold clear when ``camera_x_screen ≥ target``

Post-screen-4 bottleneck (2026-08-09/10 probe, rr-54ui — open):

- Last solid under screen2 recipe: ``AirFanPlatform`` (prog~949, sy84). Solid
  extent **prog 937–984** (tiles: ``tile_feet``/``tile_center``). Pink head
  LEFT = type36 damage enemy (slot14 @~39,49) — periodic teleport-hit when
  inv=0; **not** landable. ``AirLeftPlatform`` short ledge prog~902–905.
  Ladder bar never ``tile_feet==2``. No wind; camera_y=0.
- Gap to screen5 (prog 1280) ≈ **296px**; pure RIGHT jump max ~1065–1071;
  Pipi boost ~1086 still pit. Freefall tile sample past 984 = 0 solids.
  1000+ goblin-top / phase hops: 0 elevated lands.
- LL **does spawn** (fpd6): types **0x3D/0x3E** mapset4 from prog~961;
  body y≈32–36. Goblin **0x40** (was mislabeled type36).
- rr-54ui PARTIAL: rider kill Clean; empty cloud solid never arms (body never
  LDA #$90; only appear-block AI does). Zero-mask force = global solid OK.
  No Air-first Clean alt (Item-1 needs Heat; gap ~296px).
- rr-f3nr PARTIAL: Heat→Air Item-1 scaffold — Heat1 entry + HeatScreen1 dual
  green; Heat clear / Item-1 / Air-with-Item-1 residual.

NES buttons: B=shoot, A=jump (fceumm 9-button layout).
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action


def _run_buttons(*, jump: bool, shoot: bool) -> tuple[list[str], str]:
    buttons = ["RIGHT"]
    if jump:
        buttons.append("A")
    if shoot:
        buttons.append("B")
    if jump and shoot:
        reason = "run_jump_shoot"
    elif jump:
        reason = "run_jump"
    elif shoot:
        reason = "run_shoot"
    else:
        reason = "run"
    return buttons, reason


@dataclass
class AirScreen1Policy:
    """Periodic jump-run to clear Air Man screen 0 → screen 1 (legacy)."""

    jump_period: int = 50
    jump_hold: int = 12
    shoot_period: int = 40
    shoot_hold: int = 2
    target_camera_screen: int = 1

    def tick(
        self,
        *,
        frame: int,
        health: int,
        camera_x_screen: int,
        fallen: bool = False,
    ) -> FrameAction:
        """Choose one frame of controller input."""
        if health <= 0 or fallen:
            return FrameAction(nes_idle_action(), "dead")
        if camera_x_screen >= self.target_camera_screen:
            return FrameAction(nes_idle_action(), "clear_hold")

        i = max(0, frame - 1)
        jump = self.jump_period > 0 and (i % self.jump_period) < self.jump_hold
        shoot = self.shoot_period > 0 and (i % self.shoot_period) < self.shoot_hold
        buttons, reason = _run_buttons(jump=jump, shoot=shoot)
        return FrameAction(nes_action(*buttons), reason)


@dataclass
class HeatManPolicy:
    """Multi-phase Heat Man route (Clean Bronze mid-stage).

    Start modes (0-based frame index ``i``):

    - ``early`` (Heat1 / HeatScreen1): period 50/12 → screen ≥1 / ≥2.
    - ``screen2`` (HeatScreen2): period 60/14 until ``mid_until``, then 25/12
      → grounded screen ≥3 (~351f).
    - ``screen3`` (HeatScreen3 pillars): period 25/10 with ``jump_phase`` 10
      → grounded screen ≥4 (~181f).
    - ``screen4`` (HeatScreen4): period 20/12 with ``jump_phase`` 4 → screen ≥5
      (~131f cam / ~320f grounded).
    - ``screen5`` (HeatScreen5Ground): A-edge idle → j1 → LEFT → j2 → rising-edge
      short hops (hold 9 / ground_wait 3) → camera ≥7 pre-boss (~305f; 2026-08-10).
    - ``screen7`` (HeatScreen7Mid): high-path — LEFT back to cam6, climb sy~68,
      cross ABOVE sx152 wall into cam7, micro-hop to mapset7 ladder, DOWN
      scroll_down → camera ≥8 Sniper shaft (~587f; 3/3; 2026-08-10).
    - ``screen8`` (HeatScreen8): Yoku room → cam ≥ 9. Wait 187 (upper
      ceiling off), land first Yoku, jump up to catch appearing upper B,
      jump LEFT to D, hop to left-wall ledge, walk onto ladder, DOWN
      scroll_down (~680f). Upper B is a ceiling while first is solid on
      the opening phase — do not jump-from-below then.

    Residual: Heat cam ≥ 9 section E columns / F lava Yoku / G Sniper →
    boss door + Item-1 (rr-k1ea / rr-809 PARTIAL). Low alcove sx152 is a
    dead-end. Do not jump straight up into upper Yoku while it is already
    solid (bonks underside).
    """

    # early (Heat1 / HeatScreen1)
    jump_period: int = 50
    jump_hold: int = 12
    # screen2 mid → late handoff
    mid_until: int = 260
    mid_period: int = 60
    mid_hold: int = 14
    late_period: int = 25
    late_hold: int = 12
    # screen3 pillars
    s3_period: int = 25
    s3_hold: int = 10
    s3_phase: int = 10
    # screen4 late
    s4_period: int = 20
    s4_hold: int = 12
    s4_phase: int = 4
    # screen5 pre-boss (stateful; needs tile_feet)
    s5_idle: int = 2
    s5_j1_hold: int = 20
    s5_left: int = 4
    s5_j2_hold: int = 24
    s5_hop_hold: int = 9
    s5_ground_wait: int = 3
    shoot_period: int = 40
    shoot_hold: int = 2
    target_camera_screen: int = 1
    # early | screen2 | screen3 | screen4 | screen5 | screen7 | screen8
    start: str = "early"
    # screen5 phase machine (mutated across ticks)
    _s5_phase: str = "idle"
    _s5_pt: int = 0
    _s5_a_down: bool = False
    _s5_a_left: int = 0
    _s5_gnd_run: int = 0

    def tick(
        self,
        *,
        frame: int,
        health: int,
        camera_x_screen: int,
        fallen: bool = False,
        tile_feet: int = 0,
    ) -> FrameAction:
        """Choose one frame of controller input."""
        if health <= 0 or fallen:
            return FrameAction(nes_idle_action(), "dead")
        if camera_x_screen >= self.target_camera_screen:
            return FrameAction(nes_idle_action(), "clear_hold")

        if self.start == "screen5":
            return self._tick_screen5(frame=frame, tile_feet=tile_feet)
        if self.start == "screen7":
            return self._tick_screen7(frame=frame, tile_feet=tile_feet)
        if self.start == "screen8":
            return self._tick_screen8(frame=frame, tile_feet=tile_feet)

        i = max(0, frame - 1)
        jump = self._want_jump(i)
        shoot = self.shoot_period > 0 and (i % self.shoot_period) < self.shoot_hold
        buttons, reason = _run_buttons(jump=jump, shoot=shoot)
        if jump and self.start == "screen2" and i >= self.mid_until:
            reason = "late_jump" if not shoot else "late_jump_shoot"
        elif jump and self.start == "screen2":
            reason = "mid_jump" if not shoot else "mid_jump_shoot"
        elif jump and self.start in {"screen3", "screen4"}:
            reason = "pillar_jump" if not shoot else "pillar_jump_shoot"
        return FrameAction(nes_action(*buttons), reason)

    def _tick_screen5(self, *, frame: int, tile_feet: int) -> FrameAction:
        """Stateful late Heat: gap hops into pre-boss (cam ≥7)."""
        i = max(0, frame - 1)
        shoot = self.shoot_period > 0 and (i % self.shoot_period) < self.shoot_hold
        feet_gnd = tile_feet == 1
        feet_ladder = tile_feet == 2
        if feet_gnd:
            self._s5_gnd_run += 1
        else:
            self._s5_gnd_run = 0

        jump = False
        go_left = False
        go_up = False
        reason = "run"

        if self._s5_phase == "idle":
            self._s5_pt += 1
            if self._s5_pt >= self.s5_idle:
                self._s5_phase = "j1"
                self._s5_pt = 0
            return FrameAction(nes_idle_action(), "s5_idle")
        if self._s5_phase == "j1":
            jump = self._s5_pt < self.s5_j1_hold
            self._s5_pt += 1
            if self._s5_pt >= self.s5_j1_hold:
                self._s5_phase = "air1"
                self._s5_pt = 0
            reason = "s5_j1"
        elif self._s5_phase == "air1":
            if feet_gnd:
                self._s5_phase = "left"
                self._s5_pt = 0
            reason = "s5_air1"
        elif self._s5_phase == "left":
            go_left = True
            self._s5_pt += 1
            if self._s5_pt >= self.s5_left:
                self._s5_phase = "j2"
                self._s5_pt = 0
            reason = "s5_left"
        elif self._s5_phase == "j2":
            jump = self._s5_pt < self.s5_j2_hold
            self._s5_pt += 1
            if self._s5_pt >= self.s5_j2_hold:
                self._s5_phase = "air2"
                self._s5_pt = 0
            reason = "s5_j2"
        elif self._s5_phase == "air2":
            if feet_gnd:
                self._s5_phase = "hop"
                self._s5_pt = 0
                self._s5_a_down = False
                self._s5_a_left = 0
            reason = "s5_air2"
        elif self._s5_phase == "hop":
            if feet_ladder:
                go_up = True
                self._s5_a_down = False
                reason = "s5_ladder"
            elif self._s5_a_left > 0:
                jump = True
                self._s5_a_left -= 1
                self._s5_a_down = True
                reason = "s5_hop"
            elif feet_gnd and self._s5_gnd_run >= self.s5_ground_wait:
                if self._s5_a_down:
                    # release one frame for rising edge
                    self._s5_a_down = False
                    jump = False
                    reason = "s5_release"
                else:
                    jump = True
                    self._s5_a_down = True
                    self._s5_a_left = self.s5_hop_hold - 1
                    reason = "s5_hop"
            else:
                self._s5_a_down = False
                reason = "s5_run"
        else:
            reason = "s5_run"

        if go_up:
            buttons = ["UP"]
            if shoot:
                buttons.append("B")
                reason = "s5_ladder_shoot"
            return FrameAction(nes_action(*buttons), reason)
        if go_left:
            buttons = ["LEFT"]
            if shoot:
                buttons.append("B")
            return FrameAction(nes_action(*buttons), reason)

        buttons, base_reason = _run_buttons(jump=jump, shoot=shoot)
        if jump:
            reason = reason if reason.startswith("s5_") else base_reason
        elif shoot and reason == "run":
            reason = "run_shoot"
        return FrameAction(nes_action(*buttons), reason)

    def _tick_screen7(self, *, frame: int, tile_feet: int) -> FrameAction:
        """Frame script: high-path past s7 wall → ladder → scroll_down cam≥8.

        0-based index ``i`` windows (verified 3/3 from HeatScreen7Mid → cam8):

        - 0–11: LEFT (drop off low alcove; may scroll cam6)
        - 3× climb: A+LEFT 24 / LEFT 6 / idle 3
        - A+RIGHT 30, RIGHT 40, A+RIGHT 16, RIGHT 60 (high cross past sx152)
        - 4× micro: A+RIGHT 6 / RIGHT 25 / idle 15 → mapset7 ladder
        - then DOWN (ladder / scroll_down); UP if already on ladder early
        """
        i = max(0, frame - 1)
        if tile_feet == 2:
            return FrameAction(nes_action("DOWN"), "s7_ladder_down")

        # Build cumulative windows matching dual-green PLAN
        segs: list[tuple[tuple[str, ...], int, str]] = [
            (("LEFT",), 12, "s7_left_off"),
        ]
        for _ in range(3):
            segs += [
                (("A", "LEFT"), 24, "s7_climb"),
                (("LEFT",), 6, "s7_climb_walk"),
                ((), 3, "s7_climb_idle"),
            ]
        segs += [
            (("A", "RIGHT"), 30, "s7_cross_ar0"),
            (("RIGHT",), 40, "s7_cross_r1"),
            (("A", "RIGHT"), 16, "s7_cross_ar1"),
            (("RIGHT",), 60, "s7_cross_r2"),
        ]
        for _ in range(4):
            segs += [
                (("A", "RIGHT"), 6, "s7_micro_ar"),
                (("RIGHT",), 25, "s7_micro_r"),
                ((), 15, "s7_micro_idle"),
            ]
        # remainder: climb down
        segs.append((("DOWN",), 500, "s7_down"))

        t = 0
        for btns, n, reason in segs:
            if i < t + n:
                if not btns:
                    return FrameAction(nes_idle_action(), reason)
                return FrameAction(nes_action(*btns), reason)
            t += n
        return FrameAction(nes_action("DOWN"), "s7_down")

    def _tick_screen8(self, *, frame: int, tile_feet: int) -> FrameAction:
        """Frame script: Yoku room (no-ceiling catch) → left ladder → cam ≥ 9.

        0-based ``i`` (HeatScreen8 → camera 9 via scroll_down):

        - 0–186: idle (wait until first+D on, upper B off)
        - land first Yoku (LEFT 8, idle 1, A+LEFT 14, LEFT 18)
        - idle 4, A 20: jump up; appearing B catches at sy~52
        - idle 1, A+LEFT 16, LEFT 32: jump to D (104,55) then hop
        - idle 1, A+LEFT 8, LEFT 50: left-wall ledge → ladder
        - DOWN: scroll_down into section E (cam ≥ 9)
        """
        i = max(0, frame - 1)
        segs: list[tuple[tuple[str, ...], int, str]] = [
            ((), 187, "s8_wait"),
            (("LEFT",), 8, "s8_approach"),
            ((), 1, "s8_release"),
            (("A", "LEFT"), 14, "s8_yoku_jump"),
            (("LEFT",), 18, "s8_yoku_coast"),
            ((), 4, "s8_catch_gap"),
            (("A",), 20, "s8_catch"),
            ((), 3, "s8_catch_hang"),
            ((), 1, "s8_to_d_rel"),
            (("A", "LEFT"), 16, "s8_to_d"),
            (("LEFT",), 32, "s8_to_d_coast"),
            ((), 1, "s8_ledge_rel"),
            (("A", "LEFT"), 8, "s8_ledge_hop"),
            (("LEFT",), 50, "s8_ledge_walk"),
            (("DOWN",), 400, "s8_down"),
        ]
        t = 0
        for btns, n, reason in segs:
            if i < t + n:
                if not btns:
                    return FrameAction(nes_idle_action(), reason)
                return FrameAction(nes_action(*btns), reason)
            t += n
        return FrameAction(nes_action("DOWN"), "s8_down")

    def _want_jump(self, i: int) -> bool:
        if self.start == "screen2":
            if i < self.mid_until:
                return (i % self.mid_period) < self.mid_hold
            return (i % self.late_period) < self.late_hold
        if self.start == "screen3":
            return ((i + self.s3_phase) % self.s3_period) < self.s3_hold
        if self.start == "screen4":
            return ((i + self.s4_phase) % self.s4_period) < self.s4_hold
        return self.jump_period > 0 and (i % self.jump_period) < self.jump_hold

    @staticmethod
    def start_for_state(state_name: str) -> str:
        """Map checkpoint name → recipe start mode."""
        if state_name.startswith("HeatScreen8") or state_name.startswith(
            "HeatS8"
        ):
            return "screen8"
        if state_name.startswith("HeatScreen7") or state_name.startswith(
            "HeatLadder"
        ) or state_name.startswith("HeatS7"):
            return "screen7"
        if state_name.startswith("HeatScreen5") or state_name.startswith(
            "HeatScreen6"
        ):
            return "screen5"
        if state_name.startswith("HeatScreen4"):
            return "screen4"
        if state_name.startswith("HeatScreen3"):
            return "screen3"
        if state_name.startswith("HeatScreen2"):
            return "screen2"
        return "early"


@dataclass
class AirManPolicy:
    """Multi-phase Air Man route: Level1 / AirLanded / AirScreen2 starts."""

    # Early screen-0 hop
    early_until: int = 180
    early_period: int = 50
    early_hold: int = 12
    # Land onto screen-1 platforms
    land_jump_start: int = 216
    land_jump_hold: int = 14
    # Absolute 0-based frame of grounded AirLanded (Level1 path)
    land_frame: int = 301
    # From AirLanded / post-land: approach + gap into screen 2
    mid_period: int = 50
    mid_hold: int = 12
    gap_rel: int = 142
    gap_hold: int = 14
    # AirScreen2 late-stage (fans / gaps → screens 3–4)
    s2_approach_start: int = 48
    s2_approach_period: int = 45
    s2_approach_hold: int = 16
    s2_fan_start: int = 145
    s2_fan_end: int = 180
    s2_late_period: int = 40
    s2_late_hold: int = 16
    shoot_period: int = 40
    shoot_hold: int = 2
    target_camera_screen: int = 2
    # "level1" full recipe; "landed" post-land only; "screen2" late-stage
    start: str = "level1"

    def tick(
        self,
        *,
        frame: int,
        health: int,
        camera_x_screen: int,
        fallen: bool = False,
    ) -> FrameAction:
        """Choose one frame of controller input."""
        if health <= 0 or fallen:
            return FrameAction(nes_idle_action(), "dead")
        if camera_x_screen >= self.target_camera_screen:
            return FrameAction(nes_idle_action(), "clear_hold")

        i = max(0, frame - 1)
        shoot = self.shoot_period > 0 and (i % self.shoot_period) < self.shoot_hold
        jump = self._want_jump(i)
        buttons, reason = _run_buttons(jump=jump, shoot=shoot)
        if jump and self.start == "level1" and self.land_jump_start <= i < (
            self.land_jump_start + self.land_jump_hold
        ):
            reason = "land_jump" if not shoot else "land_jump_shoot"
        elif jump and self.start == "screen2" and self._in_s2_fan(i):
            reason = "fan_hold" if not shoot else "fan_hold_shoot"
        elif jump and self.start == "screen2" and self._in_s2_late(i):
            reason = "late_jump" if not shoot else "late_jump_shoot"
        elif jump and self._in_gap(i):
            reason = "gap_jump" if not shoot else "gap_jump_shoot"
        return FrameAction(nes_action(*buttons), reason)

    def _in_gap(self, i: int) -> bool:
        r = self._rel_mid(i)
        if r is None:
            return False
        return self.gap_rel <= r < self.gap_rel + self.gap_hold

    def _rel_mid(self, i: int) -> int | None:
        if self.start == "landed":
            return i
        if self.start == "level1" and i >= self.land_frame:
            return i - self.land_frame
        return None

    def _in_s2_fan(self, i: int) -> bool:
        return self.s2_fan_start <= i < self.s2_fan_end

    def _in_s2_late(self, i: int) -> bool:
        return i >= self.s2_fan_end

    def _want_jump(self, i: int) -> bool:
        if self.start == "screen2":
            return self._screen2_jump(i)
        if self.start == "landed":
            return self._mid_jump(i)
        if i < self.early_until:
            return (i % self.early_period) < self.early_hold
        if self.land_jump_start <= i < self.land_jump_start + self.land_jump_hold:
            return True
        r = self._rel_mid(i)
        if r is not None:
            return self._mid_jump(r)
        return False

    def _screen2_jump(self, i: int) -> bool:
        """Late-stage from AirScreen2: approach → fan hold → period 40/16."""
        if i < self.s2_approach_start:
            return False
        if i < self.s2_fan_start:
            r = i - self.s2_approach_start
            return (r % self.s2_approach_period) < self.s2_approach_hold
        if i < self.s2_fan_end:
            return True
        r = i - self.s2_fan_end
        return (r % self.s2_late_period) < self.s2_late_hold

    def _mid_jump(self, r: int) -> bool:
        if r < self.gap_rel:
            return (r % self.mid_period) < self.mid_hold
        return self.gap_rel <= r < self.gap_rel + self.gap_hold
