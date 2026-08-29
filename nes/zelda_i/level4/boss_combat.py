"""Level 4 Gleeok fight + HC + TF bit 0x08 (rr-rvae / rr-vdnc / rr-gjey).

Live facts (from ``Level4GleeokEnter``, room **0x13**)::

    - Body type ``0x43`` starts HP≈160; TYPE-only residual after head split.
    - Detached head type ``0x46`` appears mid-fight (do **not** chase — south
      stand on body clears residual faster and safer than head kite).
    - Fireball residual ``0x56`` (dodge when close; **post-boss residual
      kills** if you stand still — rr-gjey).
    - Clean policy: south stand ``(body.x, body.y+STAND_DY)`` face UP + A
      (rr-vdnc). Bombs do not damage Gleeok.
    - Boss dead when body type ``0x43`` absent; HC RoomItemId ``0x1A`` mid-room.
    - UP @x≈120 → TF room **0x03**; walk mid → ``ADDR_TRIFORCE & 0x08``.
    - Health floor (lab poke from GleeokEnter): stock south-stand dual-green
      at health≥107; 106 kills boss then residual fireball death unless
      post-boss phase dodges; ≤105 dies mid-fight. Continuous Entrance→map
      peels to ~98–100 → needs heart-safe path **and/or** low-HP fight care
      (rr-gjey).

Assisted first-pass dual-green 2/2 (~3649f). Clean dual from GleeokEnter
(rr-vdnc south-stand). Not full-game Clean STATUS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import TF_BIT_L4
from zelda_i.dungeon.gleeok import (
    FIREBALL_DODGE_DIST,
    GLEEOK_FIREBALL_TYPE,
    GLEEOK_HEAD_OBJECT_TYPE,
    GLEEOK_OBJECT_TYPE,
    STAND_DY,
    _fireball_dodge_dir,
    _south_stand_action,
    gleeok_fireballs,
    gleeok_heads_live,
    gleeok_live,
)
from zelda_i.dungeon.ops import (
    PUSH_FRAMES,
    exit_door,
    goto,
    idle,
    room_fields,
)
from zelda_i.level4.dungeon import (
    LEVEL4,
    ROOM_L4_GLEEOK_13,
)
from zelda_i.ram import (
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

ROOM_L4_TRIFORCE = 0x03  # north of boss 0x13 after clear

# Approach-only wider dodge when start health is depleted (rr-gjey). Do **not**
# widen mid-fight dodge thr — that walks into body and breaks ≥107 Clean.
FIREBALL_DODGE_DIST_LOW_HP = 22
# Lab poke cliff (7 containers, full=0x6F=111): stock mid-fight dual-green at
# start health ≥107. Below that, approach is more careful; post-boss always
# dodges residual fireballs (rr-gjey).
LOW_HP_THRESHOLD = 107
FIGHT_MAX_FRAMES = 20000
APPROACH_SOUTH_Y = 165
# Post-boss HC hunt waypoints (mid + north band; residual fireball danger).
HC_STANDS: tuple[tuple[int, int], ...] = (
    (120, 125),
    (124, 111),
    (112, 125),
    (128, 125),
    (120, 117),
    (120, 141),
    (104, 133),
    (136, 133),
    (120, 109),
    (96, 141),
    (144, 141),
    (120, 101),
    (120, 157),
    (80, 125),
    (160, 125),
    (88, 133),
    (152, 133),
    (120, 93),
)
TF_STANDS: tuple[tuple[int, int], ...] = (
    (120, 141),
    (120, 149),
    (112, 141),
    (128, 141),
    (120, 125),
    (120, 157),
    (120, 109),
    (104, 141),
    (136, 141),
    (120, 93),
    (120, 173),
    (96, 141),
    (144, 141),
)
UP_APPROACHES: tuple[tuple[int, int], ...] = (
    (120, 93),
    (112, 93),
    (128, 93),
    (120, 101),
    (100, 93),
    (140, 93),
    (120, 109),
)


def _nearest_fireball_dist(snap: ZeldaSnapshot) -> int | None:
    balls = gleeok_fireballs(snap)
    if not balls:
        return None
    return min(
        abs(o.x - snap.link_x) + abs(o.y - snap.link_y) for o in balls
    )


def level4_tf08(ram: Any) -> bool:
    return bool(int(read_u8(ram, ADDR_TRIFORCE)) & TF_BIT_L4)


def level4_gleeok_cleared(snap: ZeldaSnapshot) -> bool:
    """Boss body absent on 0x13 (heads/fireballs may linger one frame)."""
    return (
        snap.level == LEVEL4
        and snap.screen == ROOM_L4_GLEEOK_13
        and not gleeok_live(snap)
    )


def level4_complete_success(ram: Any) -> bool:
    """Inventory fact: TF bit 0x08 set (mode 18 fanfare ok)."""
    return level4_tf08(ram)


@dataclass
class Level4GleeokFightController:
    """Melee Gleeok → HC → UP 0x03 → TF 0x08 from Level4GleeokEnter.

    Assisted dual-green 2/2 from ``Level4GleeokEnter`` (rr-rvae).
    Clean dual via south-stand policy (rr-vdnc): hold south of body face UP+A;
    do not chase detached heads while body residual remains.
    """

    tag: str = "l4_gleeok"
    max_frames: int = FIGHT_MAX_FRAMES
    stand_dy: int = STAND_DY
    fireball_dodge_dist: int = FIREBALL_DODGE_DIST
    success: bool = False
    boss_beaten: bool = False
    hc_collected: bool = False
    tf08: bool = False
    frames: int = 0
    dmg_events: int = 0
    notes: list[str] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)
    fight_report: dict[str, Any] = field(default_factory=dict)
    _approached: bool = field(default=False, repr=False)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "boss_beaten": self.boss_beaten,
            "hc_collected": self.hc_collected,
            "tf08": self.tf08,
            "frames": self.frames,
            "dmg_events": self.dmg_events,
            "notes": list(self.notes),
            "log_tail": self.log[-20:],
            "fight": self.fight_report,
            "segment": "level4_gleeok_fight_tf",
            "policy": "south_stand",
            "stand_dy": self.stand_dy,
            "target_room": f"0x{ROOM_L4_GLEEOK_13:02x}",
            "tf_room": f"0x{ROOM_L4_TRIFORCE:02x}",
            "tf_bit": f"0x{TF_BIT_L4:02x}",
        }

    def run(
        self,
        env: Any,
        assist: Any | None,
        total: list[int],
    ) -> dict[str, Any]:
        """Full fight from current state (expect play-ready 0x13)."""
        snap0 = read_snapshot(env.get_ram())
        if not (
            snap0.level == LEVEL4
            and snap0.screen == ROOM_L4_GLEEOK_13
            and snap0.mode in (PLAY_MODE, 5)
        ):
            result = {
                "ok": False,
                "error": (
                    f"expected L4 0x13 play; got L{snap0.level} "
                    f"sc=0x{snap0.screen:02x} mode={snap0.mode}"
                ),
            }
            self.fight_report = result
            return result

        hc0 = snap0.heart_containers
        start_health = int(snap0.health)
        last_body_hp: int | None = None
        last_filled: int | None = snap0.filled_hearts
        invuln = 0
        phase = "fight"
        self._approached = False
        hc_hunt_i = 0
        # Stock approach+mid-fight (rr-vdnc). Continuous path needs enter
        # health ≥~108 (approach costs more than GleeokEnter lab). Post-boss
        # residual fireball care is the rr-gjey harden (see phase hc/tf_exit).
        self.notes.append(
            f"policy=south_stand dy={self.stand_dy} "
            f"fb_dodge<={self.fireball_dodge_dist} start_hp={start_health}"
        )

        for frame in range(self.max_frames):
            snap = read_snapshot(env.get_ram())
            ram = env.get_ram()
            self.frames = total[0]
            filled = snap.filled_hearts
            if last_filled is not None and filled < last_filled:
                invuln = 48
            last_filled = filled
            if invuln > 0:
                invuln -= 1

            if level4_tf08(ram):
                self.tf08 = True
                self.boss_beaten = True
                self.success = True
                self.notes.append(f"tf08 frame={frame}")
                self.log.append(
                    {"event": "tf08", "frame": frame, **room_fields(snap, ram)}
                )
                # Settle through mode-18 fanfare a bit.
                idle(env, assist, total, 60)
                final = room_fields(read_snapshot(env.get_ram()), env.get_ram())
                result = {
                    "ok": True,
                    "tf08": True,
                    "boss_beaten": True,
                    "hc_collected": self.hc_collected
                    or final.get("heart_containers", 0) > hc0,
                    "frames": frame,
                    "dmg_events": self.dmg_events,
                    "notes": list(self.notes),
                    "log": self.log[-40:],
                    "final": final,
                    "policy": "south_stand",
                    "stand_dy": self.stand_dy,
                }
                if final.get("heart_containers", 0) > hc0:
                    self.hc_collected = True
                    result["hc_collected"] = True
                self.fight_report = result
                self.frames = total[0]
                return result

            if snap.mode == 17:
                result = {
                    "ok": False,
                    "error": "death",
                    "frames": frame,
                    "notes": list(self.notes),
                    "log": self.log[-20:],
                    "policy": "south_stand",
                }
                self.fight_report = result
                return result

            if phase == "fight":
                if snap.mode not in (PLAY_MODE, 5):
                    env.step(nes_idle_action())
                    total[0] += 1
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
                    continue

                bodies = gleeok_live(snap)
                heads = gleeok_heads_live(snap)
                body_hp = sum(max(0, int(o.hp)) for o in bodies)
                if last_body_hp is not None and body_hp < last_body_hp:
                    self.dmg_events += 1
                    self.log.append(
                        {
                            "event": "hp_drop",
                            "frame": frame,
                            "body_hp": body_hp,
                            "prev": last_body_hp,
                            "heads": len(heads),
                        }
                    )
                last_body_hp = body_hp

                if not bodies and frame > 30:
                    self.boss_beaten = True
                    self.notes.append(
                        f"boss_dead f={frame} rad={snap.room_all_dead} "
                        f"doors={snap.cur_opened_doors} heads={len(heads)} "
                        f"hp={snap.health}"
                    )
                    self.log.append(
                        {
                            "event": "boss_dead",
                            "frame": frame,
                            **room_fields(snap, ram),
                        }
                    )
                    # rr-gjey: residual fireball 0x56 kills if we idle/goto
                    # unprotected — go straight into active HC/exit care.
                    phase = "hc"
                    hc_hunt_i = 0
                    continue

                dodge_thr = self.fireball_dodge_dist

                # Entry: drop south first (avoid left-band body contact), then
                # align under body x before engaging stand. Dodge fireballs
                # during approach (rr-zavx Clean death was approach-phase).
                if not self._approached:
                    if invuln <= 0:
                        dodge_a = _fireball_dodge_dir(snap, thr=dodge_thr)
                        if dodge_a is not None:
                            env.step(nes_action(dodge_a))
                            total[0] += 1
                            if assist is not None:
                                assist.apply_env(env, frame=total[0])
                            continue
                    if snap.link_y < APPROACH_SOUTH_Y:
                        env.step(nes_action("DOWN"))
                        total[0] += 1
                        if assist is not None:
                            assist.apply_env(env, frame=total[0])
                        continue
                    bx = bodies[0].x if bodies else 124
                    if abs(snap.link_x - bx) > 8:
                        env.step(
                            nes_action(
                                "RIGHT" if snap.link_x < bx else "LEFT"
                            )
                        )
                        total[0] += 1
                        if assist is not None:
                            assist.apply_env(env, frame=total[0])
                        continue
                    self._approached = True
                    self.notes.append(
                        f"approach_south f={frame} xy=({snap.link_x},{snap.link_y}) "
                        f"hp={snap.health} dodge_thr={dodge_thr}"
                    )

                # Tight fireball dodge (horizontal) when not invulnerable.
                dodge = (
                    None
                    if invuln > 0
                    else _fireball_dodge_dir(snap, thr=dodge_thr)
                )
                if dodge is not None:
                    env.step(nes_action(dodge))
                    total[0] += 1
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
                    continue

                if bodies:
                    # South stand on body for full fight — do not chase heads
                    # while residual body remains (rr-vdnc Clean).
                    act = _south_stand_action(
                        snap, bodies[0], stand_dy=self.stand_dy
                    )
                    env.step(act)
                elif heads:
                    # Body gone, heads linger: brief face-and-slash.
                    nearest = min(
                        heads,
                        key=lambda o: abs(o.x - snap.link_x)
                        + abs(o.y - snap.link_y),
                    )
                    dx = nearest.x - snap.link_x
                    dy = nearest.y - snap.link_y
                    if abs(dx) >= abs(dy):
                        face = "RIGHT" if dx > 0 else "LEFT"
                    else:
                        face = "DOWN" if dy > 0 else "UP"
                    env.step(nes_action(face, "A"))
                else:
                    env.step(
                        nes_action(
                            ("UP", "RIGHT", "DOWN", "LEFT")[frame // 18 % 4],
                            "A",
                        )
                    )
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
                continue

            if phase == "hc":
                # rr-gjey: residual 0x56 fireball approaches from south and
                # kills filled=0 Link if we walk north into it. 2D flee while
                # any ball is near; only hunt HC / exit once clear.
                if snap.mode not in (PLAY_MODE, 5, 8):
                    env.step(nes_idle_action())
                    total[0] += 1
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
                    continue
                if snap.heart_containers > hc0:
                    self.hc_collected = True
                    self.notes.append(
                        f"HC f={frame} {hc0}->{snap.heart_containers} "
                        f"hp={snap.health}"
                    )
                    self.log.append(
                        {
                            "event": "hc",
                            "frame": frame,
                            "hc": snap.heart_containers,
                            "health": snap.health,
                        }
                    )
                    phase = "tf_exit"
                    continue
                fb_dist = _nearest_fireball_dist(snap)
                # While ANY residual fireball exists, only lateral flee / hold —
                # do not walk north into its path (rr-gjey).
                if fb_dist is not None and invuln <= 0:
                    dodge0 = _fireball_dodge_dir(
                        snap, thr=200, allow_vertical=True
                    )
                    if dodge0 is not None:
                        env.step(nes_action(dodge0))
                    else:
                        env.step(nes_idle_action())
                    total[0] += 1
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
                    hc_hunt_i += 1
                    # Fireballs sometimes linger; after long evade, try exit.
                    if hc_hunt_i > 400:
                        self.notes.append(
                            "HC skip after fireball evade (UP path)"
                        )
                        phase = "tf_exit"
                    continue
                # Fireballs clear: short door-open wait, then waypoint hunt.
                if hc_hunt_i < 20 and (snap.cur_opened_doors & 0x08) == 0:
                    env.step(nes_idle_action())
                    total[0] += 1
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
                    hc_hunt_i += 1
                    continue
                tx, ty = HC_STANDS[hc_hunt_i // 28 % len(HC_STANDS)]
                if abs(snap.link_x - tx) > 4 or abs(snap.link_y - ty) > 4:
                    if abs(snap.link_y - ty) >= abs(snap.link_x - tx):
                        d = "DOWN" if snap.link_y < ty else "UP"
                    else:
                        d = "RIGHT" if snap.link_x < tx else "LEFT"
                    env.step(nes_action(d))
                else:
                    env.step(nes_idle_action())
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
                hc_hunt_i += 1
                if hc_hunt_i > 900:
                    self.notes.append(
                        "HC not mid-room yet (may collect on UP path)"
                    )
                    phase = "tf_exit"
                continue

            if phase == "tf_exit":
                # Do not idle long — residual fireball (rr-gjey).
                dodge_e = _fireball_dodge_dir(
                    snap, thr=48, allow_vertical=True
                )
                if dodge_e is not None and invuln <= 0:
                    env.step(nes_action(dodge_e))
                    total[0] += 1
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
                    continue
                # Brief settle only when fireballs are clear.
                idle(env, assist, total, 4)
                pr = exit_door(
                    env,
                    assist,
                    total,
                    "UP",
                    x_force=120,
                    y_force=93,
                    push=PUSH_FRAMES + 100,
                )
                self.notes.append(
                    f"exit_up result={pr.get('result')} "
                    f"to={(pr.get('after') or {}).get('sc')}"
                )
                self.log.append(
                    {
                        "event": "exit_up",
                        "result": pr.get("result"),
                        "after": pr.get("after"),
                    }
                )
                after = pr.get("after") or {}
                if after.get("screen") == ROOM_L4_TRIFORCE or (
                    pr.get("changed_room")
                    and after.get("screen") == ROOM_L4_TRIFORCE
                ):
                    self.notes.append("entered TF room 0x03")
                    phase = "tf_collect"
                    continue

                st = env.em.get_state()
                ok_exit = False
                for ax, ay in UP_APPROACHES:
                    env.em.set_state(st)
                    pr2 = exit_door(
                        env,
                        assist,
                        total,
                        "UP",
                        x_force=ax,
                        y_force=ay,
                        push=PUSH_FRAMES + 120,
                    )
                    a2 = pr2.get("after") or {}
                    if (
                        pr2.get("changed_room")
                        and a2.get("screen") == ROOM_L4_TRIFORCE
                    ):
                        self.notes.append(f"exit_up@({ax},{ay})")
                        ok_exit = True
                        phase = "tf_collect"
                        break
                if not ok_exit:
                    final = room_fields(
                        read_snapshot(env.get_ram()), env.get_ram()
                    )
                    result = {
                        "ok": False,
                        "error": "no_tf_exit",
                        "frames": frame,
                        "notes": list(self.notes),
                        "final": final,
                        "log": self.log[-30:],
                    }
                    self.fight_report = result
                    return result
                continue

            if phase == "tf_collect":
                for tx, ty in TF_STANDS:
                    goto(env, assist, total, tx, ty, tol=3, max_f=300)
                    if level4_tf08(env.get_ram()):
                        self.notes.append(f"TF at ({tx},{ty})")
                        break
                else:
                    for ty in range(93, 189, 8):
                        for tx in range(80, 177, 8):
                            goto(env, assist, total, tx, ty, tol=2, max_f=100)
                            if level4_tf08(env.get_ram()):
                                self.notes.append(f"TF dense ({tx},{ty})")
                                break
                        else:
                            continue
                        break
                    else:
                        final = room_fields(
                            read_snapshot(env.get_ram()), env.get_ram()
                        )
                        result = {
                            "ok": False,
                            "error": "tf_miss",
                            "frames": frame,
                            "notes": list(self.notes),
                            "final": final,
                            "log": self.log[-30:],
                        }
                        self.fight_report = result
                        return result
                # Next loop iteration catches tf bit.
                continue

        result = {
            "ok": False,
            "error": "timeout",
            "frames": self.max_frames,
            "notes": list(self.notes),
            "log": self.log[-30:],
            "final": room_fields(read_snapshot(env.get_ram()), env.get_ram()),
        }
        self.fight_report = result
        return result


def make_gleeok_fight_controller(
    *, tag: str = "l4_gleeok"
) -> Level4GleeokFightController:
    return Level4GleeokFightController(tag=tag)


__all__ = [
    "FIGHT_MAX_FRAMES",
    "FIREBALL_DODGE_DIST",
    "GLEEOK_FIREBALL_TYPE",
    "GLEEOK_HEAD_OBJECT_TYPE",
    "HC_STANDS",
    "Level4GleeokFightController",
    "ROOM_L4_TRIFORCE",
    "STAND_DY",
    "TF_STANDS",
    "gleeok_fireballs",
    "gleeok_heads_live",
    "gleeok_live",
    "level4_complete_success",
    "level4_gleeok_cleared",
    "level4_tf08",
    "make_gleeok_fight_controller",
]
