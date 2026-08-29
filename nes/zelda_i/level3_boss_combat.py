"""Level 3 boss-path combat: prep clear, 0x5d UP gate, Manhandla fight.

Used by ``Level3BossPathController`` (mixin). Public helpers re-exported from
``level3_boss_path``.
"""

from __future__ import annotations

from typing import Any

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import save_rgb_png
from zelda_i.door_graph.core import DoorDir
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon_ops import (
    GEL_ALT_OBJECT_TYPE,
    GEL_SPLIT_OBJECT_TYPE,
    PUSH_FRAMES,
    bomb_stand,
    ensure_bomb,
    exit_door,
    goto,
    idle,
    live_killables,
    poke_bombs,
    push_dir,
    room_fields,
)
from zelda_i.level3_dungeon import (
    KEESE_OBJECT_TYPE,
    LEVEL3_TRIFORCE_BIT,
    MANHANDLA_OBJECT_TYPE,
    PASSAGE_EXIT_WAYPOINTS,
    ROOM_L3_BOSS,
    ROOM_L3_BOSS_PREP,
    ROOM_L3_RAFT_PASSAGE,
    ROOM_L3_SOUTH_DARKNUTS,
    ZOL_OBJECT_TYPE,
    level3_manhandla_live,
)
from zelda_i.level3_overworld import LEVEL3
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

# 0x5d killables only — 0x2b invuln must NOT be in clear set.
# Wooden sword splits Zol (0x13) → Gel (0x14); include gels. Keese often HP=0.
PREP_CLEAR_TYPES: tuple[int, ...] = (
    ZOL_OBJECT_TYPE,
    GEL_SPLIT_OBJECT_TYPE,
    GEL_ALT_OBJECT_TYPE,
    KEESE_OBJECT_TYPE,
)
# LIVE: after only 0x2b remain, doors raw=10 (U|L) and walk-UP → 0x4d.

# North-door UP approaches on 0x5d (order matters; x≈120 is primary).
UP_APPROACHES: tuple[tuple[int, int], ...] = (
    (120, 93),
    (120, 101),
    (112, 93),
    (128, 93),
    (100, 93),
    (140, 93),
    (120, 109),
    (96, 101),
    (144, 101),
)

# Bomb north stands if walk-UP fails after clear.
BOMB_NORTH_STANDS: tuple[tuple[int, int], ...] = (
    (120, 101),
    (96, 101),
    (144, 101),
    (120, 109),
    (112, 101),
    (128, 101),
)


def prep_5d_still_killable(snap: ZeldaSnapshot) -> list:
    """Enemies that must die before 0x5d UP shutter (ignore 0x2b)."""
    return live_killables(snap, PREP_CLEAR_TYPES)


def exit_raft_passage(env: Any, assist: Any | None, total: list[int]) -> dict[str, Any]:
    """Leave mode-9 0x0f via reverse channel + NW stairs UP → 0x69 play."""
    snap0 = read_snapshot(env.get_ram())
    before = room_fields(snap0, env.get_ram())
    if not (
        snap0.mode == 9
        and snap0.screen == ROOM_L3_RAFT_PASSAGE
        and snap0.level == LEVEL3
    ):
        if (
            snap0.mode == PLAY_MODE
            and snap0.level == LEVEL3
            and snap0.screen == ROOM_L3_SOUTH_DARKNUTS
        ):
            return {"ok": True, "skipped": True, "after": before}
        return {
            "ok": False,
            "error": (
                f"expected mode9 0x0f; got mode={snap0.mode} "
                f"sc=0x{snap0.screen:02x}"
            ),
            "before": before,
        }

    wp_log: list[dict] = []
    for tx, ty in PASSAGE_EXIT_WAYPOINTS:
        ok = goto(env, assist, total, tx, ty, tol=3, max_f=600)
        s = read_snapshot(env.get_ram())
        wp_log.append(
            {
                "target": [tx, ty],
                "ok": ok,
                "mode": s.mode,
                "sc": f"0x{s.screen:02x}",
                "xy": [s.link_x, s.link_y],
            }
        )
        if s.mode != 9 or s.screen != ROOM_L3_RAFT_PASSAGE:
            break

    stairs_i = 0
    for stairs_i in range(220):
        s = read_snapshot(env.get_ram())
        if s.mode != 9 or s.screen != ROOM_L3_RAFT_PASSAGE:
            break
        env.step(nes_action("UP"))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])

    idle(env, assist, total, 120)
    after = room_fields(read_snapshot(env.get_ram()), env.get_ram())
    ok = (
        after["mode"] == PLAY_MODE
        and after["level"] == LEVEL3
        and after["screen"] == ROOM_L3_SOUTH_DARKNUTS
    )
    return {
        "ok": ok,
        "before": before,
        "after": after,
        "waypoints": wp_log,
        "stairs_push_frames": stairs_i + 1,
    }


class Level3BossCombatMixin:
    """Prep clear / open_5d_up / Manhandla fight methods."""

    def clear_5d_prep(
        self,
        env: Any,
        assist: Any | None,
        total: list[int],
        *,
        max_frames: int = 14000,
    ) -> dict[str, Any]:
        """Clear Zol/Gel/Keese on 0x5d until only invuln 0x2b remain."""
        if self.poke_bombs is not None:
            poke_bombs(env, self.poke_bombs)
            ensure_bomb(env)

        patrol = (
            (64, 109),
            (120, 109),
            (176, 109),
            (176, 141),
            (176, 173),
            (120, 173),
            (64, 173),
            (64, 141),
            (120, 141),
            (100, 125),
            (140, 157),
            (80, 157),
            (160, 125),
        )
        spec = DungeonRoomSpec(
            spec_id="l3_5d_prep",
            source_room=ROOM_L3_BOSS_PREP,
            room_id=ROOM_L3_BOSS_PREP,
            entry=DoorRoute("LEFT", ((32, 141),)),
            enemy_types=PREP_CLEAR_TYPES,
            expected_enemy_count=1,
            alive_rule=AliveRule.TYPE,
            combat=CombatTuning(
                patrol=patrol,
                engage_distance=48,
                attack_phase=2,
                patrol_attack_period=5,
                patrol_attack_hold=3,
                engage_attack_period=4,
                engage_attack_hold=3,
            ),
            reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
            max_frames=max_frames,
            level=LEVEL3,
        )
        ctl = GenericDungeonRoomController(spec)
        zero_streak = 0
        bomb_cd = 0
        for frame in range(max_frames):
            snap = read_snapshot(env.get_ram())
            if snap.mode == 17:
                return {"ok": False, "error": "death", "frames": frame}
            if snap.screen != ROOM_L3_BOSS_PREP:
                return {
                    "ok": True,
                    "left_room": True,
                    "frames": frame,
                    "final": room_fields(snap, env.get_ram()),
                }
            live = prep_5d_still_killable(snap)
            if not live:
                zero_streak += 1
                doors_up = bool(snap.cur_opened_doors & DoorDir.UP)
                all_dead_ok = snap.room_all_dead >= 20
                if zero_streak >= 80 and doors_up and all_dead_ok:
                    idle(env, assist, total, 50)
                    s2 = read_snapshot(env.get_ram())
                    if prep_5d_still_killable(s2):
                        zero_streak = 0
                        continue
                    return {
                        "ok": True,
                        "frames": frame,
                        "final": room_fields(s2, env.get_ram()),
                    }
                if zero_streak > 60 and zero_streak % 40 < 8:
                    env.step(
                        nes_action(
                            ("LEFT", "UP", "RIGHT", "DOWN")[zero_streak // 40 % 4]
                        )
                    )
                else:
                    env.step(nes_idle_action())
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
                if zero_streak >= 600:
                    s2 = read_snapshot(env.get_ram())
                    return {
                        "ok": True,
                        "frames": frame,
                        "soft": True,
                        "final": room_fields(s2, env.get_ram()),
                    }
                continue
            zero_streak = 0

            if bomb_cd > 0:
                bomb_cd -= 1
            if live and bomb_cd <= 0 and snap.bombs > 0 and frame % 70 == 0:
                nearest = min(
                    live,
                    key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
                )
                dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
                if dist < 36:
                    ensure_bomb(env)
                    face = "RIGHT" if nearest.x >= snap.link_x else "LEFT"
                    env.step(nes_action(face, "B"))
                    total[0] += 1
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
                    bomb_cd = 85
                    continue

            act = ctl.step(snap)
            env.step(act.action)
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])

        return {
            "ok": False,
            "error": "timeout",
            "frames": max_frames,
            "final": room_fields(read_snapshot(env.get_ram()), env.get_ram()),
        }

    def open_5d_up(
        self,
        env: Any,
        assist: Any | None,
        total: list[int],
    ) -> dict[str, Any]:
        """Stabilize 0x5d → 0x4d: full killable clear → doors U|L → walk UP."""
        self._set_phase("clear_prep")
        report: dict[str, Any] = {
            "ok": False,
            "attempts": [],
            "clear": None,
            "pre": None,
            "post": None,
        }
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_L3_BOSS_PREP:
            report["error"] = f"expected 0x5d; got 0x{snap.screen:02x}"
            self.gate_5d_report = report
            return report

        idle(env, assist, total, 80)
        report["pre"] = room_fields(read_snapshot(env.get_ram()), env.get_ram())

        clr = self.clear_5d_prep(env, assist, total, max_frames=14000)
        report["clear"] = {
            "ok": clr.get("ok"),
            "frames": clr.get("frames"),
            "error": clr.get("error"),
            "final_doors": (clr.get("final") or {}).get("doors"),
            "final_types": (clr.get("final") or {}).get("type_counts"),
            "room_all_dead": (clr.get("final") or {}).get("room_all_dead"),
        }

        for wait_i in range(20):
            s = read_snapshot(env.get_ram())
            fields = room_fields(s, env.get_ram())
            killable = prep_5d_still_killable(s)
            report["attempts"].append(
                {
                    "kind": "settle_wait",
                    "i": wait_i,
                    "doors": fields["doors"],
                    "mask": fields["open_doorway_mask"],
                    "all_dead": fields["room_all_dead"],
                    "types": fields["type_counts"],
                    "killable": len(killable),
                }
            )
            if killable:
                self.clear_5d_prep(env, assist, total, max_frames=4000)
                continue
            if s.cur_opened_doors & DoorDir.UP:
                break
            idle(env, assist, total, 30)

        self._set_phase("open_up")
        st_base = None if self.continuous_mode else env.em.get_state()

        side_paths: tuple[tuple[tuple[int, int], ...], ...] = (
            ((160, 141), (160, 109), (120, 109), (120, 93)),
            ((80, 141), (80, 109), (120, 109), (120, 93)),
            ((120, 141), (120, 109), (120, 93)),
            ((120, 141),),
            ((120, 125),),
            ((120, 157),),
        )
        for path in side_paths:
            if not self.continuous_mode:
                self._restore_state(env, st_base)
                idle(env, assist, total, 2)
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            ok_path = True
            for tx, ty in path:
                if not goto(env, assist, total, tx, ty, tol=4, max_f=500):
                    ok_path = False
                    break
            s = read_snapshot(env.get_ram())
            report["attempts"].append(
                {
                    "kind": "side_path",
                    "path": [list(p) for p in path],
                    "ok_path": ok_path,
                    "at": [s.link_x, s.link_y],
                    "doors": room_fields(s, env.get_ram())["doors"],
                }
            )
            push_dir(env, assist, total, "UP", frames=PUSH_FRAMES + 80)
            after = room_fields(read_snapshot(env.get_ram()), env.get_ram())
            report["attempts"][-1]["result"] = (
                "room_change" if after["screen"] == ROOM_L3_BOSS else "blocked"
            )
            report["attempts"][-1]["to"] = after["sc"]
            if after["screen"] == ROOM_L3_BOSS:
                report["ok"] = True
                report["method"] = f"side_path_up@{path[0]}"
                report["post"] = after
                self.reached_4d = True
                self._set_phase("manhandla", f"0x4d via {report['method']}")
                obs, *_ = env.step(nes_idle_action())
                total[0] += 1
                save_rgb_png(obs, RECORDINGS_DIR / f"{self.tag}_boss_0x4d.png")
                self.gate_5d_report = report
                self.frames = total[0]
                return report

        for ax, ay in (() if self.continuous_mode else UP_APPROACHES):
            self._restore_state(env, st_base)
            idle(env, assist, total, 2)
            pr = exit_door(
                env,
                assist,
                total,
                "UP",
                x_force=ax,
                y_force=ay,
                push=PUSH_FRAMES + 80,
            )
            report["attempts"].append(
                {
                    "kind": "walk_up",
                    "xy": [ax, ay],
                    "result": pr["result"],
                    "to": pr["after"]["sc"] if pr["changed_room"] else None,
                    "at_doors": pr["at_door"]["doors"],
                }
            )
            if pr["changed_room"] and pr["after"]["screen"] == ROOM_L3_BOSS:
                report["ok"] = True
                report["method"] = f"walk_up@({ax},{ay})"
                report["post"] = pr["after"]
                self.reached_4d = True
                self._set_phase("manhandla", f"0x4d via {report['method']}")
                obs, *_ = env.step(nes_idle_action())
                total[0] += 1
                save_rgb_png(obs, RECORDINGS_DIR / f"{self.tag}_boss_0x4d.png")
                self.gate_5d_report = report
                self.frames = total[0]
                return report

        if self.poke_bombs is not None:
            poke_bombs(env, self.poke_bombs)
            ensure_bomb(env)
        for bx, by in (() if self.continuous_mode else BOMB_NORTH_STANDS[:3]):
            self._restore_state(env, st_base)
            idle(env, assist, total, 2)
            br = bomb_stand(env, assist, total, "UP", bx, by)
            report["attempts"].append(
                {
                    "kind": "bomb_north",
                    "stand": [bx, by],
                    "result": br["result"],
                    "to": br["after"]["sc"] if br["changed_room"] else None,
                }
            )
            if br["changed_room"] and br["after"]["screen"] == ROOM_L3_BOSS:
                report["ok"] = True
                report["method"] = f"bomb_north@({bx},{by})"
                report["post"] = br["after"]
                self.reached_4d = True
                self._set_phase("manhandla", f"0x4d via {report['method']}")
                obs, *_ = env.step(nes_idle_action())
                total[0] += 1
                save_rgb_png(obs, RECORDINGS_DIR / f"{self.tag}_boss_0x4d.png")
                self.gate_5d_report = report
                self.frames = total[0]
                return report

        report["post"] = room_fields(read_snapshot(env.get_ram()), env.get_ram())
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        save_rgb_png(obs, RECORDINGS_DIR / f"{self.tag}_finalroom_0x5d.png")
        self.traps.append(
            "0x5d UP gate residual — walk/bomb approaches exhausted"
        )
        self.gate_5d_report = report
        self.frames = total[0]
        return report

    def fight_manhandla(
        self,
        env: Any,
        assist: Any | None,
        total: list[int],
        *,
        max_frames: int = 16000,
    ) -> dict[str, Any]:
        """Circle + bomb near Manhandla heads (type 0x3c)."""
        self._set_phase("manhandla")
        log: list[dict] = []
        notes: list[str] = []
        if self.poke_bombs is not None:
            notes.append(f"RECON poke {poke_bombs(env, self.poke_bombs)}")
        ensure_bomb(env)
        bomb_cd = 0
        last_hps: list[int] | None = None
        dmg_events = 0
        enemy_type = MANHANDLA_OBJECT_TYPE

        for frame in range(max_frames):
            snap = read_snapshot(env.get_ram())
            ram = env.get_ram()
            tf = int(read_u8(ram, ADDR_TRIFORCE))
            if tf & LEVEL3_TRIFORCE_BIT:
                log.append(
                    {"event": "tf04", "frame": frame, **room_fields(snap, ram)}
                )
                self.tf04 = True
                self.boss_beaten = True
                self.dmg_events = dmg_events
                self._set_phase("done", "tf04")
                self.success = True
                result = {
                    "ok": True,
                    "tf04": True,
                    "frames": frame,
                    "dmg_events": dmg_events,
                    "log": log[-30:],
                    "notes": notes,
                    "final": room_fields(snap, ram),
                }
                self.fight_report = result
                self.notes.extend(notes)
                self.frames = total[0]
                return result
            if snap.mode == 17:
                result = {
                    "ok": False,
                    "error": "death",
                    "frames": frame,
                    "notes": notes,
                    "dmg_events": dmg_events,
                }
                self.fight_report = result
                return result
            if snap.mode != PLAY_MODE:
                env.step(nes_idle_action())
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
                continue

            heads = [
                o
                for o in snap.objects
                if 1 <= o.slot <= 10 and o.type_id == enemy_type and o.hp > 0
            ]
            heads_any = [
                o
                for o in snap.objects
                if 1 <= o.slot <= 10 and o.type_id == enemy_type
            ]
            hps = [o.hp for o in heads]
            if last_hps is not None and hps and last_hps and sum(hps) < sum(last_hps):
                dmg_events += 1
                log.append(
                    {
                        "event": "hp_drop",
                        "frame": frame,
                        "hps": hps,
                        "prev": last_hps,
                        "bombs": snap.bombs,
                    }
                )
            last_hps = hps

            if not heads_any and snap.room_all_dead >= 12 and frame > 80:
                log.append(
                    {
                        "event": "boss_dead",
                        "frame": frame,
                        **room_fields(snap, ram),
                    }
                )
                self._set_phase("collect_tf", "boss_dead")
                hc0 = snap.heart_containers
                for tx, ty in (
                    (128, 133),
                    (120, 141),
                    (112, 133),
                    (136, 141),
                    (120, 125),
                    (104, 141),
                    (144, 133),
                ):
                    goto(env, assist, total, tx, ty, tol=4, max_f=300)
                    s2 = read_snapshot(env.get_ram())
                    if s2.heart_containers > hc0:
                        notes.append(
                            f"HC at ({tx},{ty}) → hc={s2.heart_containers}"
                        )
                        break
                else:
                    for y in range(109, 173, 12):
                        for x in range(80, 161, 12):
                            goto(env, assist, total, x, y, tol=3, max_f=120)
                            if (
                                read_snapshot(env.get_ram()).heart_containers
                                > hc0
                            ):
                                notes.append(f"HC dense ({x},{y})")
                                break
                        else:
                            continue
                        break
                log.append(
                    {
                        "event": "post_hc",
                        **room_fields(
                            read_snapshot(env.get_ram()), env.get_ram()
                        ),
                    }
                )
                pr = exit_door(
                    env,
                    assist,
                    total,
                    "UP",
                    x_force=120,
                    y_force=93,
                    push=PUSH_FRAMES + 80,
                )
                log.append(
                    {
                        "event": "post_exit",
                        "dir": "UP",
                        "result": pr["result"],
                        "to": pr["after"]["sc"] if pr["changed_room"] else None,
                        "item": pr["after"].get("room_item_id"),
                    }
                )
                if pr["changed_room"] and pr["after"]["screen"] == 0x3D:
                    notes.append("entered TF room 0x3d")
                    for tx, ty in (
                        (120, 173),
                        (120, 141),
                        (124, 109),
                        (124, 93),
                        (120, 93),
                        (112, 93),
                        (136, 93),
                        (120, 125),
                        (128, 149),
                        (120, 149),
                    ):
                        goto(env, assist, total, tx, ty, tol=3, max_f=400)
                        if (
                            int(read_u8(env.get_ram(), ADDR_TRIFORCE))
                            & LEVEL3_TRIFORCE_BIT
                        ):
                            notes.append(f"TF 0x04 at ({tx},{ty})")
                            break
                    if not (
                        int(read_u8(env.get_ram(), ADDR_TRIFORCE))
                        & LEVEL3_TRIFORCE_BIT
                    ):
                        for y in range(93, 165, 8):
                            for x in range(96, 145, 8):
                                goto(env, assist, total, x, y, tol=2, max_f=80)
                                if (
                                    int(read_u8(env.get_ram(), ADDR_TRIFORCE))
                                    & LEVEL3_TRIFORCE_BIT
                                ):
                                    notes.append(f"TF dense ({x},{y})")
                                    break
                            else:
                                continue
                            break
                    for settle_i in range(200):
                        if (
                            int(read_u8(env.get_ram(), ADDR_TRIFORCE))
                            & LEVEL3_TRIFORCE_BIT
                        ):
                            s3 = read_snapshot(env.get_ram())
                            if s3.mode == PLAY_MODE or s3.mode == 18:
                                env.step(nes_idle_action())
                                total[0] += 1
                                if assist is not None:
                                    assist.apply_env(env, frame=total[0])
                                if s3.mode == PLAY_MODE and settle_i > 30:
                                    break
                                continue
                        env.step(nes_idle_action())
                        total[0] += 1
                        if assist is not None:
                            assist.apply_env(env, frame=total[0])
                elif not pr["changed_room"] and not self.continuous_mode:
                    st_post = env.em.get_state()
                    for direction in ("RIGHT", "LEFT", "DOWN"):
                        self._restore_state(env, st_post)
                        idle(env, assist, total, 2)
                        pr2 = exit_door(
                            env,
                            assist,
                            total,
                            direction,
                            push=PUSH_FRAMES + 40,
                        )
                        log.append(
                            {
                                "event": "post_exit",
                                "dir": direction,
                                "result": pr2["result"],
                                "to": (
                                    pr2["after"]["sc"]
                                    if pr2["changed_room"]
                                    else None
                                ),
                            }
                        )
                final = room_fields(read_snapshot(env.get_ram()), env.get_ram())
                tf_final = int(final.get("triforce") or 0)
                self.tf04 = bool(tf_final & LEVEL3_TRIFORCE_BIT)
                self.boss_beaten = True
                self.dmg_events = dmg_events
                if self.tf04:
                    self.success = True
                    self._set_phase("done", "tf04_post_hc")
                result = {
                    "ok": True,
                    "tf04": self.tf04,
                    "frames": frame,
                    "dmg_events": dmg_events,
                    "log": log[-40:],
                    "notes": notes,
                    "final": final,
                }
                self.fight_report = result
                self.notes.extend(notes)
                self.frames = total[0]
                return result

            if assist is not None and snap.bombs < 2 and self.poke_bombs:
                notes.append(f"topup {poke_bombs(env, self.poke_bombs)}")
                ensure_bomb(env)

            if not heads:
                env.step(
                    nes_action(
                        ("UP", "RIGHT", "DOWN", "LEFT")[frame // 15 % 4], "A"
                    )
                )
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
                continue

            nearest = min(
                heads,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
            dx = nearest.x - snap.link_x
            dy = nearest.y - snap.link_y
            if abs(dx) >= abs(dy):
                face = "RIGHT" if dx > 0 else "LEFT"
                approach = face
                circle = "DOWN" if (frame // 30) % 2 == 0 else "UP"
            else:
                face = "DOWN" if dy > 0 else "UP"
                approach = face
                circle = "RIGHT" if (frame // 30) % 2 == 0 else "LEFT"

            if bomb_cd > 0:
                bomb_cd -= 1

            if dist < 42 and bomb_cd <= 0 and snap.bombs > 0:
                ensure_bomb(env)
                if dist > 16:
                    env.step(nes_action(approach))
                else:
                    env.step(nes_action(face, "B"))
                    bomb_cd = 65
                    log.append(
                        {
                            "event": "bomb_place",
                            "frame": frame,
                            "at": [snap.link_x, snap.link_y],
                            "target": [nearest.x, nearest.y, nearest.hp],
                            "bombs": snap.bombs,
                        }
                    )
            elif dist > 48:
                if frame % 4 == 0:
                    env.step(nes_action(approach, "A"))
                else:
                    env.step(nes_action(approach))
            else:
                d = circle if frame % 3 else approach
                if frame % 3 == 0:
                    env.step(nes_action(d, "A"))
                else:
                    env.step(nes_action(d))

            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])

            if frame % 200 == 0:
                log.append(
                    {
                        "event": "tick",
                        "frame": frame,
                        "heads": len(heads),
                        "hps": hps,
                        "xy": [snap.link_x, snap.link_y],
                        "bombs": snap.bombs,
                        "sc": f"0x{snap.screen:02x}",
                    }
                )

        self.dmg_events = dmg_events
        result = {
            "ok": False,
            "error": "timeout",
            "frames": max_frames,
            "dmg_events": dmg_events,
            "log": log[-30:],
            "notes": notes,
            "final": room_fields(read_snapshot(env.get_ram()), env.get_ram()),
        }
        self.fight_report = result
        self.notes.extend(notes)
        self.frames = total[0]
        return result


__all__ = [
    "BOMB_NORTH_STANDS",
    "Level3BossCombatMixin",
    "PREP_CLEAR_TYPES",
    "UP_APPROACHES",
    "exit_raft_passage",
    "prep_5d_still_killable",
]
