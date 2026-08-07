"""Assisted Survival library: Level3Raft → Manhandla 0x4d → TF bit 0x04.

Directed LIVE path (2026-08-07)::

    0x0f mode9 reverse channel + NW stairs UP → 0x69
    UP → 0x59
    BOMB_RIGHT@(192,141) → 0x5a   *** walk-RIGHT sealed post-Raft ***
    RIGHT → 0x5b
    BOMB_RIGHT@(192,141) → 0x5c (3× Darknut)
    full clear (doors raw=3) → RIGHT @ y≈141 → 0x5d
    clear Zol+Keese only (ignore invuln 0x2b) → UP → 0x4d Manhandla 0x3c
    bombs → HC → TF room (bit 0x04)

Intervention: Survival. Not Clean STATUS.

Hybrid controller: high-level phase methods driven by a thin runner, using
``zelda_i.dungeon_ops`` for bomb/door/clear primitives (FrameAction alone is
awkward for bomb-stand + save/restore recon).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import save_rgb_png
from zelda_i.door_graph.core import DoorDir
from zelda_i.dungeon_ops import (
    PUSH_FRAMES,
    bomb_stand,
    ensure_bomb,
    exit_door,
    fight_clear,
    idle,
    live_killables,
    poke_bombs,
    room_fields,
)
from zelda_i.level3_boss_combat import (
    BOMB_NORTH_STANDS,
    Level3BossCombatMixin,
    PREP_CLEAR_TYPES,
    UP_APPROACHES,
    exit_raft_passage,
    prep_5d_still_killable,
)
from zelda_i.level3_dungeon import (
    BOMB_STAND_59_RIGHT,
    BOMB_STAND_5B_RIGHT,
    DARKNUT_OBJECT_TYPE,
    DOOR_5C_RIGHT_Y,
    INVULN_MOVER_0X2B,
    LEVEL3_TRIFORCE_BIT,
    MANHANDLA_OBJECT_TYPE,
    PASSAGE_EXIT_WAYPOINTS,
    ROOM_L3_BOSS,
    ROOM_L3_BOSS_PREP,
    ROOM_L3_BOMB_SHORTCUT,
    ROOM_L3_COMPASS,
    ROOM_L3_DARKNUTS,
    ROOM_L3_SOUTH_DARKNUTS,
    ROOM_L3_WEST_DARKNUTS,
    level3_manhandla_live,
)
from zelda_i.level3_overworld import LEVEL3
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import (
    PLAY_MODE,
    read_snapshot,
)

BOSS_PATH_PHASES: tuple[str, ...] = (
    "exit_passage",
    "up_69",
    "bomb_59",
    "right_5a",
    "bomb_5b",
    "clear_5c",
    "right_5d",
    "clear_prep",
    "open_up",
    "manhandla",
    "collect_tf",
    "done",
    "failed",
)

BOSS_PATH_MAX_FRAMES = 120_000


@dataclass
class Level3BossPathController(Level3BossCombatMixin):
    """Assisted Survival: Level3Raft → Manhandla → TF bit 0x04.

    Phases (see ``BOSS_PATH_PHASES``)::

        exit_passage → up_69 → bomb_59 → right_5a → bomb_5b → clear_5c
        → right_5d → clear_prep → open_up → manhandla → collect_tf → done

    Hybrid: methods take ``env`` / ``assist`` / ``total`` frame counter.
    """

    frames: int = 0
    phase: str = "exit_passage"
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    traps: list[str] = field(default_factory=list)
    path_log: list[dict] = field(default_factory=list)
    max_frames: int = BOSS_PATH_MAX_FRAMES
    poke_bombs: int | None = None  # RECON opt-in; durable default off
    tag: str = "l3_to_boss"
    # Outcome flags
    reached_5d: bool = False
    reached_4d: bool = False
    boss_beaten: bool = False
    tf04: bool = False
    manhandla_confirmed: bool = False
    dmg_events: int = 0
    last_error: str | None = None
    # Sub-reports
    path_to_5d_report: dict | None = None
    gate_5d_report: dict | None = None
    fight_report: dict | None = None

    def _set_phase(self, phase: str, note: str = "") -> None:
        if phase != self.phase:
            self.phase = phase
            if note:
                self.notes.append(note)

    def _fail(self, error: str) -> dict[str, Any]:
        self.failed = True
        self.last_error = error
        self._set_phase("failed", error)
        return {"ok": False, "error": error}

    def _maybe_poke(self, env: Any, *, note: bool = True) -> None:
        if self.poke_bombs is None:
            return
        msg = poke_bombs(env, self.poke_bombs)
        if note:
            self.notes.append(f"RECON poke {msg}")
        ensure_bomb(env)

    def path_to_5d(
        self,
        env: Any,
        assist: Any | None,
        total: list[int],
    ) -> dict[str, Any]:
        """Directed: Level3Raft / 0x0f → 0x5d boss prep."""
        path_log: list[dict] = []
        traps: list[str] = []
        notes: list[str] = []
        self._set_phase("exit_passage")

        # --- passage exit ---
        ex = exit_raft_passage(env, assist, total)
        path_log.append(
            {
                "step": "passage_exit",
                "ok": ex.get("ok"),
                "to": (ex.get("after") or {}).get("sc"),
                "error": ex.get("error"),
            }
        )
        if not ex.get("ok"):
            out = self._fail("passage_exit_failed")
            out["path_log"] = path_log
            out["exit"] = ex
            return out
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{self.tag}_exit_0x69.png")
        total[0] += 1

        if self.poke_bombs is not None:
            notes.append(f"RECON poke {poke_bombs(env, self.poke_bombs)}")
            ensure_bomb(env)

        # --- 0x69 UP → 0x59 ---
        self._set_phase("up_69", "entered_0x69")
        idle(env, assist, total, 60)
        snap = read_snapshot(env.get_ram())
        pr: dict[str, Any] = {}
        if snap.screen == ROOM_L3_SOUTH_DARKNUTS:
            live_dn = live_killables(snap, (DARKNUT_OBJECT_TYPE,))
            if live_dn:
                st = env.em.get_state()
                pr = exit_door(env, assist, total, "UP")
                if not (
                    pr["changed_room"]
                    and pr["after"]["screen"] == ROOM_L3_WEST_DARKNUTS
                ):
                    env.em.set_state(st)
                    idle(env, assist, total, 2)
                    clr = fight_clear(
                        env,
                        assist,
                        total,
                        enemy_types=(DARKNUT_OBJECT_TYPE,),
                        max_frames=6000,
                    )
                    path_log.append(
                        {
                            "step": "clear_69",
                            "ok": clr.get("ok"),
                            "frames": clr.get("frames"),
                        }
                    )
                    pr = exit_door(env, assist, total, "UP")
            else:
                pr = exit_door(env, assist, total, "UP")
            path_log.append(
                {
                    "step": "69_up",
                    "ok": pr["changed_room"]
                    and pr["after"]["screen"] == ROOM_L3_WEST_DARKNUTS,
                    "to": pr["after"]["sc"] if pr["changed_room"] else None,
                }
            )
            if not (
                pr["changed_room"]
                and pr["after"]["screen"] == ROOM_L3_WEST_DARKNUTS
            ):
                out = self._fail("failed_69_up")
                out.update(
                    {
                        "path_log": path_log,
                        "final": pr["after"],
                        "traps": traps,
                        "notes": notes,
                    }
                )
                return out

        # --- 0x59 BOMB_RIGHT → 0x5a (walk sealed) ---
        self._set_phase("bomb_59")
        idle(env, assist, total, 40)
        if self.poke_bombs is not None and read_snapshot(env.get_ram()).bombs < 2:
            poke_bombs(env, self.poke_bombs)
        bx, by = BOMB_STAND_59_RIGHT
        st = env.em.get_state()
        walk = exit_door(env, assist, total, "RIGHT", y_force=141)
        if walk["changed_room"] and walk["after"]["screen"] == ROOM_L3_COMPASS:
            path_log.append({"step": "59_right_walk", "ok": True, "to": "0x5a"})
        else:
            env.em.set_state(st)
            idle(env, assist, total, 2)
            if not walk["changed_room"]:
                traps.append("0x59 walk-RIGHT sealed post-Raft (expected)")
            br = bomb_stand(env, assist, total, "RIGHT", bx, by)
            path_log.append(
                {
                    "step": "59_bomb_right",
                    "ok": br["changed_room"]
                    and br["after"]["screen"] == ROOM_L3_COMPASS,
                    "to": br["after"]["sc"] if br["changed_room"] else None,
                    "stand": [bx, by],
                }
            )
            if not (
                br["changed_room"] and br["after"]["screen"] == ROOM_L3_COMPASS
            ):
                out = self._fail("failed_59_bomb_right")
                out.update(
                    {
                        "path_log": path_log,
                        "final": br["after"],
                        "traps": traps,
                        "notes": notes,
                    }
                )
                return out

        # --- 0x5a RIGHT → 0x5b ---
        self._set_phase("right_5a")
        idle(env, assist, total, 20)
        pr = exit_door(env, assist, total, "RIGHT", y_force=141)
        path_log.append(
            {
                "step": "5a_right",
                "ok": pr["changed_room"]
                and pr["after"]["screen"] == ROOM_L3_DARKNUTS,
                "to": pr["after"]["sc"] if pr["changed_room"] else None,
            }
        )
        if not (
            pr["changed_room"] and pr["after"]["screen"] == ROOM_L3_DARKNUTS
        ):
            out = self._fail("failed_5a_right")
            out.update(
                {
                    "path_log": path_log,
                    "final": pr["after"],
                    "traps": traps,
                    "notes": notes,
                }
            )
            return out

        # --- 0x5b BOMB_RIGHT → 0x5c ---
        self._set_phase("bomb_5b")
        idle(env, assist, total, 30)
        if self.poke_bombs is not None and read_snapshot(env.get_ram()).bombs < 2:
            poke_bombs(env, self.poke_bombs)
        bx, by = BOMB_STAND_5B_RIGHT
        st = env.em.get_state()
        walk = exit_door(env, assist, total, "RIGHT", y_force=141)
        if (
            walk["changed_room"]
            and walk["after"]["screen"] == ROOM_L3_BOMB_SHORTCUT
        ):
            path_log.append({"step": "5b_right_walk", "ok": True, "to": "0x5c"})
        else:
            env.em.set_state(st)
            idle(env, assist, total, 2)
            br = bomb_stand(env, assist, total, "RIGHT", bx, by)
            path_log.append(
                {
                    "step": "5b_bomb_right",
                    "ok": br["changed_room"]
                    and br["after"]["screen"] == ROOM_L3_BOMB_SHORTCUT,
                    "to": br["after"]["sc"] if br["changed_room"] else None,
                }
            )
            if not (
                br["changed_room"]
                and br["after"]["screen"] == ROOM_L3_BOMB_SHORTCUT
            ):
                out = self._fail("failed_5b_bomb_right")
                out.update(
                    {
                        "path_log": path_log,
                        "final": br["after"],
                        "traps": traps,
                        "notes": notes,
                    }
                )
                return out

        # --- 0x5c clear Darknuts → RIGHT @ y≈141 → 0x5d ---
        self._set_phase("clear_5c")
        idle(env, assist, total, 110)
        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_L3_BOMB_SHORTCUT:
            for _ in range(6):
                live = live_killables(
                    read_snapshot(env.get_ram()), (DARKNUT_OBJECT_TYPE,)
                )
                if live:
                    break
                idle(env, assist, total, 25)
            live = live_killables(
                read_snapshot(env.get_ram()), (DARKNUT_OBJECT_TYPE,)
            )
            if live:
                if self.poke_bombs is not None:
                    poke_bombs(env, self.poke_bombs)
                    ensure_bomb(env)
                clr = fight_clear(
                    env,
                    assist,
                    total,
                    enemy_types=(DARKNUT_OBJECT_TYPE,),
                    max_frames=16000,
                    use_bombs=True,
                    require_door_pair=True,
                )
                path_log.append(
                    {
                        "step": "clear_5c",
                        "ok": clr.get("ok"),
                        "frames": clr.get("frames"),
                        "doors": (clr.get("final") or {}).get("doors"),
                        "live_after": len(
                            live_killables(
                                read_snapshot(env.get_ram()),
                                (DARKNUT_OBJECT_TYPE,),
                            )
                        ),
                    }
                )
                still = live_killables(
                    read_snapshot(env.get_ram()), (DARKNUT_OBJECT_TYPE,)
                )
                if still:
                    traps.append(
                        f"0x5c clear residual: {len(still)} darknuts still live"
                    )
                    out = self._fail("failed_5c_clear")
                    out.update(
                        {
                            "path_log": path_log,
                            "final": room_fields(
                                read_snapshot(env.get_ram()), env.get_ram()
                            ),
                            "traps": traps,
                            "notes": notes,
                        }
                    )
                    return out
            else:
                path_log.append({"step": "clear_5c", "ok": True, "skipped": True})

            doors_ok = False
            for wait_i in range(40):
                s = read_snapshot(env.get_ram())
                live_n = len(live_killables(s, (DARKNUT_OBJECT_TYPE,)))
                raw = s.cur_opened_doors
                pair = (raw & (DoorDir.RIGHT | DoorDir.LEFT)) == (
                    DoorDir.RIGHT | DoorDir.LEFT
                )
                if live_n == 0 and pair and s.room_all_dead >= 10:
                    doors_ok = True
                    path_log.append(
                        {
                            "step": "5c_doors_ready",
                            "wait_i": wait_i,
                            "doors": {
                                "R": bool(raw & DoorDir.RIGHT),
                                "L": bool(raw & DoorDir.LEFT),
                                "raw": raw,
                            },
                            "all_dead": s.room_all_dead,
                        }
                    )
                    break
                if live_n > 0:
                    if self.poke_bombs is not None:
                        poke_bombs(env, self.poke_bombs)
                        ensure_bomb(env)
                    fight_clear(
                        env,
                        assist,
                        total,
                        enemy_types=(DARKNUT_OBJECT_TYPE,),
                        max_frames=5000,
                        use_bombs=True,
                        require_door_pair=True,
                    )
                else:
                    idle(env, assist, total, 30)

            if not doors_ok:
                f = room_fields(read_snapshot(env.get_ram()), env.get_ram())
                traps.append(
                    f"0x5c doors not raw=3 after clear (got raw={f['doors']['raw']} "
                    f"all_dead={f['room_all_dead']})"
                )

            self._set_phase("right_5d")
            st_5c = env.em.get_state()
            pr = None
            for ytry in (DOOR_5C_RIGHT_Y, 141):
                env.em.set_state(st_5c)
                idle(env, assist, total, 2)
                pr = exit_door(
                    env,
                    assist,
                    total,
                    "RIGHT",
                    y_force=ytry,
                    push=PUSH_FRAMES + 100,
                )
                path_log.append(
                    {
                        "step": "5c_right",
                        "y": ytry,
                        "ok": pr["changed_room"]
                        and pr["after"]["screen"] == ROOM_L3_BOSS_PREP,
                        "to": pr["after"]["sc"] if pr["changed_room"] else None,
                        "at_xy": [pr["at_door"]["x"], pr["at_door"]["y"]],
                        "doors": pr["at_door"]["doors"],
                        "mask": pr["at_door"]["open_doorway_mask"],
                        "all_dead": pr["at_door"]["room_all_dead"],
                    }
                )
                if pr["changed_room"] and pr["after"]["screen"] == ROOM_L3_BOSS_PREP:
                    break
            if not (
                pr
                and pr["changed_room"]
                and pr["after"]["screen"] == ROOM_L3_BOSS_PREP
            ):
                env.em.set_state(st_5c)
                idle(env, assist, total, 2)
                if self.poke_bombs is not None:
                    poke_bombs(env, self.poke_bombs)
                    ensure_bomb(env)
                br = bomb_stand(env, assist, total, "RIGHT", 192, 141)
                path_log.append(
                    {
                        "step": "5c_bomb_right",
                        "ok": br["changed_room"]
                        and br["after"]["screen"] == ROOM_L3_BOSS_PREP,
                        "to": br["after"]["sc"] if br["changed_room"] else None,
                    }
                )
                if br["changed_room"] and br["after"]["screen"] == ROOM_L3_BOSS_PREP:
                    pr = {"changed_room": True, "after": br["after"]}
            if not (
                pr
                and pr["changed_room"]
                and pr["after"]["screen"] == ROOM_L3_BOSS_PREP
            ):
                traps.append(
                    "0x5c walk-RIGHT y≈141 failed after raw=3 clear "
                    "(bomb-RIGHT fallback also failed)"
                )
                out = self._fail("failed_5c_right")
                out.update(
                    {
                        "path_log": path_log,
                        "final": (pr or {}).get("after")
                        or room_fields(
                            read_snapshot(env.get_ram()), env.get_ram()
                        ),
                        "traps": traps,
                        "notes": notes,
                    }
                )
                return out

        # Settle scroll → play in 0x5d
        for _ in range(90):
            snap = read_snapshot(env.get_ram())
            if (
                snap.screen == ROOM_L3_BOSS_PREP
                and snap.mode == PLAY_MODE
                and not snap.transitioning
            ):
                break
            env.step(nes_idle_action())
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
        snap = read_snapshot(env.get_ram())
        ok = snap.screen == ROOM_L3_BOSS_PREP and snap.level == LEVEL3
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        save_rgb_png(obs, RECORDINGS_DIR / f"{self.tag}_prep_0x5d.png")
        if ok:
            self.reached_5d = True
            self._set_phase("clear_prep", "arrived_0x5d")
        else:
            self._fail("not_at_5d")
        result = {
            "ok": ok,
            "path_log": path_log,
            "traps": traps,
            "notes": notes,
            "final": room_fields(snap, env.get_ram()),
            "mode_at_5d": snap.mode,
        }
        self.path_log.extend(path_log)
        self.traps.extend(traps)
        self.notes.extend(notes)
        self.path_to_5d_report = result
        self.frames = total[0]
        return result

    def confirm_manhandla(self, env: Any) -> list:
        """Record live Manhandla heads on current snapshot."""
        snap = read_snapshot(env.get_ram())
        heads = level3_manhandla_live(snap)
        self.manhandla_confirmed = len(heads) > 0
        if heads:
            self.notes.append(
                f"Manhandla type 0x3c: {len(heads)} live heads "
                f"hps={[o.hp for o in heads]}"
            )
        return heads

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "phase": self.phase,
            "frames": self.frames,
            "notes": list(self.notes),
            "traps": list(self.traps),
            "path_log": list(self.path_log),
            "reached_5d": self.reached_5d,
            "reached_4d": self.reached_4d,
            "boss_beaten": self.boss_beaten,
            "tf04": self.tf04,
            "manhandla_confirmed": self.manhandla_confirmed,
            "dmg_events": self.dmg_events,
            "last_error": self.last_error,
            "phases": list(BOSS_PATH_PHASES),
            "poke_bombs": self.poke_bombs,
            "path": (
                "0x0f exit→0x69 UP→0x59 BOMB_R→0x5a R→0x5b BOMB_R→0x5c "
                "clear R@y141→0x5d clear→UP→0x4d bombs→TF 0x04"
            ),
            "intervention_class": "survival",
            "track": "assisted",
            "geometry": {
                "bomb_stand_59": list(BOMB_STAND_59_RIGHT),
                "bomb_stand_5b": list(BOMB_STAND_5B_RIGHT),
                "door_5c_right_y": DOOR_5C_RIGHT_Y,
                "passage_exit_waypoints": [list(w) for w in PASSAGE_EXIT_WAYPOINTS],
                "prep_clear_types": [f"0x{t:02x}" for t in PREP_CLEAR_TYPES],
                "invuln_ignored": f"0x{INVULN_MOVER_0X2B:02x}",
                "manhandla_type": f"0x{MANHANDLA_OBJECT_TYPE:02x}",
            },
        }


__all__ = [
    "BOMB_NORTH_STANDS",
    "BOSS_PATH_MAX_FRAMES",
    "BOSS_PATH_PHASES",
    "Level3BossPathController",
    "PREP_CLEAR_TYPES",
    "UP_APPROACHES",
    "exit_raft_passage",
    "prep_5d_still_killable",
]
