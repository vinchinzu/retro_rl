"""Survival-spine L5 hops from L4 TF settle through TF 0x10.

Whistle / Digdogger suffixes stay env-stepping library calls. Recorder
cellar leftover is 0x04 mode 9 (135,141).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.anchors import LEVEL5_ENTRY_ROOM, LEVEL5_TF_ROOM, TF_BIT_L4, TF_BIT_L5
from zelda_i.chain import ControllerStageResult
from zelda_i.level5_dungeon import (
    ROOM_66_SPEC,
    ROOM_77_SPEC,
    ROOM_L5_GIBDO_66,
    ROOM_L5_POLS_77,
    make_pols_voice_controller,
)
from zelda_i.level5_path import (
    BLUE_DARKNUT_TYPE,
    ROOM_L5_BLUE_64,
    ROOM_L5_PASSAGE_06,
    ROOM_L5_WHISTLE_05,
    ROOM_L5_WHISTLE_ITEM,
    bomb_west_from_65,
    bomb_west_from_66,
    cellar_other_mouth,
    fight_blue_darknuts,
    hunt_whistle,
    key_west_to,
    level5_east_key_step,
    make_return_66_controller,
    push_block_stairs,
    take_center_stairs_64,
    take_whistle_04,
)
from zelda_i.level5_overworld import (
    LEVEL5_LEVEL_ID,
    POST_L4_PATH_MAX_FRAMES,
    POST_L4_SETTLE_MAX_FRAMES,
    PostL4TriforceSettleController,
    make_post_l4_level5_controller,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot
from zelda_i.spine_hops import SpineHop, attach_hops, fight_stage, play_ready

__all__ = [
    "L5_STOPS",
    "L5_THROUGH",
    "ROOM_66_SPINE_SPEC",
    "Level5EastKeyNavController",
    "attach_level5_tf_suffix",
    "attach_level5_whistle_suffix",
    "continue_level5_spine",
    "validate_l5_endpoint",
]

L5_THROUGH: tuple[str, ...] = (
    "level5-entry",
    "level5-clear66",
    "level5-east77",
    "level5-whistle",
    "level5-exit04",
    "level5",
)
L5_STOPS: dict[str, str] = {
    "level5-entry": "level5_entry_0x76",
    "level5-clear66": "level5_clear_0x66",
    "level5-east77": "level5_east_key_0x77",
    "level5-whistle": "level5_whistle_0x04",
    "level5-exit04": "level5_exit_0x04",
    "level5": "level5_triforce_0x10",
}
_L5_PREFIX = frozenset(L5_THROUGH[:3])

# v1 leftover 0x66 (119,173) timeout 12000f, 2/3 Gibdo north of the river.
# Cardinal patrol never crossed; OccupancyWalker miss-blocks water.
ROOM_66_SPINE_SPEC = replace(
    ROOM_66_SPEC,
    combat=replace(
        ROOM_66_SPEC.combat,
        occupancy_patrol=True,
        occupancy_bounds=(16, 216, 77, 205),
    ),
    max_frames=20000,
)


def level5_entry_success(snap: ZeldaSnapshot, **_) -> bool:
    """Room-ready Lizard entry 0x76 with L4 inventory. Do not require 0x66."""
    return (
        play_ready(
            snap, level=LEVEL5_LEVEL_ID, screen=LEVEL5_ENTRY_ROOM, tf_bit=TF_BIT_L4,
            item="raft",
        )
        and snap.ladder > 0
    )


def level5_clear66_success(snap: ZeldaSnapshot, **_) -> bool:
    """Cleared 0x66; east door bit 0x08. Do not poke the key door yet."""
    spec = ROOM_66_SPEC
    return (
        play_ready(
            snap, level=spec.level, screen=ROOM_L5_GIBDO_66, spec=spec,
            tf_bit=TF_BIT_L4,
        )
        and snap.room_all_dead >= spec.reward.settle_all_dead
        and (snap.cur_opened_doors & spec.required_open_doors)
        == spec.required_open_doors
        and snap.ladder > 0
    )


@dataclass
class Level5EastKeyNavController:
    """Walk cleared 0x66 → 0x76 east key door → 0x77. No combat. No pokes."""

    max_frames: int = 8000
    settle_frames: int = 40
    frames: int = 0
    settle_left: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    last_room: int = -1

    def report(self) -> dict:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": "level5_east_key_nav_0x77",
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")
        if snap.screen != self.last_room:
            self.notes.append(
                f"room_0x{snap.screen:02x}_f{self.frames}_xy={snap.link_x},{snap.link_y}_k={snap.keys}"
            )
            self.last_room = snap.screen
        if (
            snap.level == LEVEL5_LEVEL_ID
            and snap.screen == ROOM_L5_POLS_77
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            if self.settle_left <= 0 and "settling_77" not in self.notes:
                self.settle_left = self.settle_frames
                self.notes.append("settling_77")
            if self.settle_left > 0:
                self.settle_left -= 1
                if self.settle_left > 0:
                    return FrameAction(nes_idle_action(), "settle_77")
            self.success = True
            self.notes.append("arrived_77")
            return FrameAction(nes_idle_action(), "arrived_77")
        return level5_east_key_step(snap)


def _east77_stages():
    nav = Level5EastKeyNavController()
    fight = make_pols_voice_controller()
    return (
        ("level5_east_key_0x77", nav, nav.max_frames),
        ("level5_clear_0x77", fight, ROOM_77_SPEC.max_frames),
    )


def level5_east77_success(snap: ZeldaSnapshot, **_) -> bool:
    """Play-ready empty 0x77. Spine already carries keys from 0x66."""
    return (
        play_ready(
            snap, level=LEVEL5_LEVEL_ID, screen=ROOM_L5_POLS_77, spec=ROOM_77_SPEC,
            tf_bit=TF_BIT_L4,
        )
        and snap.ladder > 0
    )


def level5_whistle_success(snap: ZeldaSnapshot, *, whistle: int) -> bool:
    """Recorder owned in cellar 0x04 / play 0x05. Do not require Digdogger."""
    return (
        whistle >= 1
        and snap.level == LEVEL5_LEVEL_ID
        and bool(snap.triforce & TF_BIT_L4)
        and snap.ladder > 0
        and snap.screen in (ROOM_L5_WHISTLE_ITEM, ROOM_L5_WHISTLE_05)
    )


def level5_exit04_success(snap: ZeldaSnapshot, *, whistle: int) -> bool:
    """Play-ready 0x05 after the Recorder cellar ladder. Do not require 0x24."""
    return (
        whistle >= 1
        and play_ready(
            snap, level=LEVEL5_LEVEL_ID, screen=ROOM_L5_WHISTLE_05, tf_bit=TF_BIT_L4,
        )
        and snap.ladder > 0
    )


def level5_tf_success(snap: ZeldaSnapshot) -> bool:
    """L5 shard in the Triforce room. ``validate_l5_endpoint`` is the report gate."""
    return (
        snap.level == LEVEL5_LEVEL_ID
        and snap.screen == LEVEL5_TF_ROOM
        and bool(snap.triforce & TF_BIT_L5)
    )


def l5_hops() -> tuple[SpineHop, ...]:
    return (
        SpineHop(
            "level5-entry",
            "level5_entry_0x76",
            (
                (
                    "settle_l4_tf",
                    PostL4TriforceSettleController(),
                    POST_L4_SETTLE_MAX_FRAMES,
                ),
                (
                    "enter_level5",
                    make_post_l4_level5_controller(),
                    POST_L4_PATH_MAX_FRAMES,
                ),
            ),
            level5_entry_success,
        ),
        SpineHop(
            "level5-clear66",
            "level5_clear_0x66",
            (fight_stage("level5_clear_0x66", ROOM_66_SPINE_SPEC),),
            level5_clear66_success,
        ),
        SpineHop(
            "level5-east77",
            "level5_east_key_0x77",
            _east77_stages,
            level5_east77_success,
        ),
    )


def _step_return_66(env, assist, total: list[int]) -> bool:
    ctl = make_return_66_controller()
    while not ctl.success and not ctl.failed and ctl.frames < ctl.max_frames:
        action = ctl.step(read_snapshot(env.get_ram()))
        env.step(action.action)
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    snap = read_snapshot(env.get_ram())
    return bool(ctl.success and snap.screen == ROOM_L5_GIBDO_66)


def run_level5_whistle_suffix(env, *, assist, frame_base: int):
    """0x77 leftover → 0x66 bomb-west → 0x04 Recorder. Env-stepping, no pokes."""
    from zelda_i.ram import ADDR_WHISTLE, read_u8

    total = [int(frame_base)]
    hops: list[dict] = []
    if not _step_return_66(env, assist, total):
        return False, total[0], {"failed": "return_66", "hops": hops}
    hops.append({"hop": "return_66", "ok": True})

    bomb66 = bomb_west_from_66(env, assist, total)
    hops.append({"hop": "bomb_west_66", "ok": bool(bomb66.get("success"))})
    if not bomb66.get("success"):
        return False, total[0], {"failed": "bomb_west_66", "hops": hops}

    bomb65 = bomb_west_from_65(env, assist, total)
    hops.append({"hop": "bomb_west_65", "ok": bool(bomb65.get("success"))})
    if not bomb65.get("success"):
        return False, total[0], {"failed": "bomb_west_65", "hops": hops}

    snap = read_snapshot(env.get_ram())
    n_dn = sum(
        1
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id == BLUE_DARKNUT_TYPE and obj.hp > 0
    )
    if n_dn:
        fight64 = fight_blue_darknuts(
            env, assist, total, ROOM_L5_BLUE_64, expected=n_dn, source=0x65
        )
        hops.append({"hop": "fight_64", "ok": bool(fight64.get("ok"))})
        if not fight64.get("ok"):
            return False, total[0], {"failed": "fight_64", "hops": hops}
        pushed64 = push_block_stairs(env, assist, total, ROOM_L5_BLUE_64)
        stairs = pushed64 if pushed64.get("success") else take_center_stairs_64(
            env, assist, total
        )
    else:
        stairs = take_center_stairs_64(env, assist, total)
    hops.append({"hop": "stairs_64", "ok": bool(stairs.get("success"))})
    if not stairs.get("success"):
        return False, total[0], {"failed": "stairs_64", "hops": hops}

    cellar = cellar_other_mouth(env, assist, total)
    hops.append({"hop": "cellar_07", "ok": bool(cellar.get("success"))})
    if not cellar.get("success"):
        return False, total[0], {"failed": "cellar_07", "hops": hops}

    west = key_west_to(env, assist, total, ROOM_L5_WHISTLE_05)
    hops.append({"hop": "key_west_05", "ok": bool(west.get("success"))})
    if not west.get("success"):
        return False, total[0], {"failed": "key_west_05", "hops": hops}

    snap = read_snapshot(env.get_ram())
    n_dn = sum(
        1
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id == BLUE_DARKNUT_TYPE and obj.hp > 0
    )
    fight = fight_blue_darknuts(
        env,
        assist,
        total,
        ROOM_L5_WHISTLE_05,
        expected=max(6, n_dn),
        source=ROOM_L5_PASSAGE_06,
    )
    hops.append({"hop": "fight_05", "ok": bool(fight.get("ok"))})
    if not fight.get("ok"):
        return False, total[0], {"failed": "fight_05", "hops": hops}

    pushed = push_block_stairs(env, assist, total, ROOM_L5_WHISTLE_05)
    hops.append({"hop": "push_05", "ok": bool(pushed.get("success"))})
    if not pushed.get("success"):
        return False, total[0], {"failed": "push_05", "hops": hops}

    snap = read_snapshot(env.get_ram())
    if snap.screen == ROOM_L5_WHISTLE_ITEM or snap.mode in (9, 11):
        walk = take_whistle_04(env, assist, total)
    else:
        walk = hunt_whistle(env, assist, total)
    whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    hops.append({"hop": "take_whistle", "ok": whistle >= 1, "got": walk.get("got")})
    ok = whistle >= 1
    return ok, total[0], {"failed": None if ok else "whistle_still_0", "hops": hops}


class _SuffixReport:
    def __init__(self, detail, success: bool):
        self._detail = detail
        self.success = success

    def report(self) -> dict:
        return self._detail


def record_env_suffix(run, name: str, ok: bool, end_frame: int, detail) -> None:
    """Append one env-stepping suffix stage onto ``run``."""
    run.stages.append(
        ControllerStageResult(
            name=name,
            controller=_SuffixReport(detail, ok),
            max_frames=40000,
            frames=end_frame - run.end_frame,
            success=ok,
            frame_base=run.end_frame,
            end_frame=end_frame,
        )
    )
    run.end_frame = end_frame


def _finish_suffix_obs(env, run, *, assist) -> ZeldaSnapshot:
    obs, *_ = env.step(nes_idle_action())
    run.end_frame += 1
    if assist is not None:
        assist.apply_env(env, frame=run.end_frame)
    run.obs = obs
    return read_snapshot(env.get_ram())


def attach_level5_whistle_suffix(env, run, *, assist) -> bool:
    """Append East Key → Recorder onto a spine run. False if whistle missing."""
    from zelda_i.ram import ADDR_WHISTLE, read_u8

    ok, end_frame, detail = run_level5_whistle_suffix(
        env, assist=assist, frame_base=run.end_frame
    )
    record_env_suffix(run, "level5_whistle_0x04", ok, end_frame, detail)
    snap = _finish_suffix_obs(env, run, assist=assist)
    whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    run.success = ok and level5_whistle_success(snap, whistle=whistle)
    if not run.success:
        run.failed_stage = "level5_whistle_0x04"
        if isinstance(detail, dict) and detail.get("failed"):
            run.failed_stage = f"level5_whistle_{detail['failed']}"
    return run.success


def attach_level5_tf_suffix(env, run, *, assist, through: str) -> bool:
    """Append 0x04 exit then Digdogger/TF. Library path; no pokes."""
    from zelda_i.level5_boss_path import STOP_EXIT04, STOP_TRIFORCE, run_level5_tf_suffix
    from zelda_i.ram import ADDR_WHISTLE, read_u8

    stop_at = STOP_EXIT04 if through == "level5-exit04" else STOP_TRIFORCE
    stage_name = L5_STOPS["level5-exit04"] if stop_at == STOP_EXIT04 else L5_STOPS["level5"]
    ok, end_frame, detail = run_level5_tf_suffix(
        env, assist=assist, frame_base=run.end_frame, stop_at=stop_at
    )
    record_env_suffix(run, stage_name, ok, end_frame, detail)
    snap = _finish_suffix_obs(env, run, assist=assist)
    whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    if stop_at == STOP_EXIT04:
        run.success = ok and level5_exit04_success(snap, whistle=whistle)
    else:
        run.success = ok and level5_tf_success(snap)
    if not run.success:
        run.failed_stage = stage_name
        if isinstance(detail, dict) and detail.get("failed"):
            run.failed_stage = f"{stage_name}_{detail['failed']}"
    return run.success


def continue_level5_spine(
    env,
    run,
    *,
    through: str,
    run_stages,
    room_timer=None,
    assist=None,
    on_frame=None,
) -> None:
    """Attach L5 suffix after L4 TF. Mutates ``run``; caller returns it."""
    attach_hops(
        env,
        run,
        l5_hops(),
        through=through,
        run_stages=run_stages,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    )
    if not run.success or through in _L5_PREFIX:
        return
    attach_level5_whistle_suffix(env, run, assist=assist)
    if not run.success or through == "level5-whistle":
        return
    attach_level5_tf_suffix(env, run, assist=assist, through=through)


def validate_l5_endpoint(report: dict[str, object]) -> None:
    """Accept only a continuous L5 TF stop (no stitch manifest)."""
    if not report.get("continuous_emulator_session"):
        raise ValueError("L5 endpoint must be a continuous emulator session")
    if report.get("seamed") or report.get("tape_kind") == "state_seamed_viewing_compose":
        raise ValueError("seamed L5 tapes are not a spine endpoint")
    final = report.get("final")
    if not isinstance(final, dict):
        raise ValueError("Level 5 report has no final snapshot")
    if not report.get("ok"):
        raise ValueError("Level 5 report is not successful")
    room = int(final.get("screen", final.get("room", -1)))
    if int(final.get("level", -1)) != 5 or room != LEVEL5_TF_ROOM:
        raise ValueError("Level 5 report does not end in the Triforce room (0x14)")
    if int(final.get("triforce", 0)) & TF_BIT_L5 == 0:
        raise ValueError("Level 5 report does not have Triforce bit 0x10")
    assist = report.get("assist")
    if not isinstance(assist, dict):
        raise ValueError("Level 5 report is missing Survival telemetry")
    if int(assist.get("progression_writes", -1)) != 0:
        raise ValueError("Level 5 report has progression writes")
    if int(assist.get("capacity_writes", -1)) != 0:
        raise ValueError("Level 5 report has capacity writes")
