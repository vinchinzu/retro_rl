"""South-stand four-head Gleeok in play 0x3C. Heart if 0x1A drops.

rr-5eb2 / rr-6o7.2. Pin ``Level8Interior3CNorthReconFixture``.
Do not chase heads while the body remains. Do not poke HP. No TF hop.

RAM CLAIM (written before the first live trial):

  From play 0x3C leftover (120,189), idle until body type 0x45 is
  present, south-stand (body.x, body.y+22) face UP+A, Magical Sword,
  fireball dodge manhattan ≤14. Body type 0x45 goes absent. Do not
  chase heads while body remains. Hearts Survival-refilled. Keys/bombs/
  MK/TF unchanged except the natural heart-container +1 when 0x1A is
  collected. Deaths 0. progression_writes=0 capacity_writes=0. No HP poke.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l8_3c_gleeok.py \\
        --from-state Level8Interior3CNorthReconFixture \\
        --tag 20260904_F1 --infinite-life --no-video
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name
from zelda_i.level8.gleeok import RAM_CLAIM, gleeok_4head_live, make_four_head_gleeok_controller
from zelda_i.level8.gleeok_entry import DEST, DEST_POSE
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_KEYS,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_KEY,
    ADDR_MAP,
    ADDR_OBJ_TYPE,
    ADDR_RUPEES,
    ADDR_SELECTED_ITEM,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot

LEVEL8 = 8
POST_L7_TRIFORCE = 0x7F
POSE_TOL = 8
DEATH_MODE = 17
CENSUS_IDLE = 60
SOURCE_STATE = "Level8Interior3CNorthReconFixture"


class ProbeStop(RuntimeError):
    """Expected fail-closed halt with a reportable reason."""


def _inventory(env: Any) -> dict[str, int]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    u = lambda addr: int(read_u8(ram, addr))
    return {
        "sword": u(ADDR_SWORD), "bombs": u(ADDR_BOMBS), "bow": u(ADDR_BOW),
        "arrows": u(ADDR_ARROWS), "candle": u(ADDR_CANDLE), "keys": u(ADDR_KEYS),
        "rupees": u(ADDR_RUPEES), "magic_key": u(ADDR_MAGIC_KEY),
        "selected_item": u(ADDR_SELECTED_ITEM), "triforce": u(ADDR_TRIFORCE),
        "map": u(ADDR_MAP),
        "health": int(snap.health), "heart_containers": int(snap.heart_containers),
    }


def _typed(snap: ZeldaSnapshot, *, live: bool = False) -> list:
    return [
        obj for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id not in (0, 0xFF)
        and (not live or obj.hp > 0)
    ]


def _obj_row(obj: Any) -> dict[str, Any]:
    return {
        "slot": int(obj.slot), "type": int(obj.type_id),
        "type_hex": f"0x{obj.type_id:02X}", "type_name": object_name(obj.type_id),
        "xy": [int(obj.x), int(obj.y)], "hp": int(obj.hp),
    }


def _glance(env: Any) -> dict[str, Any]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    result = leftover_from_snapshot(snap)
    result.update(
        level=int(snap.level),
        screen_hex=f"0x{snap.screen:02X}",
        xy=[int(snap.link_x), int(snap.link_y)],
        tile=int(snap.colliding_tile),
        facing=int(snap.facing),
        transitioning=bool(snap.transitioning),
        room_item_hex=f"0x{snap.room_item_id:02X}",
        treasure={
            "x": int(read_u8(ram, ADDR_LINK_X + 19)),
            "y": int(read_u8(ram, ADDR_LINK_Y + 19)),
            "type": int(read_u8(ram, ADDR_OBJ_TYPE + 19)),
            "state": int(read_u8(ram, 0x00AC + 19)),
        },
        cur_opened_doors=int(snap.cur_opened_doors),
        open_doorway_mask=int(snap.open_doorway_mask),
        inventory=_inventory(env),
        live_objects=[_obj_row(obj) for obj in _typed(snap, live=True)],
        typed_objects=[_obj_row(obj) for obj in _typed(snap)],
    )
    return result


def _pin_ok(start: dict[str, Any]) -> bool:
    inv, (x, y) = start["inventory"], start["xy"]
    ox, oy = DEST_POSE
    return (
        start["level"] == LEVEL8 and start["mode"] == PLAY_MODE
        and int(start["screen"]) == int(DEST)
        and abs(int(x) - int(ox)) <= POSE_TOL
        and abs(int(y) - int(oy)) <= POSE_TOL
        and inv["magic_key"] == 1 and inv["keys"] == 8
        and inv["bombs"] == 5 and inv["triforce"] == POST_L7_TRIFORCE
        and inv["sword"] == 3
    )


@dataclass
class ObservedEnv:
    env: Any
    assist: Any
    tag: str
    frame: int = 0

    def __post_init__(self) -> None:
        self.phase = "start"
        self.reason = "start"
        self.latest_obs: Any | None = None
        snap = read_snapshot(self.env.get_ram())
        self.last_key = (int(snap.level), int(snap.mode), int(snap.screen))
        self.last_xy = (int(snap.link_x), int(snap.link_y))
        self.stuck = 0
        self.saw_body = False
        self.body_gone_shot = False
        self.samples: list[dict[str, Any]] = []
        self.screenshots: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def set_reason(self, phase: str, reason: str) -> None:
        self.phase = phase
        self.reason = reason

    def save_shot(self, label: str) -> Path:
        snap = read_snapshot(self.env.get_ram())
        path = RECORDINGS_DIR / (
            f"l8_3c_gleeok_{self.tag}_{label}_f{self.frame}_"
            f"L{snap.level}_s{snap.screen:02x}_m{snap.mode}.png"
        )
        save_rgb_png(
            self.latest_obs if self.latest_obs is not None else self.env.render(),
            path,
        )
        self.screenshots.append(str(path))
        return path

    def step(self, action: Any) -> Any:
        result = self.env.step(action)
        self.latest_obs = result[0]
        self.frame += 1
        snap = read_snapshot(self.env.get_ram())
        bodies = gleeok_4head_live(snap)
        if bodies and not self.saw_body:
            self.saw_body = True
            self.save_shot("body_seen")
        if self.saw_body and not bodies and not self.body_gone_shot:
            self.body_gone_shot = True
            self.save_shot("body_gone")
        key = (int(snap.level), int(snap.mode), int(snap.screen))
        xy = (int(snap.link_x), int(snap.link_y))
        if key != self.last_key:
            self.save_shot("transition")
            self.last_key, self.stuck = key, 0
        elif xy == self.last_xy:
            self.stuck += 1
            if self.stuck % 500 == 0:
                self.save_shot(f"stuck_{self.stuck}")
        else:
            self.stuck = 0
        self.last_xy = xy
        return result


def _save_fixture(
    raw_env: Any, *, fixture_name: str, census: dict[str, Any],
    magic_key: int, keys: int, bombs: int,
) -> dict[str, Any]:
    from retro_harness.env import save_state, state_path
    from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance

    after = read_snapshot(raw_env.get_ram())
    path = save_state(raw_env, GAME_DIR, GAME, fixture_name)
    source_path = state_path(GAME_DIR, GAME, SOURCE_STATE)
    result = {
        "ok": True, "source_state": SOURCE_STATE,
        "fixture_state": fixture_name, "state": compact_snapshot(after),
        "census": census, "magic_key": magic_key, "keys": keys, "bombs": bombs,
        "fixture_writes": [],
    }
    write_state_provenance(
        path,
        source_state_path=source_path if source_path.exists() else None,
        request={
            "bead": "rr-5eb2",
            "phase": "level8_interior_0x3c_gleeok_kill",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "natural_entry": False,
            "fixture_writes": [],
            "notes": [
                "south-stand 0x45 kill in play 0x3C, heart if 0x1A,"
                " do not chain TF, not on L8_THROUGH",
            ],
        },
        selected_trial=result,
        natural_entry=False,
    )
    return {
        "state_path": str(path),
        "provenance": str(path.with_suffix(".provenance.json")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    add_common_args(
        parser, default_state=SOURCE_STATE, default_tag="20260904_F1"
    )
    parser.add_argument("--save-fixture", default=None)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if not args.infinite_life:
        raise SystemExit("this fixture fight probe requires --infinite-life")

    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    assist = make_assist(True)
    raw_env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    payload: dict[str, Any] = {
        "trial": "0x3c_gleeok_south_stand",
        "from_state": args.from_state,
        "fixture_only": True, "natural_entry": False, "route_eligible": False,
        "infinite_life": True,
        "prediction": {
            "written_before_run": True,
            "claim": RAM_CLAIM,
            "contingency": (
                "If ghp never drops, halt (sideways facing). Need a bare"
                " UP face then UP+A. Do not spray. Do not chase 0x46"
                " while 0x45 remains. Do not poke HP."
            ),
        },
    }
    env: ObservedEnv | None = None
    try:
        obs, _ = reset_obs(raw_env)
        env = ObservedEnv(raw_env, assist, args.tag)
        env.latest_obs = obs
        start = _glance(env)
        payload["start"] = start
        env.save_shot("start")
        if not _pin_ok(start):
            raise ProbeStop("pin_mismatch")
        assist.apply_env(env, frame=0)
        inv0 = start["inventory"]
        hc_in = int(inv0["heart_containers"])
        keys_in, bombs_in = int(inv0["keys"]), int(inv0["bombs"])
        mk_in, tf_in = int(inv0["magic_key"]), int(inv0["triforce"])

        ctl = make_four_head_gleeok_controller()
        for _ in range(ctl.max_frames):
            snap = read_snapshot(env.get_ram())
            if snap.mode == DEATH_MODE:
                raise ProbeStop("death")
            act = ctl.step(snap)
            env.set_reason("fight", act.reason)
            env.step(act.action)
            assist.apply_env(env, frame=env.frame)
            if ctl.success or ctl.failed:
                break

        payload["controller"] = ctl.report()
        if ctl.failed or not ctl.success:
            payload["failed"] = (
                (ctl.notes[-1] if ctl.notes else "") or "gleeok_failed"
            )
            payload["success"] = False
            payload["final"] = _glance(env)
            env.save_shot("final_halt")
        else:
            arrival = _glance(env)
            env.save_shot("arrived")
            if int(arrival["inventory"]["heart_containers"]) > hc_in:
                env.save_shot("heart")
            for _ in range(CENSUS_IDLE):
                env.step(nes_idle_action())
                assist.apply_env(env, frame=env.frame)
            final = _glance(env)
            env.save_shot("final")
            settled = read_snapshot(raw_env.get_ram())
            deaths = int(assist.telemetry.deaths)
            prog = int(assist.telemetry.progression_writes)
            cap = int(assist.telemetry.capacity_writes)
            hc_out = int(final["inventory"]["heart_containers"])
            keys_out = int(final["inventory"]["keys"])
            bombs_out = int(final["inventory"]["bombs"])
            mk_out = int(final["inventory"]["magic_key"])
            tf_out = int(final["inventory"]["triforce"])
            misses = [
                msg for cond, msg in (
                    (settled.level != LEVEL8, f"left L8 L{settled.level}"),
                    (settled.screen != DEST, f"left 0x3C to 0x{settled.screen:02x}"),
                    (settled.mode != PLAY_MODE, f"mode {settled.mode}"),
                    (gleeok_4head_live(settled), "body 0x45 still present"),
                    (mk_out != 1, f"magic_key {mk_in}->{mk_out}"),
                    (tf_out != POST_L7_TRIFORCE, f"triforce {tf_out:#x}"),
                    (keys_out != keys_in, f"keys {keys_in}->{keys_out}"),
                    (bombs_out != bombs_in, f"bombs {bombs_in}->{bombs_out}"),
                    (deaths, f"deaths {deaths}"),
                    (prog, f"progression_writes {prog}"),
                    (cap, f"capacity_writes {cap}"),
                ) if cond
            ]
            ghp = [
                (s["frame"], s["bhp"])
                for s in ctl.samples
                if s.get("bhp") is not None
            ]
            payload["gleeok_grade"] = {
                "pass": not misses, "misses": misses,
                "hc": [hc_in, hc_out], "keys": [keys_in, keys_out],
                "bombs": [bombs_in, bombs_out], "magic_key": [mk_in, mk_out],
                "triforce": tf_out, "ghp_samples": ghp[-16:],
                "saw_0x45": ctl.saw_0x45, "saw_0x46": ctl.saw_0x46,
                "settled_xy": [int(settled.link_x), int(settled.link_y)],
                "settled_doors": int(settled.cur_opened_doors),
                "settled_mask": int(settled.open_doorway_mask),
                "room_item": f"0x{settled.room_item_id:02X}",
                "deaths": deaths, "progression_writes": prog,
                "capacity_writes": cap, "hp_poke": False,
            }
            payload["arrival"] = arrival
            payload["final"] = final
            payload["success"] = not misses
            if misses:
                payload["failed"] = "gleeok_grade_miss"
            elif args.save_fixture:
                payload["saved_fixture"] = _save_fixture(
                    raw_env, fixture_name=args.save_fixture, census=final,
                    magic_key=mk_out, keys=keys_out, bombs=bombs_out,
                )
    except ProbeStop as exc:
        payload["success"] = False
        payload.setdefault("failed", str(exc))
        if env is not None:
            payload["final"] = _glance(env)
            env.save_shot("final_halt")
    finally:
        if env is not None:
            tel = assist.telemetry
            payload["frames"] = int(env.frame)
            payload["screenshots"] = env.screenshots
            payload["assist"] = assist.report()
            payload["deaths"] = int(tel.deaths)
            payload["progression_writes"] = int(tel.progression_writes)
            payload["capacity_writes"] = int(tel.capacity_writes)
            payload["runtime_integrity"] = {
                "deaths": int(tel.deaths),
                "progression_writes": int(tel.progression_writes),
                "capacity_writes": int(tel.capacity_writes),
                "hp_poke": False, "position_writes": 0,
            }
        report = write_report("l8_3c_gleeok_fixture", payload, tag=args.tag)
        raw_env.close()

    grade = payload.get("gleeok_grade") or {}
    print(f"report={report}")
    print(f"success={payload.get('success')} failed={payload.get('failed')}")
    print(f"grade={grade}")
    print(f"assist_deaths={payload.get('deaths')}")
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
