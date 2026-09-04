"""Kill the ``0x1E`` body with wooden arrows, then take ONE east gate.

rr-6o7.2, one boundary past ``probe_l8_2e_north.py``.  Starts from the frontier
pin that probe saved (``Level8Interior1EReconFixture``, the settled 0x1E frame
of the same fixture-only chain, ``natural_entry=false``).

PREDICTION (written before the first run, graded in the report):

  ``0x1E`` holds ONE body: RAM shows live type ``0x33`` HP 96 in slot 1 at
  ~(119,112), plus two ``0x55`` statue fireballs and one ``0x56`` residual that
  are projectile state, not room population.  ``dungeon/ids.py`` registers
  ``0x33`` as the L6 "gohma_red" body -- the walkthrough calls this room's boss
  a *blue* Gohma (``0x34``) -- so no colour is asserted; what is claimed is the
  mechanic: it is a Gohma, invulnerable except to an arrow that arrives while
  its eye is open.

  The kill uses the wooden arrows already in the fixture (bow 1, arrows 1,
  rupees 255; each shot spends one rupee).  No arrow poke, no L6 0x1C detour,
  no bombs.  B is switched to arrows through the pause menu, never a RAM write.
  The firing policy is the L6 0x1C one: climb off the south door tile onto the
  ``STAND_Y`` line, lead the body's x, face NORTH, and loose ``UP+B`` on the
  rising edge of an eye-open window (RAM ``0x03C7`` leaving the ``0xC0`` blink).
  ``BLUE_GOHMA_ARROWS_REQUIRED`` = 3 is a walkthrough number; the observed
  connect count and the rupee spend are recorded rather than assumed.

  With the body gone, the hypothesis edge ``blue_gohma -> magic_key_stairs
  RIGHT kill_clear`` says the kill raises the RIGHT bit ``0x01`` and ONE east
  push lands in the column-4 neighbour ``0x1F``:

    expected destination 0x1F, keys 8 -> 8, bombs 6 -> 6.

  Declared contingencies, recorded rather than hidden:

  * if the kill does NOT raise the RIGHT bit, the east gate is not a kill-clear
    shutter -- the hypothesis edge gate is refuted and the probe halts with the
    observed door/mask bytes instead of forcing a way through.
  * if ``0x03C7`` never leaves ``0xC0`` (a different slot or a different anim
    byte in L8), the eye tracker degrades to the L6 ``STUCK_FRAMES`` forced
    shot; every sampled ``0x03C0..0x03CF`` byte is in the report so the real
    eye address can be identified from a red run.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l8_1e_gohma.py \
        --from-state Level8Interior1EReconFixture \
        --tag 20260904_D1 --infinite-life
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name
from zelda_i.dungeon.ops import exit_door, room_fields
from zelda_i.level5.whistle_path import select_b_item_menu
from zelda_i.level6.gohma import (
    ALIGN_TOL,
    ARROW_SPEED,
    EYE_ADDR,
    EYE_EDGE_WINDOW,
    EYE_SHUT,
    FACE_NORTH,
    FIRE_TOL,
    LEAD_CLAMP,
    LINK_X_MAX,
    LINK_X_MIN,
    SHOT_COOLDOWN,
    STAND_Y,
    STAND_Y_TOL,
    STUCK_FRAMES,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOOK,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_COMPASS,
    ADDR_FOOD,
    ADDR_KEYS,
    ADDR_MAGIC_KEY,
    ADDR_MAP,
    ADDR_MAX_BOMBS,
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
GOHMA_ROOM = 0x1E
PREDICTED_EAST_ROOM = 0x1F
POST_L7_TRIFORCE = 0x7F
B_ITEM_ARROWS = 2
# Observed live in 0x1E; ids.py calls 0x33 the L6 red body. Recorded, not named.
GOHMA_BODY_TYPE_0X1E = 0x33
GOHMA_BODY_HP_0X1E = 96
# door_graph.core bit layout: RIGHT 0x01, LEFT 0x02, DOWN 0x04, UP 0x08.
RIGHT_DOOR_BIT = 0x01
EYE_WINDOW = range(0x03C0, 0x03D0)
BODY_GONE_FRAMES = 45


class ProbeStop(RuntimeError):
    """Expected fail-closed halt with a reportable reason."""


def _inventory(env: Any) -> dict[str, int]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    return {
        "sword": int(read_u8(ram, ADDR_SWORD)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "max_bombs": int(read_u8(ram, ADDR_MAX_BOMBS)),
        "bow": int(read_u8(ram, ADDR_BOW)),
        "arrows": int(read_u8(ram, ADDR_ARROWS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "food": int(read_u8(ram, ADDR_FOOD)),
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "rupees": int(read_u8(ram, ADDR_RUPEES)),
        "magic_key": int(read_u8(ram, ADDR_MAGIC_KEY)),
        "map": int(read_u8(ram, ADDR_MAP)),
        "compass": int(read_u8(ram, ADDR_COMPASS)),
        "book": int(read_u8(ram, ADDR_BOOK)),
        "selected_item": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "triforce": int(read_u8(ram, ADDR_TRIFORCE)),
        "health": int(snap.health),
        "heart_containers": int(snap.heart_containers),
    }


def _glance(env: Any) -> dict[str, Any]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    result = leftover_from_snapshot(snap)
    live = [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id not in (0, 0xFF) and obj.hp > 0
    ]
    result.update(
        {
            "level": int(snap.level),
            "screen_hex": f"0x{snap.screen:02X}",
            "xy": [int(snap.link_x), int(snap.link_y)],
            "tile": int(snap.colliding_tile),
            "facing": int(snap.facing),
            "room_item_id": int(snap.room_item_id),
            "room_item_hex": f"0x{snap.room_item_id:02X}",
            "room_obj_count": int(snap.room_obj_count),
            "room_all_dead": int(snap.room_all_dead),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
            "inventory": _inventory(env),
            "live_objects": [
                {
                    "slot": int(obj.slot),
                    "type": int(obj.type_id),
                    "type_hex": f"0x{obj.type_id:02X}",
                    "type_name": object_name(obj.type_id),
                    "xy": [int(obj.x), int(obj.y)],
                    "hp": int(obj.hp),
                }
                for obj in live
            ],
        }
    )
    return result


@dataclass
class ObservedEnv:
    """Delegate an emulator env while collecting transition/stuck evidence."""

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
        self.samples: list[dict[str, Any]] = []
        self.screenshots: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def set_reason(self, phase: str, reason: str) -> None:
        self.phase = phase
        self.reason = reason

    def _sample(self, snap: ZeldaSnapshot, reason: str | None = None) -> None:
        live = [
            obj
            for obj in snap.objects
            if 1 <= obj.slot <= 12 and obj.type_id not in (0, 0xFF) and obj.hp > 0
        ]
        self.samples.append(
            {
                "frame": int(self.frame),
                "phase": self.phase,
                "reason": reason or self.reason,
                "level": int(snap.level),
                "screen": f"0x{snap.screen:02X}",
                "mode": int(snap.mode),
                "xy": [int(snap.link_x), int(snap.link_y)],
                "tile": int(snap.colliding_tile),
                "keys": int(snap.keys),
                "bombs": int(snap.bombs),
                "rupees": int(snap.rupees),
                "doors": int(snap.cur_opened_doors),
                "mask": int(snap.open_doorway_mask),
                "live": len(live),
                "live_types": [f"0x{obj.type_id:02X}" for obj in live],
            }
        )

    def save_shot(self, label: str) -> Path:
        snap = read_snapshot(self.env.get_ram())
        path = RECORDINGS_DIR / (
            f"l8_1e_gohma_{self.tag}_{label}_f{self.frame}_"
            f"L{snap.level}_s{snap.screen:02x}_m{snap.mode}.png"
        )
        save_rgb_png(
            self.latest_obs if self.latest_obs is not None else self.env.render(), path
        )
        self.screenshots.append(str(path))
        return path

    def step(self, action: Any) -> Any:
        result = self.env.step(action)
        self.latest_obs = result[0]
        self.frame += 1
        snap = read_snapshot(self.env.get_ram())
        key = (int(snap.level), int(snap.mode), int(snap.screen))
        xy = (int(snap.link_x), int(snap.link_y))
        if key != self.last_key:
            self._sample(snap, f"transition:{self.last_key}->{key}:{self.reason}")
            self.save_shot("transition")
            self.last_key = key
            self.stuck = 0
        elif xy == self.last_xy:
            self.stuck += 1
            if self.stuck % 250 == 0:
                self._sample(snap, f"stuck_{self.stuck}:{self.reason}")
                self.save_shot(f"stuck_{self.stuck}")
        else:
            self.stuck = 0
        if self.frame % 250 == 0:
            self._sample(snap, f"periodic:{self.reason}")
        self.last_xy = xy
        return result


def _idle(env: ObservedEnv, assist: Any, total: list[int], frames: int, phase: str) -> None:
    env.set_reason(phase, "idle_census")
    for _ in range(frames):
        env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])


def _anim_window(env: Any) -> list[int]:
    ram = env.get_ram()
    return [int(ram[addr]) for addr in EYE_WINDOW]


def _bodies(snap: ZeldaSnapshot) -> list:
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and int(obj.type_id) == GOHMA_BODY_TYPE_0X1E
    ]


def _watch(
    env: ObservedEnv, assist: Any, total: list[int], frames: int
) -> list[dict[str, Any]]:
    """Idle-only observation of the eye window and the body's patrol."""
    out: list[dict[str, Any]] = []
    env.set_reason("watch_0x1e", "idle_watch")
    for i in range(frames):
        snap = read_snapshot(env.get_ram())
        bodies = _bodies(snap)
        body = bodies[0] if bodies else None
        out.append(
            {
                "i": i,
                "anim": _anim_window(env),
                "gx": None if body is None else int(body.x),
                "gy": None if body is None else int(body.y),
                "ghp": None if body is None else int(body.hp),
                "gslot": None if body is None else int(body.slot),
                "gstate": None if body is None else int(body.state),
            }
        )
        env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
    return out


def _fight_gohma(
    env: ObservedEnv,
    assist: Any,
    total: list[int],
    *,
    room: int,
    max_frames: int,
) -> dict[str, Any]:
    """L6 0x1C firing policy, re-aimed at the live 0x1E body.

    Climb off the south door tile, lead the body's x, face NORTH, and loose one
    arrow on the rising edge of each eye-open window.  No poke of any kind: the
    arrows and the bow come from the fixture and every shot spends a rupee.
    """
    eye_open_since = -1
    cooldown = 0
    shots = 0
    last_fire = 0
    gone = 0
    gx_hist: list[int] = []
    samples: list[dict[str, Any]] = []
    notes: list[str] = []
    saw_body = False
    rupees0 = int(read_snapshot(env.get_ram()).rupees)
    for frame in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            raise ProbeStop("fight_0x1e:death")
        if snap.level != LEVEL8:
            raise ProbeStop(f"fight_0x1e:left_level8:L{snap.level}")
        if snap.screen != room and snap.mode == PLAY_MODE:
            raise ProbeStop(f"fight_0x1e:left_room:0x{snap.screen:02X}")
        if snap.mode != PLAY_MODE:
            env.set_reason("fight_0x1e", f"wait_mode_{snap.mode}")
            env.step(nes_idle_action())
            total[0] += 1
            assist.apply_env(env, frame=total[0])
            continue

        eye_byte = int(env.get_ram()[EYE_ADDR])
        if eye_byte == EYE_SHUT:
            eye_open_since = -1
        elif eye_open_since < 0:
            eye_open_since = 0
        else:
            eye_open_since = min(9999, eye_open_since + 1)

        bodies = _bodies(snap)
        if bodies:
            saw_body = True
            gone = 0
        elif saw_body:
            gone += 1
            if gone >= BODY_GONE_FRAMES:
                return {
                    "ok": True,
                    "frames": frame,
                    "shots": shots,
                    "rupees_before": rupees0,
                    "rupees_after": int(snap.rupees),
                    "notes": notes,
                    "samples": samples,
                }

        if cooldown > 0:
            cooldown -= 1

        if not bodies:
            env.set_reason("fight_0x1e", "wait_body_gone_settle")
            env.step(nes_idle_action())
            total[0] += 1
            assist.apply_env(env, frame=total[0])
            continue

        body = bodies[0]
        gx = int(body.x)
        gx_hist.append(gx)
        del gx_hist[:-8]
        gvx = (
            (gx_hist[-1] - gx_hist[0]) / (len(gx_hist) - 1)
            if len(gx_hist) >= 4
            else 0.0
        )
        ly = int(snap.link_y)

        if frame % 8 == 0 or cooldown == SHOT_COOLDOWN:
            samples.append(
                {
                    "frame": frame,
                    "x": int(snap.link_x),
                    "y": ly,
                    "gx": gx,
                    "gy": int(body.y),
                    "ghp": int(body.hp),
                    "gslot": int(body.slot),
                    "eye": eye_byte,
                    "eye_since": eye_open_since,
                    "shots": shots,
                    "rupees": int(snap.rupees),
                }
            )

        if ly > STAND_Y + STAND_Y_TOL:
            env.set_reason("fight_0x1e", "climb")
            env.step(nes_action("UP"))
            total[0] += 1
            assist.apply_env(env, frame=total[0])
            continue

        flight = max(1.0, (ly - int(body.y)) / ARROW_SPEED)
        lead = int(round(gvx * flight))
        lead = max(-LEAD_CLAMP, min(LEAD_CLAMP, lead))
        target_x = max(LINK_X_MIN, min(LINK_X_MAX, gx + lead))
        dx = target_x - int(snap.link_x)

        if int(snap.rupees) <= 0:
            raise ProbeStop("fight_0x1e:out_of_ammo_rupees")
        forced = frame - last_fire >= STUCK_FRAMES
        fresh = 0 <= eye_open_since <= EYE_EDGE_WINDOW
        if cooldown <= 0 and (fresh or forced) and abs(dx) <= FIRE_TOL:
            if int(snap.facing) != FACE_NORTH:
                env.set_reason("fight_0x1e", "face_up")
                env.step(nes_action("UP"))
            else:
                cooldown = SHOT_COOLDOWN
                last_fire = frame
                shots += 1
                if forced and not fresh:
                    notes.append(f"forced_shot_{shots}_at_{frame}")
                env.set_reason("fight_0x1e", "arrow_shot")
                env.step(nes_action("UP", "B"))
            total[0] += 1
            assist.apply_env(env, frame=total[0])
            continue

        if abs(dx) > ALIGN_TOL:
            env.set_reason("fight_0x1e", "strafe")
            env.step(nes_action("RIGHT" if dx > 0 else "LEFT"))
        elif ly < STAND_Y - STAND_Y_TOL:
            env.set_reason("fight_0x1e", "settle")
            env.step(nes_action("DOWN"))
        else:
            env.set_reason("fight_0x1e", "eye_wait")
            env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])

    return {
        "ok": False,
        "error": "timeout",
        "frames": max_frames,
        "shots": shots,
        "rupees_before": rupees0,
        "rupees_after": int(read_snapshot(env.get_ram()).rupees),
        "notes": notes,
        "samples": samples,
    }


def _save_fixture(
    raw_env: Any,
    *,
    fixture_name: str,
    gate: str,
    shots: int,
    rupees_before: int,
    rupees_after: int,
    keys: int,
    bombs: int,
    census: dict[str, Any],
) -> dict[str, Any]:
    """Save the settled-0x1F state + an auditable provenance sidecar.

    Disclosed writes: NONE.  The only inventory delta from
    ``Level8Interior1EReconFixture`` is the natural rupee spend of the arrows
    loosed at the 0x1E body plus the pause-menu B selection.
    """
    from retro_harness.env import save_state, state_path
    from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance

    after = read_snapshot(raw_env.get_ram())
    if not (
        after.level == LEVEL8
        and after.mode == PLAY_MODE
        and after.screen == PREDICTED_EAST_ROOM
    ):
        raise ProbeStop("save_fixture_pin_mismatch")
    path = save_state(raw_env, GAME_DIR, GAME, fixture_name)
    source_path = state_path(GAME_DIR, GAME, "Level8Interior1EReconFixture")
    result = {
        "ok": True,
        "source_state": "Level8Interior1EReconFixture",
        "fixture_state": fixture_name,
        "state": compact_snapshot(after),
        "census": census,
        "gate": gate,
        "natural_arrow_spend": {
            "shots": shots,
            "rupees_before": rupees_before,
            "rupees_after": rupees_after,
        },
        "keys_unchanged": keys,
        "bombs_unchanged": bombs,
        "fixture_writes": [],
    }
    write_state_provenance(
        path,
        source_state_path=source_path if source_path.exists() else None,
        request={
            "bead": "rr-6o7.2",
            "phase": "level8_interior_0x1f_recon",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "natural_entry": False,
            "fixture_writes": [],
            "notes": [
                "Continuation pin: Level8Interior1EReconFixture is the settled"
                " 0x1E frame of the probe_l8_2e_north replay, itself a"
                " fixture-only chain from Level8InteriorReconFixture.",
                "Kill the one 0x1E body with the fixture's own wooden arrows"
                " (bow 1, arrows 1) on the eye-open edge, then ONE east gate"
                " 0x1E -> 0x1F.",
                "No RAM poke: B was switched to arrows through the pause menu"
                f" and each of the {shots} shots spent one rupee"
                f" ({rupees_before} -> {rupees_after}).",
                f"0x1E east gate observed as: {gate}.",
                "Not route eligible; not on L8_THROUGH.",
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
        parser,
        default_state="Level8Interior1EReconFixture",
        default_tag="20260904_D1",
    )
    parser.add_argument("--save-fixture", default=None)
    parser.add_argument(
        "--watch-frames",
        type=int,
        default=0,
        help="idle-only eye/patrol observation before the fight (recon runs)",
    )
    args = parser.parse_args()
    if not args.infinite_life:
        raise SystemExit("this fixture route probe requires --infinite-life")

    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    assist = make_assist(True)
    raw_env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    payload: dict[str, Any] = {
        "trial": "kill_0x1e_body_with_arrows_then_one_east_gate_attempt",
        "from_state": args.from_state,
        "fixture_only": True,
        "natural_entry": False,
        "route_eligible": False,
        "infinite_life": True,
        "watch_frames": int(args.watch_frames),
        "prediction": {
            "written_before_run": True,
            "claim": (
                "kill the one live 0x33 HP96 body in 0x1E with the fixture's"
                " own wooden arrows on the eye-open edge, then ONE east"
                " kill-clear shutter -> first settled L8 play room 0x1F"
            ),
            "source_room": "0x1E",
            "direction": "RIGHT",
            "expected_destination": "0x1F",
            "expected_gate": "east_kill_clear_shutter",
            "expected_keys": [8, 8],
            "expected_bombs": [6, 6],
            "hypothesis_edge": "blue_gohma -> magic_key_stairs RIGHT kill_clear",
            "observed_body": {
                "type_hex": f"0x{GOHMA_BODY_TYPE_0X1E:02X}",
                "hp": GOHMA_BODY_HP_0X1E,
                "ids_label": "gohma_red (L6 0x1C body); no colour asserted here",
                "walkthrough_claim": "blue Gohma, three arrows",
            },
            "contingency": (
                "if the kill does not raise open_doorway_mask RIGHT bit 0x01 the"
                " east gate is not a kill-clear shutter: halt with the observed"
                " door bytes, do not force a way through. If 0x03C7 never leaves"
                " 0xC0 the tracker degrades to the L6 forced shot and the"
                " sampled 0x03C0..0x03CF window identifies the real eye byte."
            ),
            "attempt_history": [
                {
                    "tag": "20260904_D1",
                    "result": "east_gate_0x1e_not_a_kill_clear_shutter",
                    "detail": (
                        "kill ok (type 0x33 HP 96→64→32→0, 3 connects / 9"
                        " shots, rupees 255→246, body gone, PNG east black);"
                        " open_doorway_mask stayed 0x04 so the mask&RIGHT"
                        " check halted. cur_opened_doors rose 0x04→0x0D"
                        " (RIGHT bit 0x01 set). keys 8, bombs 6, deaths 0."
                    ),
                    "fix": (
                        "gate the east push on cur_opened_doors RIGHT rising"
                        " (0x04→0x0D), not open_doorway_mask; mask is not"
                        " the shutter stop (L6 post-Gleeok / L9 Patra)."
                    ),
                }
            ],
        },
        "runtime_controller_writes": {
            "room": 0,
            "door": 0,
            "position": 0,
            "inventory": 0,
            "triforce": 0,
            "magic_key": 0,
            "capacity": 0,
        },
    }
    env: ObservedEnv | None = None
    total = [0]
    try:
        obs, _ = reset_obs(raw_env)
        env = ObservedEnv(raw_env, assist, args.tag)
        env.latest_obs = obs
        start = _glance(env)
        payload["start"] = start
        env._sample(read_snapshot(env.get_ram()), "fixture_start")
        env.save_shot("start")
        inv0 = start["inventory"]
        bodies0 = [
            obj
            for obj in start["live_objects"]
            if obj["type"] == GOHMA_BODY_TYPE_0X1E
        ]
        if not (
            start["level"] == LEVEL8
            and start["screen"] == GOHMA_ROOM
            and start["mode"] == PLAY_MODE
            and inv0["triforce"] == POST_L7_TRIFORCE
            and inv0["sword"] == 3
            and inv0["keys"] == 8
            and inv0["bombs"] == 6
            and inv0["bow"] == 1
            and inv0["arrows"] == 1
            and inv0["magic_key"] == 0
            and len(bodies0) == 1
            and bodies0[0]["hp"] == GOHMA_BODY_HP_0X1E
        ):
            raise ProbeStop("fixture_start_mismatch")

        assist.apply_env(env, frame=0)
        keys_in = int(inv0["keys"])
        bombs_in = int(inv0["bombs"])

        if args.watch_frames:
            payload["watch"] = _watch(env, assist, total, args.watch_frames)

        before_menu = _inventory(env)
        env.set_reason("select_arrows", "pause_menu_input")
        menu = select_b_item_menu(env, assist, total, B_ITEM_ARROWS)
        after_menu = _inventory(env)
        payload["pause_menu_selection"] = {
            "result": menu,
            "before": before_menu,
            "after": after_menu,
            "ram_selection_write": False,
        }
        if after_menu["selected_item"] != B_ITEM_ARROWS:
            raise ProbeStop("pause_menu_failed_to_select_arrows")

        fight = _fight_gohma(
            env, assist, total, room=GOHMA_ROOM, max_frames=6000
        )
        payload["fight_0x1e"] = fight
        env.save_shot("after_fight_0x1e")
        if not fight.get("ok"):
            payload["failed"] = "fight_0x1e_timeout"
            raise ProbeStop("fight_0x1e_timeout")
        _idle(env, assist, total, 60, "post_kill_0x1e_census")
        post_kill = _glance(env)
        payload["post_kill_0x1e"] = post_kill
        env.save_shot("cleared_0x1e")

        # D1 (20260904): kill dropped HP 96→64→32→0 in three connects (9
        # wooden shots, rupees 255→246) and PNG east went black, but
        # open_doorway_mask stayed 0x04. cur_opened_doors rose 0x04→0x0D
        # (RIGHT+DOWN+UP). Same byte as L6 post-Gleeok / L9 Patra: mask is
        # not the shutter stop. Gate on the doors byte that actually rose.
        doors0 = int(start["cur_opened_doors"])
        doors1 = int(post_kill["cur_opened_doors"])
        east_open = bool(doors1 & RIGHT_DOOR_BIT) and not bool(
            doors0 & RIGHT_DOOR_BIT
        )
        payload["east_gate_0x1e"] = {
            "east_open_after_kill": east_open,
            "cur_opened_doors_before": doors0,
            "cur_opened_doors": doors1,
            "open_doorway_mask": post_kill["open_doorway_mask"],
            "room_all_dead": post_kill["room_all_dead"],
            "keys_in": keys_in,
            "bombs_in": bombs_in,
        }
        if not east_open:
            payload["failed"] = "east_gate_0x1e_not_a_kill_clear_shutter"
            raise ProbeStop("east_gate_0x1e_not_a_kill_clear_shutter")

        env.set_reason("east_gate_0x1e", "exit_door_right")
        door = exit_door(env, assist, total, "RIGHT", push=180)
        payload["east_gate_0x1e"]["door"] = door
        settled = read_snapshot(env.get_ram())
        env._sample(settled, "first_settled_after_east_gate")
        env.save_shot("first_settled_after_east_gate")
        _idle(env, assist, total, 150, "east_gate_dest_idle_census")
        final = _glance(env)
        settled = read_snapshot(env.get_ram())
        env.save_shot("final_census")
        keys_out = int(settled.keys)
        bombs_out = int(settled.bombs)
        payload["east_gate_0x1e"].update(
            {
                "observed_gate": "east_kill_clear_shutter",
                "keys_out": keys_out,
                "bombs_out": bombs_out,
                "settled_room": f"0x{settled.screen:02X}",
                "settled_xy": [int(settled.link_x), int(settled.link_y)],
                "settled_mode": int(settled.mode),
                "settled_level": int(settled.level),
                "census": final,
                "room_fields": room_fields(settled, env.get_ram()),
            }
        )
        misses: list[str] = []
        if not (settled.level == LEVEL8 and settled.mode == PLAY_MODE):
            misses.append(f"settled L{settled.level} mode={settled.mode}")
        if settled.screen == GOHMA_ROOM:
            misses.append("east gate did not open (still 0x1E)")
        elif settled.screen != PREDICTED_EAST_ROOM:
            misses.append(f"settled 0x{settled.screen:02X} != predicted 0x1F")
        if keys_out != keys_in:
            misses.append(f"keys changed {keys_in}->{keys_out}")
        if bombs_out != bombs_in:
            misses.append(f"bombs changed {bombs_in}->{bombs_out}")
        payload["east_gate_0x1e"]["grade"] = {
            "pass": not misses,
            "misses": misses,
            "shots_loosed": int(fight.get("shots", 0)),
            "walkthrough_three_arrows_matched": int(fight.get("shots", 0)) == 3,
        }
        payload["final"] = final
        payload["success"] = not misses
        if misses:
            payload["failed"] = "east_gate_0x1e_miss"
        elif args.save_fixture:
            payload["saved_fixture"] = _save_fixture(
                raw_env,
                fixture_name=args.save_fixture,
                gate="east_kill_clear_shutter",
                shots=int(fight.get("shots", 0)),
                rupees_before=int(fight.get("rupees_before", 0)),
                rupees_after=int(fight.get("rupees_after", 0)),
                keys=keys_out,
                bombs=bombs_out,
                census=final,
            )
    except ProbeStop as exc:
        payload["success"] = False
        payload.setdefault("failed", str(exc))
        if env is not None:
            payload["final"] = _glance(env)
            env._sample(read_snapshot(env.get_ram()), f"halt:{exc}")
            env.save_shot("final_halt")
    finally:
        if env is not None:
            payload["frames"] = int(env.frame)
            payload["samples"] = env.samples[-96:]
            payload["screenshots"] = env.screenshots
            payload["assist"] = assist.report()
            payload["deaths"] = int(assist.telemetry.deaths)
            payload["progression_writes"] = int(assist.telemetry.progression_writes)
            payload["capacity_writes"] = int(assist.telemetry.capacity_writes)
            gate_evidence = payload.get("east_gate_0x1e", {})
            keys_out = gate_evidence.get("keys_out")
            bombs_out = gate_evidence.get("bombs_out")
            payload["runtime_integrity"] = {
                "deaths": int(assist.telemetry.deaths),
                "progression_writes": int(assist.telemetry.progression_writes),
                "capacity_writes": int(assist.telemetry.capacity_writes),
                "direct_ram_writes": 0,
                "state_loads_after_start": 0,
                "arrow_poke": False,
                "b_slot_poke": False,
                "keys_spent_at_new_boundary": (
                    None
                    if keys_out is None
                    else int(gate_evidence.get("keys_in", 0)) - int(keys_out)
                ),
                "bombs_spent_at_new_boundary": (
                    None
                    if bombs_out is None
                    else int(gate_evidence.get("bombs_in", 0)) - int(bombs_out)
                ),
            }
        report = write_report("l8_1e_gohma_fixture", payload, tag=args.tag)
        raw_env.close()

    print(f"report={report}")
    print(f"success={payload.get('success')} failed={payload.get('failed')}")
    print(f"fight={ {k: v for k, v in payload.get('fight_0x1e', {}).items() if k != 'samples'} }")
    print(f"east_gate={payload.get('east_gate_0x1e', {}).get('grade')}")
    print(f"final={payload.get('final')}")
    print(f"assist_deaths={payload.get('deaths')}")
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
