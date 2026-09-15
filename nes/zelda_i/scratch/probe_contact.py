"""Ring-buffer the frames before every pre-L1 contact: who actually touched Link.

``probe_kill_streak.py`` logs the frame a streak dies, which is one frame too
late -- knockback ($00D3=32) has already moved Link away from whatever hit him.
This keeps the last ``WINDOW`` play frames and dumps them when $04F0 arms, so
the collider is named rather than guessed.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_contact.py --tag contact1
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

from retro_harness.audit import AuditCapabilities, AuditedEnv
from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.combat import chebyshev
from zelda_i.dungeon.ids import OBJECT_NAMES
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.overworld.path import OverworldPathController
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.spine.survival import run_survival_spine, spine_final_fields

OUT_DIR = Path(__file__).resolve().parent
WINDOW = 48


def _live(snap) -> list[dict]:
    out = []
    for obj in snap.objects:
        if int(obj.slot) < 1 or int(obj.type_id) in (0, 0xFF, 0x64):
            continue
        # Everything, hp 0 included: an octorok rock is a slot with no hp,
        # and the first pass filtered exactly the thing that was hitting Link.
        if int(obj.hp) >= 200:
            continue
        out.append(
            {
                "slot": int(obj.slot),
                "t": int(obj.type_id),
                "n": OBJECT_NAMES.get(int(obj.type_id), f"unk_{obj.type_id:#04x}"),
                "hp": int(obj.hp),
                "x": int(obj.x),
                "y": int(obj.y),
                "d": chebyshev(int(snap.link_x), int(snap.link_y), int(obj.x), int(obj.y)),
            }
        )
    return sorted(out, key=lambda o: o["d"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="contact")
    parser.add_argument("--shield", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--off-line", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--min-hearts", type=int, default=None)
    parser.add_argument("--screen-budget", type=int, default=None)
    args = parser.parse_args(argv)
    configure_headless()

    # Ablation knobs. ``path.OverworldPathController`` builds its own
    # ``ScreenHunter()``, so the only seam a probe has is the field default.
    import dataclasses
    from zelda_i.overworld import hunt as hunt_mod

    overrides = {
        "shield": args.shield,
        "avoid_firing_lines": args.off_line,
        "min_hearts": args.min_hearts,
        "screen_max_frames": args.screen_budget,
    }
    for name, value in overrides.items():
        if value is None:
            continue
        for f in dataclasses.fields(hunt_mod.ScreenHunter):
            if f.name == name:
                f.default = value
        setattr(hunt_mod.ScreenHunter, name, value)
    print("overrides:", {k: v for k, v in overrides.items() if v is not None})

    # Which rule owned the frame. ``_do_hop`` runs four position rules
    # ahead of ``_hunt_action`` (occupancy align, stall escape, unstick
    # wiggle, recover_off_edge), and a contact window that cannot name the
    # driver cannot tell "the hunt swung and missed" from "the hunt never
    # got the frame".
    driver = {"reason": None, "btn": "-"}
    _orig_step = OverworldPathController.step

    def _step(self, snap):  # type: ignore[no-untyped-def]
        act = _orig_step(self, snap)
        driver["reason"] = getattr(act, "reason", None)
        raw = list(getattr(act, "action", ()) or ())
        # The NES action is 9 wide with an unused hole at index 1, so
        # ``enumerate(NES_BUTTON_NAMES)`` is off by one from index 2 up and
        # drops A (index 8) entirely — which is the one button that matters
        # here. Decode through the name->index map.
        driver["btn"] = "+".join(
            name
            for name, i in NES_BUTTON_NAME_TO_INDEX.items()
            if i is not None and i < len(raw) and raw[i]
        ) or "-"
        return act

    OverworldPathController.step = _step  # type: ignore[assignment]

    ring: deque[dict] = deque(maxlen=WINDOW)
    contacts: list[dict] = []
    prev = {"iframes": -1, "world": -1}

    def on_frame(env, _obs, _action, frame: int) -> None:
        snap = read_snapshot(env.get_ram())
        if int(snap.level) != 0 or int(snap.mode) != PLAY_MODE:
            return
        rec = {
            "f": frame,
            "scr": int(snap.screen),
            "xy": [int(snap.link_x), int(snap.link_y)],
            "face": int(snap.facing),
            "if": int(snap.link_iframes),
            "w": int(snap.world_kill_count),
            "hp": int(snap.health),
            "part": int(snap.heart_partial),
            "R": int(snap.rupees),
            "why": driver["reason"],
            "btn": driver["btn"],
            # Link's own ``$00AC`` (``link_busy``) plus every live slot with
            # its state, because the thing that stops a second swing may be
            # the sword object, not Link.
            "st": int(snap.objects[0].state) if snap.objects else -1,
            "slots": [
                [int(o.slot), int(o.type_id), int(o.state), int(o.hp)]
                for o in snap.objects
                if int(o.slot) >= 1 and int(o.type_id) not in (0, 0xFF)
            ],
            "objs": _live(snap),
        }
        iframes = int(snap.link_iframes)
        if prev["iframes"] == 0 and iframes > 0:
            contacts.append(
                {
                    "at": frame,
                    "reset": int(snap.world_kill_count) == 0 and prev["world"] > 0,
                    "streak_lost": prev["world"],
                    "window": list(ring) + [rec],
                }
            )
        ring.append(rec)
        prev["iframes"] = iframes
        prev["world"] = int(snap.world_kill_count)

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    payload = None
    try:
        obs, _ = reset_obs(env)
        env = AuditedEnv(env, capabilities=AuditCapabilities.all("zelda_i.contact"))
        run = run_survival_spine(
            env, obs, assist=None, on_frame=on_frame, through="pre-l1", allow_pokes=False
        )
        ram = env.get_ram()
        snap = read_snapshot(ram)
        hunt: dict = {}
        stages: list[dict] = []
        for stage in run.report().get("stages", []):
            ctl = stage.get("controller") or {}
            nested = ctl.get("hunt") if isinstance(ctl.get("hunt"), dict) else {}
            if "streak_resets" in ctl:
                hunt = ctl
            elif "streak_resets" in nested:
                hunt = nested
            # Which rule owned the frames. A hunt that never drove is not a
            # hunt that lost the fight; ``reason_counts`` separates the two.
            stages.append(
                {
                    "name": stage.get("name"),
                    "frames": stage.get("frames"),
                    "reason_counts": ctl.get("reason_counts"),
                    "evade_reasons": ctl.get("evade_reasons"),
                    "evades": ctl.get("evades"),
                    "stall_escapes": ctl.get("stall_escapes"),
                    "hunt": nested or None,
                }
            )
        payload = {
            "ok": run.report().get("ok"),
            "failed": run.report().get("failed_stage"),
            "final": spine_final_fields(snap, ram),
            "hunt": {k: hunt.get(k) for k in (
                "kills", "kills_counter", "rupees_banked", "damage_taken",
                "hurt_events", "streak_best", "streak_resets",
                "kills_by_screen", "drops_by_state", "peak_live_by_screen", "peak_prey_by_screen",
                "shield_frames", "guard_frames", "peel_frames",
                "off_line_frames", "hunt_frames_by_screen", "notes",
            )},
            "contacts": contacts,
            "stages": stages,
        }
    finally:
        env.close()

    out = OUT_DIR / f"{args.tag}.json"
    write_json_report(out, payload)
    write_json_report(RECORDINGS_DIR / f"{args.tag}.json", payload)
    h = payload["hunt"]
    print(
        f"ok={payload['ok']} R={payload['final']['rupees']} "
        f"kills={h.get('kills')}/{h.get('kills_counter')} "
        f"best={h.get('streak_best')} resets={h.get('streak_resets')} "
        f"contacts={len(contacts)}\n"
        f"  drops={h.get('drops_by_state')} kills_by={h.get('kills_by_screen')}\n"
        f"  live={h.get('peak_live_by_screen')}\n"
        f"  prey={h.get('peak_prey_by_screen')}\n"
        f"  shield={h.get('shield_frames')} guard={h.get('guard_frames')} "
        f"peel={h.get('peel_frames')} offline={h.get('off_line_frames')}\n"
        f"  notes={h.get('notes')}"
    )
    for c in contacts:
        last = c["window"][-1]
        near = last["objs"][0] if last["objs"] else None
        # The collider is the body that was closest one frame BEFORE knockback.
        pre = c["window"][-2] if len(c["window"]) > 1 else last
        prenear = pre["objs"][0] if pre["objs"] else None
        print(
            f"  CONTACT f={c['at']} scr=0x{last['scr']:02x} reset={c['reset']} "
            f"lost={c['streak_lost']} xy={last['xy']} "
            f"pre_xy={pre['xy']} "
            f"pre_near={prenear and (prenear['n'], prenear['slot'], prenear['x'], prenear['y'], prenear['d'])} "
            f"post_near={near and (near['n'], near['slot'], near['d'])}"
        )
    for st in payload["stages"]:
        rc = st.get("reason_counts") or {}
        top = sorted(rc.items(), key=lambda kv: -kv[1])[:12]
        print(f"  STAGE {st['name']} {st['frames']}f evades={st.get('evades')}")
        print(f"    reasons: {top}")
        if st.get("evade_reasons"):
            print(f"    evade:   {st['evade_reasons']}")
    print(f"wrote {out}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
