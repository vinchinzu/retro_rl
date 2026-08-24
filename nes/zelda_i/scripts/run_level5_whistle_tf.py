"""Thin CLI: Whistle cellar/floor → Digdogger → L5 TF 0x10.

Path lives in ``zelda_i.level5_boss_path``. Survival / infinite-life.
No key/door/item pokes. Not a Clean STATUS claim.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.level5_boss_path import TF_SUFFIX_STOPS, run_level5_tf_suffix
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, read_snapshot, read_u8

START = "Level5WhistleFrom77"


def _inv(env) -> dict:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    tf = int(read_u8(ram, ADDR_TRIFORCE))
    return {
        "room": f"0x{snap.screen:02x}",
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "tf": tf,
        "tf_l5": bool(tf & 0x10),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "health": int(snap.health),
        "item": int(snap.room_item_id),
        "doors": int(snap.cur_opened_doors),
    }


def run_once(start_state: str, tag: str, stop_at: str) -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total := [1], 16)
        start = _inv(env)
        print("START", start, flush=True)
        if start["whistle"] != 1:
            body = {
                "ok": False,
                "reason": "whistle_not_1",
                "start": start,
                "start_state": start_state,
            }
            write_json_report(RECORDINGS_DIR / f"{tag}.json", body)
            return body
        ok, frames, detail = run_level5_tf_suffix(
            env, assist=assist, frame_base=total[0], stop_at=stop_at
        )
        png = RECORDINGS_DIR / f"{tag}_final.png"
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, png)
        end = _inv(env)
        body = {
            "ok": bool(ok),
            "status_claim": None,
            "pokes": False,
            "track": "assisted",
            "start_state": start_state,
            "stop_at": stop_at,
            "frames": frames,
            "start": start,
            "hops": detail.get("hops"),
            "failed": detail.get("failed"),
            "digdogger": detail.get("digdogger"),
            "final": end,
            "screenshot": str(png.resolve()),
        }
        write_json_report(RECORDINGS_DIR / f"{tag}.json", body)
        print("OK", body["ok"], "FAILED", detail.get("failed"), "END", end, flush=True)
        return body
    finally:
        env.close()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--from-state", default=START)
    ap.add_argument("--tag", default="l5_whistle_tf")
    ap.add_argument("--stop", dest="stop_at", choices=TF_SUFFIX_STOPS, default="triforce")
    args = ap.parse_args()
    r = run_once(args.from_state, args.tag, args.stop_at)
    print("RESULT_OK", r.get("ok"))
    print("HOPS", [(h.get("hop"), h.get("ok") or h.get("success"), h.get("dest")) for h in r.get("hops") or []])


if __name__ == "__main__":
    main()
