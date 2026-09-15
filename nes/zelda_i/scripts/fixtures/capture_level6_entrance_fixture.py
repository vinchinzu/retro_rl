"""rr-d6v: capture a *coherent* Clean L6 entrance pin from a live power-on run.

The shipped ``Level6Entrance`` pin is invalid: ``$066F = 0x2F`` is 15 whole
hearts in 3 containers, a byte normal play cannot reach (``ram.full_health_byte``
exists precisely to forbid it), with ``triforce=0x00`` and no L5 inventory. Every
Clean L6 heart number to date was measured against that ~7x inflated budget --
see ``docs/tasks/rr-d6v-residual.md``.

This script does not repair the byte. It runs the continuous Survival spine from
power-on through ``level6-entry`` in one emulator session (no ``set_state``, no
mid-run state load) and saves whatever Link actually holds when he steps into
``0x79``. The pin is therefore a *measured arrival*, not a hand-written one.

Two caveats travel with the result, and both are written into the sidecar:

* Survival refills health, so the **filled** nibble is the assist's write
  (``health_byte_for_containers``, always ``n<<4|n``). Containers are real --
  the assist never grants one and clamps any it did not see granted.
* Survival pokes owned inventory counts (bombs/keys/rupees). Pass ``--no-pokes``
  for a pin whose consumables are natural; the run may not reach ``0x79``.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scripts/fixtures/capture_level6_entrance_fixture.py

``route_eligible: false``. This is a development checkpoint, not a STATUS claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from retro_harness.audit import AuditCapabilities, AuditedEnv
from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.spine.survival import run_survival_spine, spine_final_fields

BEAD = "rr-d6v"
FIXTURE_NAME = "Level6EntranceNatural"
THROUGH = "level6-entry"
LEVEL6 = 6
ROOM_79 = 0x79
PLAY_MODE = 5


def _coherent(health: int) -> bool:
    """``$066F`` is ``hi = containers-1, lo = whole hearts``; ``lo <= hi`` always."""
    return (health & 0x0F) <= (health >> 4)


def _pin_glance(snap: Any) -> dict[str, Any]:
    health = int(snap.health)
    return {
        "level": int(snap.level),
        "room": int(snap.screen),
        "room_hex": f"0x{int(snap.screen):02x}",
        "mode": int(snap.mode),
        "xy": [int(snap.link_x), int(snap.link_y)],
        "health_hex": f"0x{health:02x}",
        "hearts_lo": health & 0x0F,
        "containers": (health >> 4) + 1,
        "health_coherent": _coherent(health),
        "health_full": bool(snap.health_is_full),
        "heart_partial": int(snap.heart_partial),
        "triforce": int(snap.triforce),
        "triforce_hex": f"0x{int(snap.triforce):02x}",
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "rupees": int(snap.rupees),
        "bow": int(snap.bow),
        "arrows": int(snap.arrows),
        "raft": int(snap.raft),
        "ladder": int(snap.ladder),
        "rod": int(snap.rod),
    }


def _at_entrance(glance: dict[str, Any]) -> bool:
    return (
        glance["level"] == LEVEL6
        and glance["room"] == ROOM_79
        and glance["mode"] == PLAY_MODE
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=FIXTURE_NAME)
    parser.add_argument("--tag", default="l6_entrance_natural_pin")
    parser.add_argument(
        "--no-pokes",
        action="store_true",
        help="Skip Survival inventory count top-ups (the run may not arrive).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Save even when the arrival pose or health byte fails its check.",
    )
    args = parser.parse_args(argv)

    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    env = AuditedEnv(
        env, capabilities=AuditCapabilities.all("zelda_i.l6_entrance_pin")
    )
    assist = UnlimitedHealthAssist(enabled=True)
    try:
        run = run_survival_spine(
            env,
            obs,
            assist=assist,
            through=THROUGH,
            allow_pokes=not args.no_pokes,
        )
        run.apply_state_audit(int(env.audit().mid_run_loads or 0))
        ram = env.get_ram()
        snap = read_snapshot(ram)
        glance = _pin_glance(snap)
        report = run.report()
        ok = (
            bool(run.success)
            and _at_entrance(glance)
            and glance["health_coherent"]
            and not report.get("mid_run_state_load")
        )
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        png = RECORDINGS_DIR / f"{args.tag}_final.png"
        save_rgb_png(run.obs, png)

        if not ok and not args.force:
            print(json.dumps({"ok": False, "glance": glance, "run": report}, indent=2))
            return 1

        path = Path(save_state(env, GAME_DIR, GAME, args.name))
        write_state_provenance(
            path,
            source_state_path=None,
            request={
                "bead": BEAD,
                "track": "survival_arrival_pin",
                "phase": "l6_entrance_natural",
                "route_eligible": False,
                "status_claim": False,
                "fixture_only": True,
                "through": THROUGH,
                "allow_pokes": not args.no_pokes,
                "notes": (
                    "Continuous power-on Survival session, no state load. "
                    "Replaces the incoherent Level6Entrance pin "
                    "($066F=0x2F: 15 hearts in 3 containers, TF 0x00, no L5 "
                    "inventory). Containers here are real -- the health assist "
                    "never grants one and reports container_clamps. The FILLED "
                    "nibble is the assist's refill, so this pin is a full-health "
                    "arrival, an upper bound on a Clean budget, not a measured "
                    "Clean one. Inventory counts carry Survival pokes unless "
                    "--no-pokes."
                ),
            },
            selected_trial={
                "ok": ok,
                "glance": glance,
                "compact": compact_snapshot(snap),
                "final": spine_final_fields(snap, ram),
                "run": report,
                "assist": assist.report(),
                "png": str(png.resolve()),
                "state": str(path.resolve()),
            },
            natural_entry=True,
        )
        print(json.dumps({"ok": ok, "state": str(path), "glance": glance}, indent=2))
        return 0 if ok else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
