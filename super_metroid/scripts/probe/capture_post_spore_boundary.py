"""Capture the natural post-Spore boundary for suffix development.

The resulting save state is a local development checkpoint.  It is produced
by replaying the accepted power-on prefix and is never itself acceptance
evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, write_state_bytes  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.paths import GAME, GAME_DIR  # noqa: E402
from super_metroid.progression import START_TO_SPORE_SPAWN_GRAPH  # noqa: E402
from super_metroid.ram import GameplayPhase  # noqa: E402
from super_metroid.routes.spore_spawn_controller import (  # noqa: E402
    play_post_torizo_to_spore_spawn,
)
from super_metroid.routes.continuous import (  # noqa: E402
    _EarlySession,
    _sha256,
    play_start_to_bombs,
)


STATE_PATH = (
    GAME_DIR
    / "custom_integrations"
    / GAME
    / "natural_post_spore_spawn.state"
)


def main() -> None:
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedResourcesAssist()
    try:
        env.reset()
        session = _EarlySession(
            env,
            writer=None,
            assist=assist,
            graph=START_TO_SPORE_SPAWN_GRAPH,
        )
        splits = []
        segments = []
        play_start_to_bombs(session, splits, segments)
        boss = play_post_torizo_to_spore_spawn(session)
        state = session.state
        if (
            state.room_id != 0x9B5B
            or state.phase is not GameplayPhase.ORDINARY_GAMEPLAY
            or state.max_super_missiles != 0
            or state.max_power_bombs != 0
        ):
            raise RuntimeError(f"unexpected post-Spore boundary: {state}")
        write_state_bytes(STATE_PATH, env.em.get_state())
    finally:
        env.close()

    provenance = {
        "schemaVersion": 1,
        "statePath": str(STATE_PATH.resolve()),
        "stateSha256": _sha256(STATE_PATH),
        "source": "continuous power-on replay through natural Spore Spawn exit",
        "sourceFrame": session.frame,
        "sourceState": state.to_dict(),
        "sporeSpawn": boss.to_dict(),
        "stateLoadsDuringSourceReplay": 0,
        "progressionWrites": assist.telemetry.progression_writes,
        "capacityWrites": assist.telemetry.capacity_writes,
        "developmentOnly": True,
        "acceptanceWarning": (
            "This checkpoint accelerates suffix development. Acceptance must "
            "replay the resulting controller policy from power-on without "
            "loading it."
        ),
        "capturedAt": datetime.now(timezone.utc).isoformat(),
    }
    provenance_path = STATE_PATH.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
