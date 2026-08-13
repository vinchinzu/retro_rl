"""Run SMW tooling with: python -m SMW."""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"speedrun", "play-speedrun"}:
        from SMW.speedrun_play import main as speedrun_main

        speedrun_main(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] in {"capture-state", "capture_state"}:
        from SMW.capture_state import main as capture_main

        capture_main(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] in {"chain-yi", "chain_yi", "chain-yoshi"}:
        from SMW.chain_yoshi_island import main as chain_yi_main

        chain_yi_main(sys.argv[2:])
        return

    # Importing this module registers SMW levels before the shared runner
    # resolves CLI level aliases.
    import SMW.platformer_levels  # noqa: F401
    from retro_harness.platformer.runner import main as runner_main

    runner_main()


if __name__ == "__main__":
    main()
