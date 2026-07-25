"""Run SMW tooling with: python -m SMW."""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"speedrun", "play-speedrun"}:
        from SMW.speedrun_play import main as speedrun_main

        speedrun_main(sys.argv[2:])
        return

    # Importing this module registers SMW levels before the shared runner
    # resolves CLI level aliases.
    import platformer_common.levels.super_mario_world  # noqa: F401
    from platformer_common.runner import main as runner_main

    runner_main()


if __name__ == "__main__":
    main()
