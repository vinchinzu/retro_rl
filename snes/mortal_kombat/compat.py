"""Load shims for SB3 zips pickled under the pre-rename ``fighters_common`` package."""

from __future__ import annotations

import sys


def install_fighters_common_alias() -> None:
    """Map ``fighters_common.*`` to ``retro_harness.fighters.*`` for old pixel zips."""
    try:
        import retro_harness.fighters as fighters
        import retro_harness.fighters.fighting_env as fighting_env
        import retro_harness.fighters.game_configs as game_configs
        import retro_harness.fighters.menu_nav as menu_nav
        import retro_harness.fighters.ram_observation as ram_observation
    except ImportError:
        return
    aliases = {
        "fighters_common": fighters,
        "fighters_common.fighting_env": fighting_env,
        "fighters_common.game_configs": game_configs,
        "fighters_common.menu_nav": menu_nav,
        "fighters_common.ram_observation": ram_observation,
    }
    for extra in ("combo_wrapper", "train_ppo", "random_opponent_env"):
        try:
            module = __import__(f"retro_harness.fighters.{extra}", fromlist=[extra])
        except ImportError:
            continue
        aliases[f"fighters_common.{extra}"] = module
    for name, module in aliases.items():
        sys.modules.setdefault(name, module)
