from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from harvest.runtime import retro_setup


class RetroSetupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original = {
            "SCRIPT_DIR": retro_setup.SCRIPT_DIR,
            "INTEGRATION_PATH": retro_setup.INTEGRATION_PATH,
            "GAME_DIR": retro_setup.GAME_DIR,
            "STATES_DIR": retro_setup.STATES_DIR,
            "ROM_LINK": retro_setup.ROM_LINK,
            "ROM_SHA_PATH": retro_setup.ROM_SHA_PATH,
        }

    def tearDown(self) -> None:
        for key, value in self._original.items():
            setattr(retro_setup, key, value)

    def _point_to_temp_harvest(self, root: Path) -> None:
        script_dir = root / "harvest"
        game_dir = script_dir / "custom_integrations" / retro_setup.GAME
        retro_setup.SCRIPT_DIR = script_dir
        retro_setup.INTEGRATION_PATH = script_dir / "custom_integrations"
        retro_setup.GAME_DIR = game_dir
        retro_setup.STATES_DIR = game_dir
        retro_setup.ROM_LINK = game_dir / "rom.sfc"
        retro_setup.ROM_SHA_PATH = game_dir / "rom.sha"

    def test_ensure_harvest_rom_repairs_broken_link_from_known_rom_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._point_to_temp_harvest(root)
            retro_setup.GAME_DIR.mkdir(parents=True)

            rom_bytes = b"fake-rom-for-setup-test"
            rom_path = root / "roms" / "Harvest Moon.smc"
            rom_path.parent.mkdir()
            rom_path.write_bytes(rom_bytes)
            retro_setup.ROM_SHA_PATH.write_text(
                hashlib.sha1(rom_bytes).hexdigest(), encoding="utf-8"
            )

            retro_setup.ROM_LINK.symlink_to(root / "missing" / "Harvest Moon.sfc")
            self.assertFalse(retro_setup.ROM_LINK.exists())

            result = retro_setup.ensure_harvest_rom(required=True, quiet=True)

            self.assertEqual(result, retro_setup.ROM_LINK)
            self.assertTrue(retro_setup.ROM_LINK.exists())
            self.assertEqual(retro_setup.ROM_LINK.read_bytes(), rom_bytes)

    def test_backup_mutable_start_state_returns_stable_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._point_to_temp_harvest(root)
            retro_setup.STATES_DIR.mkdir(parents=True)
            source = retro_setup.STATES_DIR / "latest.state"
            source.write_bytes(b"state-data")

            backup_name = retro_setup.backup_mutable_start_state(
                "latest", "coop chores"
            )

            self.assertIsNotNone(backup_name)
            self.assertNotEqual(backup_name, "latest")
            self.assertTrue(backup_name.startswith("latest_backup_coop_chores_"))
            self.assertEqual(
                (retro_setup.STATES_DIR / f"{backup_name}.state").read_bytes(),
                b"state-data",
            )

    def test_backup_mutable_start_state_leaves_stable_state_name_alone(self) -> None:
        self.assertEqual(
            retro_setup.backup_mutable_start_state("Y1_After_Sleep", "task"),
            "Y1_After_Sleep",
        )

    def test_make_harvest_env_delegates_to_shared_game_spec(self) -> None:
        fake_retro = SimpleNamespace(
            data=SimpleNamespace(Integrations=SimpleNamespace(ALL="all")),
            Actions=SimpleNamespace(ALL="actions-all"),
        )
        game = Mock()
        game.make_env.return_value = object()

        with (
            patch.dict(sys.modules, {"stable_retro": fake_retro}),
            patch.object(retro_setup, "HARVEST_GAME", game),
            patch.object(retro_setup, "register_harvest_integration") as register,
        ):
            result = retro_setup.make_harvest_env(
                "FirstAction",
                require_rom=False,
                render_mode=None,
                foo="bar",
            )

        self.assertIs(result, game.make_env.return_value)
        register.assert_called_once_with(fake_retro, require_rom=False)
        game.make_env.assert_called_once_with(
            "FirstAction",
            render_mode=None,
            inttype="all",
            use_restricted_actions="actions-all",
            foo="bar",
        )


if __name__ == "__main__":
    unittest.main()
