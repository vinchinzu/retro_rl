"""
Model registry tests.

Tests:
- Registry CRUD (register, get, list, best_for_level)
- Benchmark recording and win rate calculation
- Model lineage tracking (parent chain)
- Registry file persistence (JSON read/write)
- Edge cases (missing models, duplicate registrations)

Uses a temp registry file so tests don't affect real data.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _get_registry_class():
    """Import Registry from mortal_kombat/model_registry.py."""
    mk_dir = ROOT_DIR / "mortal_kombat"
    if str(mk_dir) not in sys.path:
        sys.path.insert(0, str(mk_dir))
    from model_registry import Registry
    return Registry


class TestRegistryCRUD(unittest.TestCase):
    """Test basic registry operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry_path = Path(self.tmpdir) / "test_registry.json"
        Registry = _get_registry_class()
        self.reg = Registry(path=self.registry_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_register_creates_entry(self):
        entry = self.reg.register("test_model.zip", parent="base.zip", steps=1000)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["parent"], "base.zip")
        self.assertEqual(entry["steps"], 1000)

    def test_get_returns_registered_model(self):
        self.reg.register("test_model.zip", notes="test")
        entry = self.reg.get("test_model.zip")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["notes"], "test")

    def test_get_returns_none_for_missing(self):
        self.assertIsNone(self.reg.get("nonexistent.zip"))

    def test_list_models(self):
        self.reg.register("model_a.zip")
        self.reg.register("model_b.zip")
        models = self.reg.list_models()
        self.assertEqual(len(models), 2)
        self.assertIn("model_a.zip", models)
        self.assertIn("model_b.zip", models)

    def test_registry_persists_to_disk(self):
        self.reg.register("persistent.zip", notes="should persist")

        # Create new registry instance from same file
        Registry = _get_registry_class()
        reg2 = Registry(path=self.registry_path)
        entry = reg2.get("persistent.zip")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["notes"], "should persist")

    def test_registry_json_is_valid(self):
        self.reg.register("test.zip")
        content = self.registry_path.read_text()
        data = json.loads(content)
        self.assertIn("models", data)
        self.assertIn("test.zip", data["models"])

    def test_register_overwrites_existing(self):
        self.reg.register("model.zip", notes="version 1")
        self.reg.register("model.zip", notes="version 2")
        entry = self.reg.get("model.zip")
        self.assertEqual(entry["notes"], "version 2")


class TestBenchmarkRecording(unittest.TestCase):
    """Test benchmark score recording."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry_path = Path(self.tmpdir) / "test_registry.json"
        Registry = _get_registry_class()
        self.reg = Registry(path=self.registry_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_benchmark(self):
        self.reg.register("model.zip")
        results = {
            "LiuKang": {"wins": 5, "losses": 0, "matches": 5},
            "Scorpion": {"wins": 2, "losses": 3, "matches": 5},
        }
        wr = self.reg.record_benchmark("model.zip", level=1, results=results)
        self.assertAlmostEqual(wr, 70.0)  # 7/10 = 70%

    def test_benchmark_stored_correctly(self):
        self.reg.register("model.zip")
        results = {
            "A": {"wins": 3, "losses": 2, "matches": 5},
        }
        self.reg.record_benchmark("model.zip", level=2, results=results)

        entry = self.reg.get("model.zip")
        self.assertIn("match2", entry["benchmarks"])
        bm = entry["benchmarks"]["match2"]
        self.assertEqual(bm["total_wins"], 3)
        self.assertEqual(bm["total_matches"], 5)
        self.assertAlmostEqual(bm["overall_win_rate"], 60.0)

    def test_benchmark_auto_registers_missing_model(self):
        """Recording benchmark for unregistered model should auto-register it."""
        results = {"A": {"wins": 1, "losses": 0, "matches": 1}}
        self.reg.record_benchmark("new_model.zip", level=1, results=results)
        entry = self.reg.get("new_model.zip")
        self.assertIsNotNone(entry)

    def test_win_rate_100_percent(self):
        self.reg.register("perfect.zip")
        results = {
            "A": {"wins": 5, "losses": 0, "matches": 5},
            "B": {"wins": 5, "losses": 0, "matches": 5},
        }
        wr = self.reg.record_benchmark("perfect.zip", level=1, results=results)
        self.assertAlmostEqual(wr, 100.0)

    def test_win_rate_0_percent(self):
        self.reg.register("terrible.zip")
        results = {
            "A": {"wins": 0, "losses": 5, "matches": 5},
        }
        wr = self.reg.record_benchmark("terrible.zip", level=1, results=results)
        self.assertAlmostEqual(wr, 0.0)

    def test_best_for_level(self):
        self.reg.register("good.zip")
        self.reg.register("better.zip")
        self.reg.record_benchmark("good.zip", level=1,
                                  results={"A": {"wins": 3, "losses": 2, "matches": 5}})
        self.reg.record_benchmark("better.zip", level=1,
                                  results={"A": {"wins": 5, "losses": 0, "matches": 5}})

        name, rate = self.reg.best_for_level(1)
        self.assertEqual(name, "better.zip")
        self.assertAlmostEqual(rate, 100.0)


class TestModelLineage(unittest.TestCase):
    """Test model parent/child tracking."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry_path = Path(self.tmpdir) / "test_registry.json"
        Registry = _get_registry_class()
        self.reg = Registry(path=self.registry_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parent_chain(self):
        """Can trace lineage through parent pointers."""
        self.reg.register("base.zip", parent=None)
        self.reg.register("child.zip", parent="base.zip")
        self.reg.register("grandchild.zip", parent="child.zip")

        gc = self.reg.get("grandchild.zip")
        self.assertEqual(gc["parent"], "child.zip")

        child = self.reg.get(gc["parent"])
        self.assertEqual(child["parent"], "base.zip")

        base = self.reg.get(child["parent"])
        self.assertIsNone(base["parent"])

    def test_training_metadata_preserved(self):
        self.reg.register(
            "model.zip",
            parent="base.zip",
            script="train_match2.py",
            steps=1000000,
            lr=1e-4,
            training_mix="40% Match2 + 60% Fight",
            notes="conservative LR",
        )
        entry = self.reg.get("model.zip")
        self.assertEqual(entry["script"], "train_match2.py")
        self.assertEqual(entry["steps"], 1000000)
        self.assertEqual(entry["lr"], 1e-4)
        self.assertIn("Match2", entry["training_mix"])


class TestRealRegistry(unittest.TestCase):
    """Test the actual registry file (if it exists)."""

    def test_real_registry_is_valid_json(self):
        registry_path = ROOT_DIR / "mortal_kombat" / "models" / "registry.json"
        if not registry_path.exists():
            self.skipTest("No registry.json")

        content = registry_path.read_text()
        data = json.loads(content)
        self.assertIn("models", data)

    def test_real_registry_models_have_required_fields(self):
        registry_path = ROOT_DIR / "mortal_kombat" / "models" / "registry.json"
        if not registry_path.exists():
            self.skipTest("No registry.json")

        data = json.loads(registry_path.read_text())

        for name, entry in data["models"].items():
            with self.subTest(model=name):
                self.assertIn("registered", entry, f"{name}: missing 'registered' timestamp")
                self.assertIn("benchmarks", entry, f"{name}: missing 'benchmarks' dict")
                # Parent can be None for base models
                self.assertIn("parent", entry, f"{name}: missing 'parent' field")

    def test_real_registry_no_circular_lineage(self):
        """Check for circular parent references (would loop forever)."""
        registry_path = ROOT_DIR / "mortal_kombat" / "models" / "registry.json"
        if not registry_path.exists():
            self.skipTest("No registry.json")

        data = json.loads(registry_path.read_text())
        models = data["models"]

        for name in models:
            visited = set()
            current = name
            while current and current in models:
                self.assertNotIn(current, visited,
                                 f"Circular lineage detected: {name} -> ... -> {current}")
                visited.add(current)
                current = models[current].get("parent")


if __name__ == "__main__":
    unittest.main()
