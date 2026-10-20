"""Check observable build counters and credential removal from backend errors."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("build_telemetry", ROOT / ".github/scripts/build_telemetry.py")
TELEMETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TELEMETRY)


class BuildTelemetryTest(unittest.TestCase):
    def test_cache_failures_remain_visible_in_progress(self):
        stats = dict(cache_hits={"counts": {"C/C++": 42}}, cache_misses={"counts": {"PTX": 3}},
                     cache_writes=19, cache_write_errors=2, compile_fails=0)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(TELEMETRY.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 0, stdout=json.dumps({"stats": stats}))), \
                patch("builtins.print"):
            path = Path(temp) / "progress.jsonl"
            TELEMETRY.sample(path)
            report = json.loads(path.read_text())
            self.assertEqual(report["cache_write_errors"], 2)
            self.assertEqual(report["cache_hits"], stats["cache_hits"])
            self.assertGreater(report["free_disk_bytes"], 0)

    def test_stats_timeout_is_reported_without_stopping_compilation(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(TELEMETRY.subprocess, "run", side_effect=subprocess.TimeoutExpired("sccache", 5)), \
                patch("builtins.print"):
            path = Path(temp) / "progress.jsonl"
            TELEMETRY.sample(path)
            self.assertEqual(json.loads(path.read_text())["cache_statistics_unavailable"], "TimeoutExpired")

    def test_backend_errors_keep_cause_but_remove_credentials(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict(os.environ, ACTIONS_RUNTIME_TOKEN="private-fixture-token"):
            root = Path(temp)
            source = root / "raw.log"
            source.write_text('HTTP 429 Too Many Requests https://cache.example/upload?sig=private-signature\n'
                              'ACTIONS_RUNTIME_TOKEN=private-fixture-token\nAuthorization: Bearer private-header\n')
            TELEMETRY.sanitize_cache_log(source, root / "safe.log")
            result = (root / "safe.log").read_text()
            self.assertIn("HTTP 429 Too Many Requests", result)
            for secret in ("private-fixture-token", "private-signature", "private-header"):
                self.assertNotIn(secret, result)
