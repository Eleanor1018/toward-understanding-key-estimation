"""运行入口的 CPU 回归测试：不导入 Isaac，不启动模拟器或 CUDA。"""
import copy
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from estnet.config import Config
from estnet.networks import EstNet
from estnet.preflight import ASSET_HASHES
from estnet.run import close_resources, load_evaluation_checkpoint


class RunnerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="estnet-runner-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "checkpoint.pt"
        self.cfg = Config(num_envs=4, encoder_hidden=(8,), actor_hidden=(8,), critic_hidden=(8,))
        self.model = EstNet(self.cfg)
        # 路径可迁移，内容签名必须一致；两份资产记录刻意使用不同目录。
        self.asset = self.asset_record("/evaluation/assets/g1")
        self.payload = {
            "schema": self.cfg.schema,
            "config": self.cfg.to_dict(),
            "iteration": 7,
            "model": self.model.state_dict(),
            "optimizer": {"optimizer": {"state": {0: {"exp_avg": torch.ones(8)}}}},
            "torch_rng": torch.get_rng_state(),
            "asset": self.asset_record("/training/assets/g1"),
        }

    @staticmethod
    def asset_record(root):
        return {
            "ready": True,
            "files": [
                {"path": root + "/" + name, "sha256": digest,
                 "expected_sha256": digest, "matches_known_asset": True, "exists": True}
                for name, digest in ASSET_HASHES.items()
            ],
        }

    def write_checkpoint(self, payload=None):
        torch.save(self.payload if payload is None else payload, self.path)

    def test_missing_checkpoint_is_rejected_before_initialization(self):
        with self.assertRaises(FileNotFoundError):
            load_evaluation_checkpoint(self.path, self.asset)

    def test_changed_asset_is_rejected_even_when_ready_flags_are_true(self):
        self.write_checkpoint()
        different = copy.deepcopy(self.asset)
        different["files"][2]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            load_evaluation_checkpoint(self.path, different)

    def test_evaluation_payload_is_cpu_only_and_rejects_bad_model_or_metadata(self):
        self.write_checkpoint()
        checkpoint, cfg = load_evaluation_checkpoint(self.path, self.asset)
        self.assertEqual(cfg.to_dict(), self.cfg.to_dict())
        self.assertEqual(checkpoint["iteration"], 7)
        self.assertNotIn("optimizer", checkpoint)
        self.assertNotIn("torch_rng", checkpoint)
        self.assertEqual(set(checkpoint["model"]), set(self.payload["model"]))
        for key, tensor in checkpoint["model"].items():
            self.assertEqual(tensor.device.type, "cpu")
            torch.testing.assert_close(tensor, self.payload["model"][key])
        bad = copy.deepcopy(self.payload)
        key = next(k for k, value in bad["model"].items() if value.ndim == 2)
        bad["model"][key] = torch.zeros(1)
        self.write_checkpoint(bad)
        with self.assertRaises((ValueError, RuntimeError)):
            load_evaluation_checkpoint(self.path, self.asset)
        # 坏元数据必须在返回给main前拒绝，否则严格JSON写入可能再次失败，
        # 使原本应有的failed result记录也被含NaN的manifest阻断。
        for field in ("iteration", "gamma"):
            with self.subTest(nonfinite_field=field):
                bad = copy.deepcopy(self.payload)
                if field == "iteration":
                    bad["iteration"] = float("nan")
                else:
                    bad["config"]["gamma"] = float("nan")
                self.write_checkpoint(bad)
                with self.assertRaises(ValueError):
                    load_evaluation_checkpoint(self.path, self.asset)

    def test_application_closes_after_environment_close_failure_and_partial_init(self):
        calls = []

        class Environment:
            def close(self):
                calls.append("environment")
                raise RuntimeError("environment cleanup failed")

        class Application:
            def close(self):
                calls.append("application")

        with self.assertRaisesRegex(RuntimeError, "environment cleanup failed"):
            close_resources(Environment(), Application())
        self.assertEqual(calls, ["environment", "application"])
        calls.clear()
        close_resources(None, Application())
        self.assertEqual(calls, ["application"])
        close_resources(None, None)

    def test_native_application_exit_preserves_measurement_without_cleanup_success(self):
        # 必须用真实子进程：os._exit不会执行Python的finally或退出钩子。
        # 子进程只替换Isaac对象与测量过程，main和记录写入仍使用生产代码。
        run_dir = Path(self.temp.name) / "native-exit"
        repository = Path(__file__).resolve().parents[1]
        child = textwrap.dedent("""
            import os
            from pathlib import Path
            import sys
            import types
            from unittest.mock import patch

            run_dir = Path(sys.argv[1])
            sys.path.insert(0, sys.argv[2])
            from estnet import run

            class Application:
                def close(self):
                    (run_dir / "native-close-called.txt").write_text("os._exit(0)", encoding="utf-8")
                    os._exit(0)

            class Launcher:
                def __init__(self, **kwargs):
                    self._sim_app_config = {}
                    self._config_resolution(kwargs)
                    self.app = Application()

                def _config_resolution(self, args):
                    pass

            class Environment:
                def __init__(self, cfg):
                    self.device = "cpu"

                def reset(self):
                    return {}, {}

                def close(self):
                    pass

            app_module = types.ModuleType("isaaclab.app")
            app_module.AppLauncher = Launcher
            env_module = types.ModuleType("estnet.environment")
            env_module.EstNetEnv = Environment
            env_module.build_env_cfg = lambda cfg, *args: types.SimpleNamespace(baseline=cfg)
            measurement = {"status": "nominal_pose_unstable", "terminal_events": 2, "walking_verified": False}
            argv = ["estnet.run", "smoke", "--asset", "unused.usd", "--device", "cpu",
                    "--num-envs", "1", "--cpu-threads", "1", "--headless", "--run-dir", str(run_dir)]
            with patch.dict(sys.modules, {
                    "isaaclab": types.ModuleType("isaaclab"), "isaaclab.app": app_module,
                    "estnet.environment": env_module}), \\
                    patch.object(run, "inspect_asset", return_value={"ready": True, "files": []}), \\
                    patch.object(run, "smoke_check", return_value=measurement), \\
                    patch.object(sys, "argv", argv):
                code = run.main()
            (run_dir / "main-returned.txt").write_text("unexpected", encoding="utf-8")
            raise SystemExit(code)
        """)
        completed = subprocess.run(
            [sys.executable, "-B", "-c", child, str(run_dir), str(repository)],
            cwd=self.temp.name, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertTrue((run_dir / "native-close-called.txt").is_file())
        self.assertFalse((run_dir / "main-returned.txt").exists())
        measurement = json.loads((run_dir / "measurement.json").read_text(encoding="utf-8"))
        self.assertEqual(measurement, {
            "status": "nominal_pose_unstable", "terminal_events": 2, "walking_verified": False,
        })
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "cleanup_pending")
        self.assertEqual(result["measurement_file"], "measurement.json")
        self.assertNotEqual(result.get("cleanup_status"), "completed")
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "cleanup_pending")


if __name__ == "__main__":
    unittest.main()
