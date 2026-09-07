"""续训入口的 CPU 回归：只模拟两次更新，不启动 Isaac 或真实 CUDA。"""

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from estnet import run
from estnet.config import Config
from estnet.networks import EstNet
from estnet.ppo import PPO


class ResumeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="estnet-resume-runner-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # 刻意区别于CLI默认值，检查续训是否沿用检查点中的任务与PPO配置。
        self.cfg = Config(
            num_envs=3,
            seed=73,
            horizon=7,
            gamma=0.97,
            learning_rate=2e-4,
            reward_target_height=0.83,
            encoder_hidden=(8,),
            actor_hidden=(8,),
            critic_hidden=(8,),
        )
        self.cfg.validate()
        model = EstNet(self.cfg)
        learner = PPO(model, self.cfg)
        learner.updates = 500
        self.asset = {"ready": True, "files": []}
        self.payload = {
            "schema": self.cfg.schema,
            "config": self.cfg.to_dict(),
            "iteration": 500,
            "model": model.state_dict(),
            "optimizer": learner.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "asset": self.asset,
        }
        self.checkpoint = self.root / "model_00500.pt"
        torch.save(self.payload, self.checkpoint)

    def execute(self, target, device="cpu"):
        """运行真实main和文件落盘流程，仅替换仿真、采样与学习更新。"""
        output = self.root / f"target-{target}"
        calls = {
            "launcher": [],
            "environment": [],
            "rollout": [],
            "update": [],
            "close": [],
        }
        restored = {
            "updates": 500,
            "optimizer_restored": True,
            "cuda_rng_restored": False,
        }

        class Application:
            def close(self):
                calls["close"].append("application")

        class Launcher:
            def __init__(self, **kwargs):
                calls["launcher"].append(kwargs)
                self._sim_app_config = {}
                self._config_resolution(kwargs)
                self.app = Application()

            def _config_resolution(self, args):
                pass

        class Environment:
            def __init__(self, env_cfg):
                calls["environment"].append(env_cfg.baseline)
                self.device = "cpu"

            def reset(self):
                return {"token": 0}, {}

            def close(self):
                calls["close"].append("environment")

        class Learner:
            def __init__(self, model, cfg):
                self.model = model
                self.cfg = cfg
                self.updates = 0

            def update(self, batch):
                self.updates += 1
                calls["update"].append((self.updates, batch["token"], self.cfg))
                return {"loss": 0.0}

            def state_dict(self):
                return {"updates": self.updates, "optimizer": {}}

        def rollout(env, model, obs, cfg):
            token = obs["token"] + 1
            calls["rollout"].append((token, cfg))
            return {"token": token}, {"token": token}, {"reward/total": 0.5}

        def restore(model, learner, payload):
            self.assertIs(payload, self.payload)
            model.load_state_dict(payload["model"], strict=True)
            learner.updates = payload["iteration"]
            return restored

        app_module = types.ModuleType("isaaclab.app")
        app_module.AppLauncher = Launcher
        env_module = types.ModuleType("estnet.environment")
        env_module.EstNetEnv = Environment
        env_module.build_env_cfg = lambda cfg, *args: types.SimpleNamespace(
            baseline=cfg
        )
        resume_module = types.ModuleType("estnet.resume")
        resume_module.load_training_checkpoint = Mock(
            return_value=(self.payload, self.cfg)
        )
        resume_module.restore_training_state = Mock(side_effect=restore)
        argv = [
            "estnet.run",
            "train",
            "--asset",
            "unused.usd",
            "--resume",
            str(self.checkpoint),
            "--iterations",
            str(target),
            "--device",
            device,
            "--cpu-threads",
            "1",
            "--headless",
            "--run-dir",
            str(output),
        ]
        captured = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    sys.modules,
                    {
                        "isaaclab": types.ModuleType("isaaclab"),
                        "isaaclab.app": app_module,
                        "estnet.environment": env_module,
                        "estnet.resume": resume_module,
                    },
                )
            )
            stack.enter_context(patch.dict(os.environ))
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(
                patch.object(run, "inspect_asset", return_value=self.asset)
            )
            stack.enter_context(
                patch.object(run, "runtime_manifest", return_value={"test": "CPU fake"})
            )
            stack.enter_context(
                patch.object(run, "collect_rollout", side_effect=rollout)
            )
            stack.enter_context(patch("estnet.ppo.PPO", Learner))
            save = stack.enter_context(
                patch.object(run, "save_checkpoint", wraps=run.save_checkpoint)
            )
            select_cuda = stack.enter_context(
                patch.object(
                    torch.cuda,
                    "set_device",
                    side_effect=AssertionError("CUDA must not initialize in CPU test"),
                )
            )
            init_cuda = stack.enter_context(
                patch.object(
                    torch.cuda,
                    "_lazy_init",
                    side_effect=AssertionError("CUDA must not initialize in CPU test"),
                )
            )
            stack.enter_context(contextlib.redirect_stdout(captured))
            stack.enter_context(contextlib.redirect_stderr(captured))
            code = run.main()
        return types.SimpleNamespace(
            output=output,
            calls=calls,
            code=code,
            captured=captured.getvalue(),
            save=save,
            load=resume_module.load_training_checkpoint,
            restore=resume_module.restore_training_state,
            restored=restored,
            select_cuda=select_cuda,
            init_cuda=init_cuda,
        )

    def test_resume_500_to_total_502_runs_only_501_and_502_and_preserves_config(self):
        result = self.execute(502)
        self.assertEqual(result.code, 0, result.captured)
        self.assertEqual([item[0] for item in result.calls["rollout"]], [1, 2])
        self.assertEqual(
            [(item[0], item[1]) for item in result.calls["update"]],
            [(501, 1), (502, 2)],
        )
        self.assertTrue(all(item[1] is self.cfg for item in result.calls["rollout"]))
        self.assertTrue(all(item[2] is self.cfg for item in result.calls["update"]))
        self.assertEqual(result.calls["environment"], [self.cfg])
        self.assertEqual(result.calls["close"], ["environment", "application"])
        result.load.assert_called_once_with(self.checkpoint, self.asset)
        result.restore.assert_called_once()
        result.select_cuda.assert_not_called()
        result.init_cuda.assert_not_called()
        result.save.assert_called_once()
        self.assertEqual(
            result.save.call_args.args[0], result.output / "model_00502.pt"
        )
        self.assertEqual(result.save.call_args.args[4], 502)
        self.assertIs(result.save.call_args.args[3], self.cfg)
        self.assertEqual(
            sorted(p.name for p in result.output.glob("model_*.pt")), ["model_00502.pt"]
        )
        saved = torch.load(
            result.output / "model_00502.pt", map_location="cpu", weights_only=True
        )
        self.assertEqual(saved["iteration"], 502)
        self.assertEqual(saved["optimizer"]["updates"], 502)
        self.assertEqual(saved["config"], self.cfg.to_dict())
        for key, tensor in saved["model"].items():
            torch.testing.assert_close(tensor, self.payload["model"][key])
        metrics = [
            json.loads(line)
            for line in (result.output / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertEqual([row["iteration"] for row in metrics], [501, 502])
        self.assertEqual([row["ppo_updates"] for row in metrics], [501, 502])
        final = json.loads((result.output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(final["status"], "pilot_finished_requires_evaluation")
        self.assertEqual(
            (final["start_iteration"], final["iterations"], final["updates_this_run"]),
            (500, 502, 2),
        )
        self.assertFalse(final["walking_verified"])
        manifest = json.loads(
            (result.output / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["config"], json.loads(json.dumps(self.cfg.to_dict())))
        resume = manifest["resume"]
        self.assertEqual(
            (
                resume["start_iteration"],
                resume["target_iteration"],
                resume["additional_iterations"],
            ),
            (500, 502, 2),
        )
        self.assertEqual(
            resume["checkpoint_sha256"],
            hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
        )
        self.assertEqual(resume["restored_state"], result.restored)
        self.assertFalse(resume["simulation_state_restored"])
        self.assertFalse(resume["history_restored"])

    def test_target_at_or_before_checkpoint_fails_before_application_or_cuda(self):
        for target in (499, 500):
            with self.subTest(target=target):
                # 即使命令请求CUDA，已完成的目标必须在CPU预检阶段拒绝。
                result = self.execute(target, device="cuda:99")
                self.assertEqual(result.code, 2, result.captured)
                result.load.assert_called_once_with(self.checkpoint, self.asset)
                result.restore.assert_not_called()
                result.select_cuda.assert_not_called()
                result.init_cuda.assert_not_called()
                result.save.assert_not_called()
                for name in ("launcher", "environment", "rollout", "update", "close"):
                    self.assertEqual(result.calls[name], [])
                self.assertFalse((result.output / "metrics.jsonl").exists())
                self.assertEqual(list(result.output.glob("model_*.pt")), [])
                final = json.loads(
                    (result.output / "result.json").read_text(encoding="utf-8")
                )
                self.assertEqual(final["status"], "failed")
                self.assertEqual(final["failure"]["phase"], "cpu_preflight")
                self.assertEqual(final["failure"]["type"], "ValueError")
                self.assertIn("iterations", final["failure"]["message"])


if __name__ == "__main__":
    unittest.main()
