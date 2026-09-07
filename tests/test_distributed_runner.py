"""真实双进程runner续训测试；只替换Isaac环境，通信、PPO与检查点均真实。"""

import contextlib
import datetime
import io
import json
import os
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from estnet import run
from estnet.config import Config
from estnet.factory import build_model, build_ppo

ASSET = {"ready": True, "files": [{"sha256": str(i) * 64} for i in range(4)]}


def runner_worker(rank, shared, checkpoint, init_uri, fail):
    """替换GPU仿真为CPU向量环境，实际执行两rank的main与Adam更新。"""
    torch.set_num_threads(1)
    shared = Path(shared)
    observed = {"rank": rank}
    models = []

    class Application:
        def close(self):
            observed["group_closed_before_kit"] = not dist.is_initialized()

    class Launcher:
        def __init__(self, **kwargs):
            assert kwargs["distributed"] is False
            observed["portable_root"] = sys.argv[sys.argv.index("--portable-root") + 1]
            self._sim_app_config = {}
            self._config_resolution(kwargs)
            self.app = Application()

        def _config_resolution(self, args):
            assert args["distributed"] is False

    class Environment:
        def __init__(self, env_cfg):
            self.cfg = env_cfg.baseline
            self.num_envs = self.cfg.num_envs
            self.device = "cpu"
            observed["local_num_envs"] = self.num_envs
            observed["environment_seed"] = self.cfg.seed

        def observation(self):
            n = self.num_envs
            velocity = torch.full((n, 3), float("nan") if fail and rank == 1 else 0.2)
            return {"obs": torch.randn(n, 42), "history": torch.randn(n, 50, 42),
                    "command": torch.randn(n, 7), "critic": torch.randn(n, 61), "velocity": velocity}

        def reset(self):
            return self.observation(), {}

        def step(self, action):
            assert action.shape == (self.num_envs, 12)
            obs = self.observation()
            reward = torch.full((self.num_envs,), 0.2 + 0.2 * rank)
            boundary = torch.zeros(self.num_envs, dtype=torch.bool)
            return obs, reward, boundary, boundary, {
                "final_critic": obs["critic"].clone(),
                "metrics": {"forward_velocity": torch.full((self.num_envs,), 1.0 + 2 * rank)},
                "reward_terms": {"alive": reward.clone()}}

        def close(self):
            observed["environment_closed"] = True

    def env_config(cfg, *_):
        physx = types.SimpleNamespace(to_dict=lambda: {"enable_stabilization": True})
        return types.SimpleNamespace(baseline=cfg, decimation=cfg.decimation,
            sim=types.SimpleNamespace(dt=cfg.sim_dt, device="cpu", render_interval=10, physx=physx),
            scene=types.SimpleNamespace(num_envs=cfg.num_envs, replicate_physics=True, clone_in_fabric=False))

    def initialize(device):
        assert device.type == "cpu"
        dist.init_process_group("gloo", init_method=init_uri, rank=rank, world_size=2,
                                timeout=datetime.timedelta(seconds=60))

    def model_factory(cfg):
        model = build_model(cfg)
        models.append(model)
        return model

    app_module = types.ModuleType("isaaclab.app")
    app_module.AppLauncher = Launcher
    env_module = types.ModuleType("estnet.environment")
    env_module.EstNetEnv, env_module.build_env_cfg = Environment, env_config
    argv = ["estnet.run", "train", "--asset", "unused.usd", "--resume", checkpoint,
            "--iterations", "501" if fail else "502", "--device", "cpu", "--distributed",
            "--cpu-threads", "1", "--headless", "--run-dir", str(shared)]
    output = io.StringIO()
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"RANK": str(rank), "LOCAL_RANK": str(rank),
                                                   "WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2"}))
        stack.enter_context(patch.dict(sys.modules, {"isaaclab": types.ModuleType("isaaclab"),
                                                   "isaaclab.app": app_module, "estnet.environment": env_module}))
        stack.enter_context(patch.object(sys, "argv", argv))
        stack.enter_context(patch.object(run, "inspect_asset", return_value=ASSET))
        stack.enter_context(patch.object(run, "_initialize_process_group", side_effect=initialize))
        stack.enter_context(patch.object(run, "build_model", side_effect=model_factory))
        stack.enter_context(patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU test must not initialize CUDA")))
        stack.enter_context(contextlib.redirect_stdout(output))
        stack.enter_context(contextlib.redirect_stderr(output))
        observed["exit_code"] = run.main()
    observed["output"] = output.getvalue()
    (shared / f"worker-{rank}.json").write_text(json.dumps(observed), encoding="utf-8")
    if models:
        torch.save(models[-1].state_dict(), shared / f"worker-{rank}-model.pt")


class DistributedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="estnet-distributed-runner-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cfg = Config(num_envs=4, horizon=2, epochs=1, minibatches=1,
                          encoder_hidden=(8,), actor_hidden=(8,), critic_hidden=(8,))
        model = build_model(self.cfg)
        learner = build_ppo(model, self.cfg)
        # 合成已训练检查点：为所有参数创建真实Adam矩状态，再将全局计数设为500。
        sum(parameter.square().sum() for parameter in model.parameters()).backward()
        learner.optimizer.step()
        learner.updates = 500
        self.checkpoint = self.root / "model_00500.pt"
        run.save_checkpoint(self.checkpoint, model, learner, self.cfg, 500, ASSET)

    def execute_pair(self, fail=False):
        """父目录可由外部supervisor预建，但各rank必须独立创建其子目录。"""
        shared = self.root / ("failure" if fail else "success")
        shared.mkdir()
        mp.spawn(runner_worker, args=(str(shared), str(self.checkpoint),
                                     (self.root / ("group-fail" if fail else "group-ok")).as_uri(), fail),
                 nprocs=2, join=True)
        workers = [json.loads((shared / f"worker-{i}.json").read_text(encoding="utf-8")) for i in range(2)]
        return shared, workers

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU Gloo unavailable")
    def test_two_ranks_resume_to_total_502_with_global_config_and_one_checkpoint(self):
        shared, workers = self.execute_pair()
        for worker in workers:
            self.assertEqual(worker["exit_code"], 0, worker["output"])
            self.assertEqual(worker["local_num_envs"], 2)
            self.assertTrue(worker["group_closed_before_kit"])
        self.assertNotEqual(workers[0]["environment_seed"], workers[1]["environment_seed"])
        self.assertNotEqual(workers[0]["portable_root"], workers[1]["portable_root"])
        saved_path = shared / "model_00502.pt"
        saved = torch.load(saved_path, map_location="cpu", weights_only=True)
        self.assertEqual(saved["config"]["num_envs"], 4)
        self.assertEqual(saved["optimizer"]["updates"], 502)
        self.assertEqual(saved["distributed"]["global_rollout_samples"], 8)
        self.assertEqual(saved["distributed"]["local_rollout_samples"], 4)
        self.assertEqual(len(saved["distributed_rng"]), 2)
        self.assertFalse(torch.equal(saved["distributed_rng"][0]["torch_rng"], saved["distributed_rng"][1]["torch_rng"]))
        run._validate_checkpoint(saved, ASSET)
        rank_metrics = []
        for index in range(2):
            directory = shared / f"rank-{index}"
            self.assertEqual(list(directory.glob("model_*.pt")), [])
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            self.assertEqual((result["start_iteration"], result["iterations"], result["updates_this_run"]), (500, 502, 2))
            self.assertEqual(result["cleanup_status"], "completed")
            self.assertEqual(result["result_scope"], "local_rank")
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["config"]["num_envs"], 4)
            self.assertEqual(manifest["local_config"]["num_envs"], 2)
            self.assertTrue(manifest["resolved_physics"]["physx"]["enable_stabilization"])
            self.assertTrue(manifest["first_distributed_update"]["all_equal"])
            self.assertEqual(manifest["first_distributed_update"]["iteration"], 501)
            self.assertEqual(manifest["process_group_cleanup"], "completed")
            metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["iteration"] for row in metrics], [501, 502])
            self.assertTrue(all(row["env/forward_velocity"] == 2.0 for row in metrics))
            rank_metrics.append(metrics)
            local_model = torch.load(shared / f"worker-{index}-model.pt", weights_only=True)
            for name, value in local_model.items():
                torch.testing.assert_close(value, saved["model"][name], rtol=0, atol=0)
        self.assertEqual(rank_metrics[0], rank_metrics[1])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU Gloo unavailable")
    def test_one_rank_nan_fails_both_without_cleanup_barrier_or_checkpoint(self):
        shared, workers = self.execute_pair(fail=True)
        for worker in workers:
            self.assertEqual(worker["exit_code"], 2, worker["output"])
            self.assertTrue(worker["group_closed_before_kit"])
            result = json.loads((shared / f'rank-{worker["rank"]}' / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure"]["type"], "FloatingPointError")
        self.assertEqual(list(shared.glob("model_*.pt")), [])

    def test_local_partition_and_saved_rank_rng_are_explicit(self):
        layout = {"rank": 1, "world_size": 2}
        local = run._local_training_config(self.cfg, layout, 500)
        self.assertEqual((local.num_envs, self.cfg.num_envs), (2, 4))
        with self.assertRaises(ValueError):
            run._local_training_config(replace(self.cfg, num_envs=3), layout, 500)
        with self.assertRaises(ValueError):
            run._local_training_config(replace(self.cfg, horizon=3, minibatches=4), layout, 500)
        states = []
        for index in range(2):
            generator = torch.Generator().manual_seed(100 + index)
            states.append({"rank": index, "torch_rng": generator.get_state()})
        payload = {"distributed_rng": states, "distributed": {"world_size": 2}}
        run._validate_distributed_rng(payload)
        restored = run._restore_rank_rng(payload, layout, torch.device("cpu"), self.cfg, 500)
        self.assertEqual(restored["mode"], "saved_per_rank_rng")
        self.assertTrue(torch.equal(torch.get_rng_state(), states[1]["torch_rng"]))

    def test_physical_gpu_selection_matches_uuid_when_ordinals_differ(self):
        """只模拟设备元数据，证明物理编号不同于Torch ordinal时不会静默选错卡。"""
        args = types.SimpleNamespace(render_gpu_indices="3,2", render_gpu=None)
        layout = {"gpu_indices": [6, 7], "render_gpu_indices": [3, 2], "local_rank": 1}
        properties = [types.SimpleNamespace(uuid="GPU-bbbb", name="test GPU B"),
                      types.SimpleNamespace(uuid="GPU-aaaa", name="test GPU A")]
        with patch.object(run.subprocess, "run", return_value=types.SimpleNamespace(stdout="6, GPU-aaaa\n7, GPU-bbbb\n")), \
                patch.object(torch.cuda, "device_count", return_value=2), \
                patch.object(torch.cuda, "get_device_properties", side_effect=lambda index: properties[index]), \
                patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("Metadata test must not initialize CUDA")):
            device, metadata = run._distributed_cuda_device(args, layout)
        self.assertEqual(str(device), "cuda:0")
        self.assertEqual(metadata["physical_index"], 7)
        self.assertTrue(metadata["uuid_verified"])
        self.assertEqual(args.render_gpu, 2)
        with self.assertRaises(ValueError):
            run._parse_indices("0,0")


if __name__ == "__main__":
    unittest.main()
