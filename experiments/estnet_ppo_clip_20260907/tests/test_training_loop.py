"""真实 runner/采样/PPO/保存/续训的 CPU 集成测试。

只替换 Isaac AppLauncher、物理环境、USD 资产检查和现场自碰撞读回。
观测维度与 Config/schema 是真实协议；小网络及确定性环境仅作流程替身，
本测试不证明真实 G1 动力学、自碰撞响应、GPU 运行或步行通过。
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "estnet"
PACKAGE = "training_loop_cpu_estnet"
spec = importlib.util.spec_from_file_location(PACKAGE, SOURCE / "__init__.py", submodule_search_locations=[str(SOURCE)])
package = importlib.util.module_from_spec(spec)
sys.modules[PACKAGE] = package
spec.loader.exec_module(package)
run = __import__(PACKAGE + ".run", fromlist=["main"])
RealConfig = __import__(PACKAGE + ".config", fromlist=["Config"]).Config
EstNet = __import__(PACKAGE + ".networks", fromlist=["EstNet"]).EstNet
PPO = __import__(PACKAGE + ".ppo", fromlist=["PPO"]).PPO
ASSET_HASHES = __import__(PACKAGE + ".preflight", fromlist=["ASSET_HASHES"]).ASSET_HASHES


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture
def integration(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(218)
    assert not torch.cuda.is_initialized()
    calls = []
    environments = []
    asset = {"ready": True, "files": [
        {"path": str(tmp_path / name), "sha256": digest}
        for name, digest in ASSET_HASHES.items()]}

    def small_config(**kwargs):
        # 仍使用真实冻结 dataclass、42/7/61/12 和 50帧历史协议。
        # 保留三个隐藏层，因此仍是 25 个参数张量；只缩小每层宽度。
        defaults = dict(encoder_hidden=(12, 8, 4), actor_hidden=(12, 8, 4), critic_hidden=(12, 8, 4))
        return RealConfig(**{**defaults, **kwargs})

    class FakeApplication:
        def __init__(self, output):
            self.output = output

        def close(self):
            calls.append(("app_close", self.output))
            # 主入口必须在 native close 前保存测量，但不能先标为 cleanup completed。
            assert read_json(self.output / "result.json")["status"] == "cleanup_pending"
            assert (self.output / "measurement.json").is_file()

    class FakeLauncher:
        def __init__(self, **kwargs):
            output = Path(sys.argv[sys.argv.index("--run-dir") + 1]).resolve()
            calls.append(("app_init", output))
            self._sim_app_config = {}
            self._config_resolution(kwargs)
            assert self._sim_app_config["fast_shutdown"] is False
            assert kwargs["device"] == "cpu"
            self.app = FakeApplication(output)

        def _config_resolution(self, args):
            pass

    class FakeEnvironment:
        def __init__(self, cfg):
            self.params = cfg.params
            self.num_envs = cfg.params.num_envs
            self.device = torch.device("cpu")
            self.output = Path(sys.argv[sys.argv.index("--run-dir") + 1]).resolve()
            self.steps = 0
            self.boundaries = 0
            self.pre_reset_distinct = 0
            self.actions = []
            self.age = torch.zeros(self.num_envs, dtype=torch.long)
            self.lengths = torch.tensor([7, 11, 13, 17])
            self.history = torch.zeros(self.num_envs, 50, 42)
            self.velocity = torch.zeros(self.num_envs, 3)
            calls.append(("env_init", self.output))
            environments.append(self)

        def _observation(self, action=None):
            obs = torch.zeros(self.num_envs, 42)
            obs[:, :3] = self.velocity
            obs[:, 3] = self.age.float() * .01
            if action is not None:
                obs[:, -12:] = action
            command = torch.zeros(self.num_envs, 7)
            command[:, 0] = .4
            critic = torch.zeros(self.num_envs, 61)
            critic[:, :42] = obs
            critic[:, 42:49] = command
            critic[:, 49:52] = self.velocity
            return {"obs": obs, "command": command, "critic": critic,
                    "history": self.history.clone(), "velocity": self.velocity.clone()}

        def reset(self):
            self.age.zero_()
            self.velocity.zero_()
            self.history.zero_()
            calls.append(("env_reset", self.output))
            return self._observation(), {}

        def step(self, action):
            assert action.shape == (4, 12) and torch.isfinite(action).all()
            self.actions.append(action.clone())
            self.steps += 1
            self.age += 1
            self.velocity = .03 * action[:, :3].tanh()
            before = self._observation(action)
            final_critic = before["critic"].clone()
            self.history = torch.roll(self.history, -1, dims=1)
            self.history[:, -1] = before["obs"]
            done = self.age >= self.lengths
            terminal = done & torch.tensor([True, False, True, False])
            timeout = done & ~terminal
            # 确定性、非物理奖励：让真正 PPO 收到非恒定优势和监督信号。
            reward = .1 - (self.velocity[:, 0] - .04).square() - .01 * action.square().mean(-1)
            metric_velocity = self.velocity[:, 0].clone()
            self.boundaries += int(done.sum())
            self.age[done] = 0
            self.velocity[done] = 0.
            self.history[done] = 0.
            next_obs = self._observation(action)
            next_obs["obs"][done] = 0.
            next_obs["critic"][done, :42] = 0.
            self.pre_reset_distinct += int((final_critic[done] != next_obs["critic"][done]).any(-1).sum())
            return next_obs, reward, terminal, timeout, {
                "final_critic": final_critic,
                "metrics": {"fixture_velocity": metric_velocity},
                "reward_terms": {"fixture_only": reward.clone()}}

        def close(self):
            calls.append(("env_close", self.output))

    app_module = ModuleType("isaaclab.app")
    app_module.AppLauncher = FakeLauncher
    env_module = ModuleType(PACKAGE + ".environment")
    env_module.EstNetEnv = FakeEnvironment
    env_module.build_env_cfg = lambda config, *args: SimpleNamespace(params=config)
    monkeypatch.setitem(sys.modules, "isaaclab", ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    monkeypatch.setitem(sys.modules, PACKAGE + ".environment", env_module)
    monkeypatch.setattr(run, "Config", small_config)
    monkeypatch.setattr(run, "inspect_asset", lambda path: copy.deepcopy(asset))
    monkeypatch.setattr(run, "check_live_self_collision", lambda cfg: {
        "fixture_only": True, "configured": cfg.params.self_collisions,
        "contact_response_verified": False, "gpu_or_usd_readback_verified": False})
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PXR_WORK_THREAD_LIMIT"):
        monkeypatch.setenv(key, "1")  # main also writes these; restore them after the test.

    def invoke(output, iterations, resume=None):
        argv = ["estnet.run", "train", "--asset", str(tmp_path / "fixture.usd"),
                "--device", "cpu", "--cpu-threads", "1", "--num-envs", "4", "--headless",
                "--iterations", str(iterations), "--run-dir", str(output)]
        if resume is not None:
            argv.extend(["--resume", str(resume)])
        monkeypatch.setattr(sys, "argv", argv)
        return run.main()

    yield SimpleNamespace(invoke=invoke, calls=calls, environments=environments,
                          asset=asset, small_config=small_config)
    assert not torch.cuda.is_initialized()


def assert_complete_run(output, iterations, start, expected_total):
    result, measurement, manifest = (read_json(output / name) for name in
                                     ("result.json", "measurement.json", "manifest.json"))
    assert result["status"] == "training_finished_requires_evaluation"
    assert result["cleanup_status"] == "completed" and manifest["status"] == "completed"
    assert result["iterations"] == iterations and result["start_iteration"] == start
    assert result["updates_this_run"] == iterations - start
    assert result["total_optimizer_steps"] == measurement["total_optimizer_steps"] == expected_total
    assert result["learning_rate_schedule"] == "fixed" and result["kl_controls_updates"] is False
    assert result["walking_verified"] is False
    assert "total_accepted_steps" not in result
    assert manifest["self_collision_readback"]["fixture_only"] is True
    assert not (output / "failure.txt").exists()
    metrics, events = read_jsonl(output / "metrics.jsonl"), read_jsonl(output / "ppo_events.jsonl")
    assert [row["iteration"] for row in metrics] == list(range(start + 1, iterations + 1))
    assert [row["ppo_updates"] for row in metrics] == list(range(start + 1, iterations + 1))
    assert all(row["ppo/optimizer_steps"] == row["ppo/gradient_steps"] == 16 for row in metrics)
    assert all(row["ppo/total_optimizer_steps"] == row["iteration"] * 16 for row in metrics)
    assert all(row["ppo/learning_rate"] == 5e-4 for row in metrics)
    assert all("reward/total" in row and "env/fixture_velocity" in row for row in metrics)
    for iteration in range(start + 1, iterations + 1):
        selected = [e for e in events if e["iteration"] == iteration]
        minibatches = [e for e in selected if e["event"] == "ppo_minibatch_update"]
        assert len(minibatches) == 16
        assert [e["optimizer_steps"] for e in minibatches] == list(range(1, 17))
        assert all(e["rollout_samples"] == 4 * 24 and e["minibatch_samples"] == 24 for e in minibatches)
        assert len([e for e in selected if e["event"] == "ppo_epoch_kl"]) == 4
        assert len([e for e in selected if e["event"] == "ppo_update_summary"]) == 1
    checkpoint = torch.load(output / f"model_{iterations:05d}.pt", map_location="cpu", weights_only=True)
    saved_config = RealConfig(**checkpoint["config"])
    saved_config.validate()
    assert saved_config.schema == checkpoint["schema"] == "g1-estnet-ppo-clip-flat-v1"
    assert saved_config.num_envs == 4 and saved_config.horizon == 24
    assert saved_config.epochs == saved_config.minibatches == 4
    assert checkpoint["iteration"] == checkpoint["optimizer"]["updates"] == iterations
    assert set(checkpoint["optimizer"]) == {"optimizer", "updates", "total_optimizer_steps"}
    assert checkpoint["optimizer"]["total_optimizer_steps"] == expected_total
    states = checkpoint["optimizer"]["optimizer"]["state"]
    assert len(states) == 25
    assert all(float(s["step"]) == expected_total for s in states.values())
    assert all(torch.isfinite(s[name]).all() for s in states.values() for name in ("exp_avg", "exp_avg_sq"))
    assert any(torch.count_nonzero(s["exp_avg"]) for s in states.values())
    assert checkpoint["torch_rng"].device.type == "cpu" and "torch_cuda_rng" not in checkpoint
    return checkpoint, manifest


def test_real_main_collect_ppo_checkpoint_two_rounds_then_resume_only_third(integration, tmp_path):
    first = tmp_path / "train-two"
    assert integration.invoke(first, iterations=2) == 0
    checkpoint2, manifest2 = assert_complete_run(first, iterations=2, start=0, expected_total=32)
    assert integration.environments[0].steps == 2 * 24
    assert integration.environments[0].boundaries > 0
    assert integration.environments[0].pre_reset_distinct == integration.environments[0].boundaries
    assert any(torch.count_nonzero(action) for action in integration.environments[0].actions)
    resumed = tmp_path / "resume-to-three"
    assert integration.invoke(resumed, iterations=3, resume=first / "model_00002.pt") == 0
    checkpoint3, manifest3 = assert_complete_run(resumed, iterations=3, start=2, expected_total=48)
    assert integration.environments[1].steps == 24  # Target total3 is not another three rounds.
    assert checkpoint2["config"] == checkpoint3["config"]
    assert any(not torch.equal(checkpoint2["model"][key], checkpoint3["model"][key]) for key in checkpoint2["model"])
    resume = manifest3["resume"]
    assert (resume["start_iteration"], resume["target_iteration"], resume["additional_iterations"]) == (2, 3, 1)
    assert resume["restored_state"]["total_optimizer_steps"] == 32
    assert resume["restored_state"]["cpu_rng_restored"] is True
    assert resume["restored_state"]["resume_mode"] == "new_episodes"
    assert resume["restored_state"]["exact_resume"] is False
    assert resume["restored_state"]["environment_state_restored"] is False
    for output in (first, resumed):
        order = [name for name, path in integration.calls if path == output.resolve()]
        assert order == ["app_init", "env_init", "env_reset", "env_close", "app_close"]


def test_old_guard_schema_is_rejected_before_fake_isaac_initialization(integration, tmp_path):
    config = integration.small_config(num_envs=4)
    model = EstNet(config)
    ppo = PPO(model, config)
    path = tmp_path / "old-guard-schema.pt"
    run.save_checkpoint(path, model, ppo, config, 0, integration.asset)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["schema"] = payload["config"]["schema"] = "g1-estnet-klguard-flat-v1"
    payload["optimizer"]["total_accepted_steps"] = payload["optimizer"].pop("total_optimizer_steps")
    torch.save(payload, path)
    output = tmp_path / "reject-old"
    assert integration.invoke(output, iterations=1, resume=path) == 2
    assert integration.calls == [] and integration.environments == []
    result, manifest = read_json(output / "result.json"), read_json(output / "manifest.json")
    assert result["status"] == manifest["status"] == "failed"
    assert result["failure"]["phase"] == "cpu_preflight"
    assert result["failure"]["type"] == "ValueError"
    assert "schema" in result["failure"]["message"].lower()
    assert manifest["runtime"] == {"status": "not_initialized"}
    assert (output / "failure.txt").is_file()
    assert not (output / "metrics.jsonl").exists()
    assert not list(output.glob("model_*.pt"))
