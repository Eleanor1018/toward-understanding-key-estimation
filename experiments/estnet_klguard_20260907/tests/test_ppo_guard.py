"""CPU mechanism/transaction tests; no Isaac imports and no historical-415 replay."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributions import Normal

SOURCE = Path(__file__).resolve().parents[1] / "estnet"
PACKAGE = "guard_cpu_test_estnet"
spec = importlib.util.spec_from_file_location(PACKAGE, SOURCE / "__init__.py", submodule_search_locations=[str(SOURCE)])
module = importlib.util.module_from_spec(spec)
sys.modules[PACKAGE] = module
spec.loader.exec_module(module)
ppo_module = __import__(PACKAGE + ".ppo", fromlist=["PPO"])
diagnostics_module = __import__(PACKAGE + ".ppo_diagnostics", fromlist=["full_rollout_kl"])
PPO, gaussian_kl = ppo_module.PPO, ppo_module.gaussian_kl
full_rollout_kl = diagnostics_module.full_rollout_kl


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    torch.manual_seed(715)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def config(**overrides):
    fields = dict(epochs=1, minibatches=1, learning_rate=.001, min_learning_rate=1e-5,
                  max_learning_rate=.1, desired_kl=0., clip=.2, value_coef=0.,
                  entropy_coef=0., velocity_coef=1., max_grad_norm=1.,
                  kl_hard_limit=.02, kl_chunk_size=3, gradient_diagnostics=True)
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TinyModel(nn.Module):
    """Velocity bias controls the Gaussian mean, exposing auxiliary-policy coupling."""
    def __init__(self, action_dim=1):
        super().__init__()
        self.action_dim = action_dim
        self.estimator = nn.Linear(1, 3)
        self.actor = nn.Linear(3, action_dim)
        self.critic = nn.Linear(1, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -3.))
        self.register_buffer("probe_buffer", torch.tensor(0.))
        with torch.no_grad():
            for network in (self.estimator, self.actor, self.critic):
                network.weight.zero_()
                network.bias.zero_()
            self.actor.weight[:, 0] = torch.linspace(1., .5, action_dim)

    def forward(self, history, obs, command):
        velocity = self.estimator(history.flatten(1))
        mean = self.actor(velocity) + .3 * obs[:, :1]
        return Normal(mean, self.log_std.clamp(-3., 1.).exp().expand_as(mean)), velocity

    def distribution(self, history, obs, command):
        return self(history, obs, command)[0]

    def value(self, critic):
        return self.critic(critic).squeeze(-1)

    @torch.no_grad()
    def clamp_std_(self):
        self.log_std.clamp_(-3., 1.)


def batch_for(model, count=8, advantages=None, velocity_target=1.):
    history, obs, command = torch.zeros(count, 1, 1), torch.linspace(-1., 1., count).reshape(count, 1), torch.zeros(count, 1)
    with torch.no_grad():
        dist = model.distribution(history, obs, command)
        actions = dist.sample()
        mean, std = dist.mean.clone(), dist.stddev.clone()
        logp = dist.log_prob(actions).sum(-1)
    return dict(history=history, obs=obs, command=command, critic=torch.zeros(count, 1),
                velocity=torch.tensor([[velocity_target, 0., 0.]]).expand(count, 3).clone(),
                action=actions, old_mean=mean, old_std=std, old_log_prob=logp,
                old_value=torch.zeros(count), returns=torch.zeros(count),
                advantages=torch.zeros(count) if advantages is None else advantages.clone())


def equal_tree(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_tree(a, b)
    else:
        assert left == right


def test_supervised_only_original_counterexample_is_rejected_and_stops_joint_update():
    model = TinyModel()
    ppo = PPO(model, config(learning_rate=.1, epochs=3, minibatches=4))
    original = copy.deepcopy(model.state_dict())
    events = []
    ppo.diagnostic_callback = events.append
    metrics = ppo.update(batch_for(model))
    equal_tree(original, model.state_dict())
    assert ppo.optimizer.state_dict()["state"] == {}
    assert ppo.learning_rate == pytest.approx(.1 / 1.5)
    assert metrics["policy_loss"] == 0.
    assert metrics["attempted_steps"] == metrics["rejected_steps"] == 1
    assert metrics["accepted_steps"] == metrics["gradient_steps"] == 0
    assert metrics["candidate_kl"] > .02
    assert metrics["retained_kl"] == 0.
    assert ppo.updates == 1 and ppo.total_accepted_steps == 0
    attempts = [event for event in events if event["event"] == "ppo_minibatch_guard"]
    assert len(attempts) == 1
    assert attempts[0]["candidate_kl"]["mean_per_joint"][0] > .02
    assert not any(event["event"] == "ppo_epoch_schedule" for event in events)
    json.dumps(events, allow_nan=False)


def test_rejection_restores_existing_adam_momentum_steps_and_all_model_state():
    model = TinyModel()
    ppo = PPO(model, config())
    assert ppo.update(batch_for(model))["accepted_steps"] == 1
    assert ppo.optimizer.state_dict()["state"]
    assert ppo.total_accepted_steps == 1
    for group in ppo.optimizer.param_groups:
        group["lr"] = .1
    before_model, before_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(ppo.optimizer.state_dict())
    metrics = ppo.update(batch_for(model))
    assert metrics["accepted_steps"] == 0 and metrics["rejected_steps"] == 1
    equal_tree(before_model, model.state_dict())
    expected_optimizer = copy.deepcopy(before_optimizer)
    for group in expected_optimizer["param_groups"]:
        group["lr"] /= 1.5
    equal_tree(expected_optimizer, ppo.optimizer.state_dict())
    assert ppo.total_accepted_steps == 1 and ppo.updates == 2
    for state in ppo.optimizer.state_dict()["state"].values():
        assert state["step"].item() == 1


def test_normal_candidate_accepts_all_minibatches_and_tracks_real_steps():
    model = TinyModel()
    ppo = PPO(model, config(epochs=2, minibatches=2))
    data = batch_for(model, count=9)
    original_batch = copy.deepcopy(data)
    events = []
    ppo.diagnostic_callback = events.append
    metrics = ppo.update(data)
    assert metrics["accepted_steps"] == metrics["attempted_steps"] == metrics["gradient_steps"] == 4
    assert metrics["rejected_steps"] == 0
    assert 0 < metrics["retained_kl"] <= ppo.kl_hard_limit
    assert ppo.total_accepted_steps == 4
    for event in events:
        if "retained_kl" in event:
            assert event["retained_kl"]["total"] <= ppo.kl_hard_limit
    equal_tree(data, original_batch)
    for state in ppo.optimizer.state_dict()["state"].values():
        assert state["step"].item() == 4
    json.dumps(events, allow_nan=False)


def test_later_rejection_preserves_earlier_accepted_step_and_stops(monkeypatch):
    model = TinyModel()
    ppo = PPO(model, config(epochs=2, minibatches=4))
    original_step = ppo.optimizer.step
    captured = {}
    calls = 0

    def second_step_jumps():
        nonlocal calls
        calls += 1
        if calls == 2:
            captured["model"] = copy.deepcopy(model.state_dict())
            captured["optimizer"] = copy.deepcopy(ppo.optimizer.state_dict())
        original_step()
        if calls == 2:
            with torch.no_grad():
                model.estimator.bias[0].add_(1.)

    monkeypatch.setattr(ppo.optimizer, "step", second_step_jumps)
    metrics = ppo.update(batch_for(model))
    assert calls == metrics["attempted_steps"] == 2
    assert metrics["accepted_steps"] == metrics["rejected_steps"] == 1
    equal_tree(captured["model"], model.state_dict())
    for group in captured["optimizer"]["param_groups"]:
        group["lr"] = max(1e-5, group["lr"] / 1.5)
    equal_tree(captured["optimizer"], ppo.optimizer.state_dict())
    assert ppo.total_accepted_steps == 1 and ppo.updates == 1


def test_nonfinite_candidate_restores_buffers_parameters_and_adam(monkeypatch):
    model = TinyModel()
    ppo = PPO(model, config())
    assert ppo.update(batch_for(model))["accepted_steps"] == 1
    before_model, before_adam = copy.deepcopy(model.state_dict()), copy.deepcopy(ppo.optimizer.state_dict())
    original_step = ppo.optimizer.step

    def corrupting_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        with torch.no_grad():
            model.probe_buffer.add_(123)
            model.estimator.bias[0] = float("nan")
        return result

    monkeypatch.setattr(ppo.optimizer, "step", corrupting_step)
    events = []
    ppo.diagnostic_callback = events.append
    metrics = ppo.update(batch_for(model))
    equal_tree(before_model, model.state_dict())
    for group in before_adam["param_groups"]:
        group["lr"] = max(1e-5, group["lr"] / 1.5)
    equal_tree(before_adam, ppo.optimizer.state_dict())
    assert metrics["candidate_kl"] is None and metrics["candidate_kl_finite"] == 0.
    assert metrics["retained_kl"] == 0.
    assert metrics["rejected_steps"] == 1 and ppo.total_accepted_steps == 1
    json.dumps(events, allow_nan=False)
    json.dumps(metrics, allow_nan=False)


def test_chunked_full_rollout_kl_matches_whole_distribution_per_joint(monkeypatch):
    model = TinyModel(action_dim=2)
    data = batch_for(model, count=11)
    before_data = copy.deepcopy(data)
    with torch.no_grad():
        model.estimator.bias[0] += .003
        model.log_std += torch.tensor([.04, .09])
    before_model = copy.deepcopy(model.state_dict())
    current = model.distribution(data["history"], data["obs"], data["command"])
    expected = gaussian_kl(data["old_mean"], data["old_std"], current.mean, current.stddev).mean().item()
    observed_sizes = []
    original = model.distribution

    def observed(history, obs, command):
        observed_sizes.append(len(history))
        return original(history, obs, command)

    monkeypatch.setattr(model, "distribution", observed)
    result = full_rollout_kl(model, data, chunk_size=3)
    assert observed_sizes == [3, 3, 3, 2]
    assert result["samples"] == 11
    assert result["total"] == pytest.approx(expected, abs=1e-7)
    assert sum(result["per_joint"]) == pytest.approx(result["total"])
    assert all(value > 0 for value in result["mean_per_joint"] + result["variance_per_joint"])
    for index in range(2):
        assert result["per_joint"][index] == pytest.approx(result["mean_per_joint"][index] + result["variance_per_joint"][index])
    equal_tree(data, before_data)
    equal_tree(model.state_dict(), before_model)


def test_nonfinite_adam_candidate_is_rejected_even_if_policy_is_finite(monkeypatch):
    model = TinyModel()
    ppo = PPO(model, config())
    before_model = copy.deepcopy(model.state_dict())
    original_step = ppo.optimizer.step

    def corrupting_step():
        original_step()
        for state in ppo.optimizer.state.values():
            state["exp_avg_sq"].fill_(float("inf"))

    monkeypatch.setattr(ppo.optimizer, "step", corrupting_step)
    events = []
    ppo.diagnostic_callback = events.append
    result = ppo.update(batch_for(model))
    assert result["rejected_steps"] == 1 and result["candidate_kl_finite"] == 1
    equal_tree(before_model, model.state_dict())
    assert ppo.optimizer.state_dict()["state"] == {}
    assert events[0]["rejection_reason"] == "nonfinite_optimizer_state"
    assert events[0]["candidate_optimizer_state_finite"] is False


def test_gradient_probes_preserve_update_and_report_module_alignment():
    model_a, model_b = TinyModel(), TinyModel()
    model_b.load_state_dict(model_a.state_dict())
    cfg_a = config(learning_rate=.00001, value_coef=1., entropy_coef=.008, gradient_diagnostics=True)
    cfg_b = config(**{**vars(cfg_a), "gradient_diagnostics": False})
    a, b = PPO(model_a, cfg_a), PPO(model_b, cfg_b)
    data = batch_for(model_a, advantages=torch.linspace(-1., 1., 8))
    data["returns"].fill_(1.)
    events = []
    a.diagnostic_callback = events.append
    torch.manual_seed(800)
    a.update(data)
    torch.manual_seed(800)
    b.update(data)
    equal_tree(model_a.state_dict(), model_b.state_dict())
    equal_tree(a.optimizer.state_dict(), b.optimizer.state_dict())
    grad = events[0]["gradient_diagnostics"]
    norms = grad["raw_term_preclip_norms"]
    assert norms["policy"]["actor"] > 0 and norms["policy"]["estimator"] > 0
    assert norms["value"]["critic"] > 0 and norms["value"]["actor"] == 0
    assert norms["velocity"]["estimator"] > 0 and norms["velocity"]["actor"] == 0
    assert norms["entropy"]["log_std"] > 0
    assert grad["supervised_vs_ppo"]["estimator"]["cosine"] is not None
    json.dumps(grad, allow_nan=False)


def test_replay_snapshot_is_cpu_independent_normalized_and_pre_step():
    model = TinyModel()
    ppo = PPO(model, config(learning_rate=.00001, epochs=2, minibatches=3))
    data = batch_for(model, count=8, advantages=torch.arange(8, dtype=torch.float32))
    initial_model = copy.deepcopy(model.state_dict())
    snapshots = []
    ppo.replay_callback = snapshots.append
    ppo.update(data)
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["advantages_normalized_over_full_rollout"] is True
    assert snapshot["full_rollout_count"] == 8
    assert snapshot["epoch"] == snapshot["minibatch"] == snapshot["iteration"] == 1
    assert snapshot["optimizer"]["state"] == {}
    equal_tree(snapshot["model"], initial_model)
    indices = snapshot["indices"]
    expected = (data["advantages"] - data["advantages"].mean()) / (data["advantages"].std(correction=0) + 1e-8)
    assert torch.equal(snapshot["data"]["advantages"], expected[indices])
    assert torch.equal(snapshot["data"]["action"], data["action"][indices])
    for tensor in snapshot["data"].values():
        assert tensor.device.type == "cpu" and not tensor.requires_grad
    before_action = snapshot["data"]["action"].clone()
    data["action"].fill_(123)
    assert torch.equal(snapshot["data"]["action"], before_action)


def test_resume_counters_separate_rollouts_from_accepted_steps():
    model = TinyModel()
    ppo = PPO(model, config(learning_rate=.1))
    ppo.update(batch_for(model))
    state = copy.deepcopy(ppo.state_dict())
    restored = PPO(TinyModel(), config(learning_rate=.1))
    restored.load_state_dict(state)
    assert restored.updates == 1 and restored.total_accepted_steps == 0
    equal_tree(restored.state_dict(), state)
    del state["total_accepted_steps"]
    with pytest.raises(ValueError, match="total_accepted_steps"):
        restored.load_state_dict(state)


def test_rejection_lr_respects_original_floor(monkeypatch):
    model = TinyModel()
    ppo = PPO(model, config(learning_rate=1e-5))
    original_step = ppo.optimizer.step

    def forced_jump():
        original_step()
        with torch.no_grad():
            model.estimator.bias[0].add_(1.)

    monkeypatch.setattr(ppo.optimizer, "step", forced_jump)
    assert ppo.update(batch_for(model))["rejected_steps"] == 1
    assert ppo.learning_rate == 1e-5


def test_preexisting_kl_violation_does_not_attempt_optimizer_step(monkeypatch):
    model = TinyModel()
    ppo = PPO(model, config())
    data = batch_for(model)
    data["old_mean"].add_(10.)
    monkeypatch.setattr(ppo.optimizer, "step", lambda: pytest.fail("must not attempt a step"))
    with pytest.raises(FloatingPointError, match="Pre-update"):
        ppo.update(data)
    assert ppo.updates == ppo.total_accepted_steps == 0
