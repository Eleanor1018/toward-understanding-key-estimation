"""CPU mechanism tests for fixed-LR, observation-only KL; no Isaac/GPU imports."""
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
PACKAGE = "fixed_lr_cpu_test_estnet"
spec = importlib.util.spec_from_file_location(PACKAGE, SOURCE / "__init__.py", submodule_search_locations=[str(SOURCE)])
module = importlib.util.module_from_spec(spec)
sys.modules[PACKAGE] = module
spec.loader.exec_module(module)
ppo_module = __import__(PACKAGE + ".ppo", fromlist=["PPO"])
PPO = ppo_module.PPO


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    torch.manual_seed(817)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def cfg(**overrides):
    values = dict(epochs=4, minibatches=4, learning_rate=5e-4, learning_rate_schedule="fixed",
                  clip=.2, value_coef=1., entropy_coef=.008, velocity_coef=1., max_grad_norm=1.,
                  kl_chunk_size=3, gradient_diagnostics=False)
    values.update(overrides)
    # Deliberately has no desired_kl, kl_hard_limit, or min/max_learning_rate.
    return SimpleNamespace(**values)


class TinyModel(nn.Module):
    """Supervised estimator bias genuinely moves actor mean through a fixed gain."""
    def __init__(self, gain=1., action_dim=1):
        super().__init__()
        self.action_dim = action_dim
        self.estimator = nn.Linear(1, 3)
        self.actor = nn.Linear(3, action_dim)
        self.critic = nn.Linear(1, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -3.))
        self.register_buffer("unused_finite_probe", torch.tensor(0.))
        with torch.no_grad():
            for net in (self.estimator, self.actor, self.critic):
                net.weight.zero_()
                net.bias.zero_()
            self.actor.weight[:, 0] = gain

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


def batch_for(model, count=17, advantages=None):
    history = torch.zeros(count, 1, 1)
    obs, command = torch.linspace(-1., 1., count).reshape(count, 1), torch.zeros(count, 1)
    with torch.no_grad():
        dist = model.distribution(history, obs, command)
        actions = dist.sample()
    return dict(history=history, obs=obs, command=command, critic=torch.zeros(count, 1),
                velocity=torch.tensor([[1., 0., 0.]]).expand(count, 3).clone(), action=actions,
                old_mean=dist.mean.detach().clone(), old_std=dist.stddev.detach().clone(),
                old_log_prob=dist.log_prob(actions).sum(-1).detach(), old_value=torch.zeros(count),
                returns=torch.zeros(count), advantages=torch.zeros(count) if advantages is None else advantages.clone())


def equal_tree(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def test_real_supervised_gradient_exceeds_old_kl_limit_and_still_runs_all_16_steps(monkeypatch):
    model = TinyModel(gain=100.)
    ppo = PPO(model, cfg(value_coef=0., entropy_coef=0.))
    data = batch_for(model)
    original_data = copy.deepcopy(data)
    events, measurements = [], []
    ppo.diagnostic_callback = events.append
    real_kl = ppo_module.full_rollout_kl

    def counted_kl(*args, **kwargs):
        result = real_kl(*args, **kwargs)
        measurements.append(result)
        return result

    monkeypatch.setattr(ppo_module, "full_rollout_kl", counted_kl)
    metrics = ppo.update(data)
    assert len(measurements) == 4  # No before-update, per-minibatch or fifth final pass.
    assert all(item["total"] > .02 for item in measurements)
    assert metrics["kl"] > 1.
    assert metrics["gradient_steps"] == metrics["optimizer_steps"] == 16
    assert ppo.updates == 1 and ppo.total_optimizer_steps == 16
    assert model.estimator.bias[0] > .007
    assert ppo.learning_rate == 5e-4
    assert len([e for e in events if e["event"] == "ppo_minibatch_update"]) == 16
    assert len([e for e in events if e["event"] == "ppo_epoch_kl"]) == 4
    assert len([e for e in events if e["event"] == "ppo_update_summary"]) == 1
    assert all(e["learning_rate"] == 5e-4 for e in events)
    assert metrics["policy_loss"] == 0.  # Jump is from actual supervised gradient.
    assert not any("accepted" in key or "rejected" in key or "retained" in key or "guard" in key for key in metrics)
    assert all(float(s["step"]) == 16 for s in ppo.state_dict()["optimizer"]["state"].values())
    equal_tree(data, original_data)
    json.dumps(events, allow_nan=False)


def test_full_model_adam_momentum_and_rng_resume_reproduces_next_16_updates():
    model = TinyModel()
    ppo = PPO(model, cfg())
    ppo.update(batch_for(model, advantages=torch.linspace(-1., 1., 17)))
    saved_model, saved_ppo = copy.deepcopy(model.state_dict()), copy.deepcopy(ppo.state_dict())
    assert any(torch.count_nonzero(s["exp_avg"]) for s in saved_ppo["optimizer"]["state"].values())
    data = batch_for(model, advantages=torch.linspace(1., -1., 17))
    rng = torch.get_rng_state().clone()
    expected = ppo.update(data)
    expected_model, expected_ppo = copy.deepcopy(model.state_dict()), copy.deepcopy(ppo.state_dict())
    resumed_model = TinyModel()
    resumed_model.load_state_dict(saved_model)
    resumed = PPO(resumed_model, cfg())
    resumed.load_state_dict(saved_ppo)
    equal_tree(resumed.state_dict(), saved_ppo)
    torch.set_rng_state(rng)
    actual = resumed.update(data)
    equal_tree(expected_model, resumed_model.state_dict())
    equal_tree(expected_ppo, resumed.state_dict())
    assert actual == expected
    assert resumed.updates == 2 and resumed.total_optimizer_steps == 32
    assert resumed.learning_rate == 5e-4


def test_actual_25_parameter_estnet_initializes_complete_adam_and_restores_16_steps():
    Config = __import__(PACKAGE + ".config", fromlist=["Config"]).Config
    config = Config()
    model = ppo_module.EstNet(config)
    ppo = PPO(model, config)
    count = 8
    history = torch.randn(count, config.history_steps, config.obs_dim) * .1
    obs, command = torch.randn(count, config.obs_dim) * .1, torch.randn(count, config.command_dim) * .1
    critic = torch.randn(count, config.critic_dim) * .1
    with torch.no_grad():
        distribution = model.distribution(history, obs, command)
        action = distribution.sample()
        old_value = model.value(critic)
    data = dict(history=history, obs=obs, command=command, critic=critic,
                velocity=torch.randn(count, 3) * .1, action=action,
                old_mean=distribution.mean.detach(), old_std=distribution.stddev.detach(),
                old_log_prob=distribution.log_prob(action).sum(-1).detach(), old_value=old_value,
                returns=old_value + torch.linspace(-.1, .1, count), advantages=torch.linspace(-1., 1., count))
    metrics = ppo.update(data)
    state = copy.deepcopy(ppo.state_dict())
    assert len(list(model.parameters())) == len(state["optimizer"]["state"]) == 25
    assert metrics["optimizer_steps"] == state["total_optimizer_steps"] == 16
    assert all(float(item["step"]) == 16 for item in state["optimizer"]["state"].values())
    restored = PPO(model, config)
    restored.load_state_dict(state)
    equal_tree(restored.state_dict(), state)


@pytest.mark.parametrize("corruption", ["legacy_counter", "wrong_count", "wrong_lr", "missing_adam", "wrong_adam_step", "nonfinite_moment"])
def test_resume_rejects_incompatible_or_corrupted_state(corruption):
    ppo = PPO(TinyModel(), cfg())
    ppo.update(batch_for(ppo.model))
    state = copy.deepcopy(ppo.state_dict())
    raw = state["optimizer"]
    first = next(iter(raw["state"]))
    if corruption == "legacy_counter":
        state["total_accepted_steps"] = state.pop("total_optimizer_steps")
    elif corruption == "wrong_count":
        state["total_optimizer_steps"] -= 1
    elif corruption == "wrong_lr":
        raw["param_groups"][0]["lr"] *= .5
    elif corruption == "missing_adam":
        del raw["state"][first]
    elif corruption == "wrong_adam_step":
        raw["state"][first]["step"].add_(1)
    else:
        raw["state"][first]["exp_avg"].fill_(float("nan"))
    fresh = PPO(TinyModel(), cfg())
    with pytest.raises((ValueError, FloatingPointError)):
        fresh.load_state_dict(state)
    assert fresh.updates == fresh.total_optimizer_steps == 0
    assert fresh.optimizer.state_dict()["state"] == {}


def test_fresh_empty_state_is_valid_and_undersized_rollout_is_not_silently_skipped():
    ppo = PPO(TinyModel(), cfg())
    state = ppo.state_dict()
    assert state == {"optimizer": ppo.optimizer.state_dict(), "updates": 0, "total_optimizer_steps": 0}
    ppo.load_state_dict(copy.deepcopy(state))
    with pytest.raises(ValueError, match="at least one sample"):
        ppo.update(batch_for(ppo.model, count=3))
    assert ppo.total_optimizer_steps == 0


def test_nonfinite_post_step_buffer_fails_without_rollback_or_success_summary(monkeypatch):
    ppo = PPO(TinyModel(), cfg(epochs=1))
    events = []
    ppo.diagnostic_callback = events.append
    real_step = ppo.optimizer.step

    def poison_after_real_step():
        real_step()
        ppo.model.unused_finite_probe.fill_(float("nan"))

    monkeypatch.setattr(ppo.optimizer, "step", poison_after_real_step)
    with pytest.raises(FloatingPointError, match="post-epoch"):
        ppo.update(batch_for(ppo.model))
    assert ppo.updates == 0 and ppo.total_optimizer_steps == 4
    assert torch.isnan(ppo.model.unused_finite_probe)
    assert not any(e["event"] == "ppo_update_summary" for e in events)
    with pytest.raises(ValueError, match="step count"):
        ppo.state_dict()  # Failure cannot become a resumable complete-rollout checkpoint.


def test_nonfinite_input_and_probability_ratio_fail_loudly_before_any_step():
    ppo = PPO(TinyModel(), cfg())
    bad = batch_for(ppo.model)
    bad["returns"][0] = float("nan")
    with pytest.raises(FloatingPointError, match="rollout"):
        ppo.update(bad)
    bad = batch_for(ppo.model)
    bad["old_log_prob"].fill_(-1e6)  # Finite input but exp overflows.
    with pytest.raises(FloatingPointError, match="probability ratio"):
        ppo.update(bad)
    assert ppo.updates == ppo.total_optimizer_steps == 0


def test_original_ratio_clip_clipped_value_and_combined_loss_are_preserved():
    config = cfg(epochs=1, minibatches=1, value_coef=.7, entropy_coef=.008)
    model = TinyModel()
    ppo = PPO(model, config)
    data = batch_for(model, count=2, advantages=torch.tensor([-1., 1.]))
    # Current/old probability ratios .5 and 2 with advantages -1 and +1.
    data["old_log_prob"] -= torch.log(torch.tensor([.5, 2.]))
    data["old_value"].fill_(1.)  # Current value0/return0 but clipped prediction .8.
    expected_entropy = model.distribution(data["history"], data["obs"], data["command"]).entropy().sum(-1).mean().item()
    metrics = ppo.update(data)
    assert metrics["policy_loss"] == pytest.approx(-.2, abs=1e-6)
    assert metrics["value_loss"] == pytest.approx(.64, abs=1e-6)
    assert metrics["clip_fraction"] == 1.
    assert metrics["velocity_loss"] == pytest.approx(1./3)
    assert metrics["loss"] == pytest.approx(-.2 + .7*.64 + 1./3 - .008*expected_entropy, abs=1e-6)


def test_optional_replay_is_once_before_step_and_gradient_events_are_json_safe():
    model = TinyModel()
    ppo = PPO(model, cfg(gradient_diagnostics=True))
    snapshots, events = [], []
    initial_model = copy.deepcopy(model.state_dict())
    data = batch_for(model, advantages=torch.arange(17).float())
    ppo.replay_callback, ppo.diagnostic_callback = snapshots.append, events.append
    ppo.update(data)
    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap["schema"] == "estnet-fixed-lr-first-minibatch-replay-v1"
    assert snap["total_optimizer_steps"] == 0 and snap["optimizer"]["state"] == {}
    assert snap["advantages_normalized_over_full_rollout"] is True
    equal_tree(snap["model"], initial_model)
    normalized = (data["advantages"] - data["advantages"].mean()) / (data["advantages"].std(correction=0) + 1e-8)
    assert torch.equal(snap["data"]["advantages"], normalized[snap["indices"]])
    assert all(v.device.type == "cpu" for v in snap["data"].values())
    assert all(e["gradient_diagnostics"] is not None for e in events if e["event"] == "ppo_minibatch_update")
    json.dumps(events, allow_nan=False)


def test_gae_keeps_timeout_bootstrap_but_stops_trace_at_both_episode_boundaries():
    rewards = torch.tensor([[1.], [2.], [3.]])
    values = torch.tensor([[10.], [20.], [30.]])
    next_values = torch.tensor([[11.], [99.], [200.]])
    term = torch.tensor([[False], [False], [True]])
    trunc = torch.tensor([[False], [True], [False]])
    adv, returns = ppo_module.compute_gae(rewards, values, next_values, term, trunc, gamma=.9, lam=.8)
    expected = torch.tensor([[1.+.9*11-10 + .9*.8*(2.+.9*99-20)], [2.+.9*99-20], [3.-30]])
    assert torch.allclose(adv, expected)
    assert torch.allclose(returns, expected+values)
