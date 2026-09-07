"""五路线真实辅助梯度和固定LR PPO CPU测试；不初始化Isaac或CUDA。"""
import copy
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

SOURCE = Path(__file__).resolve().parents[1] / "estnet"
PACKAGE = "variant_ppo_cpu_estnet"
spec = importlib.util.spec_from_file_location(PACKAGE, SOURCE / "__init__.py", submodule_search_locations=[str(SOURCE)])
package = importlib.util.module_from_spec(spec)
sys.modules[PACKAGE] = package
spec.loader.exec_module(package)
factory = __import__(PACKAGE + ".factory", fromlist=["build_model"])
ppo_module = __import__(PACKAGE + ".ppo", fromlist=["PPO"])
VARIANTS = ("key1", "key2", "fullest", "irrest", "implicit")
SUPERVISION = {"estnet": ("velocity",), "key1": ("velocity",),
               "key2": ("velocity", "heightmap"),
               "fullest": ("velocity", "heightmap", "body_height"),
               "irrest": ("body_height",), "implicit": ()}


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    torch.manual_seed(829)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def config(variant, **kwargs):
    return replace(factory.config_for_variant(variant),
                   **{**dict(num_envs=4, encoder_hidden=(12, 8), actor_hidden=(12, 8),
                             critic_hidden=(12, 8), decoder_hidden=(8, 12), kl_chunk_size=5), **kwargs})


def batch_for(model, cfg, count=17):
    history = torch.randn(count, cfg.history_steps, cfg.obs_dim) * .1
    obs = torch.randn(count, cfg.obs_dim) * .2 + .3
    command = torch.randn(count, cfg.command_dim) * .1
    critic = torch.randn(count, cfg.critic_dim) * .1
    with torch.no_grad():
        distribution = model.distribution(history, obs, command)
        action = distribution.sample()
        value = model.value(critic)
    data = dict(history=history, obs=obs, command=command, critic=critic,
                action=action, old_mean=distribution.mean.detach().clone(),
                old_std=distribution.stddev.detach().clone(),
                old_log_prob=distribution.log_prob(action).sum(-1).detach(), old_value=value,
                returns=value + torch.linspace(-.1, .1, count), advantages=torch.linspace(-1., 1., count))
    for name, width in cfg.supervision_dims.items():
        data[name] = torch.randn(count, width) * .05 + {"velocity": .2, "heightmap": .4, "body_height": .78}[name]
    return data


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


def nonzero_grads(module):
    return any(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) for p in module.parameters())


@pytest.mark.parametrize("variant", ("estnet",) + VARIANTS)
def test_all16_fixed_lr_exact_heads_decoder_and_complete_adam_resume(variant):
    cfg = config(variant)
    model, events = factory.build_model(cfg), []
    ppo = factory.build_ppo(model, cfg)
    ppo.diagnostic_callback = events.append
    data = batch_for(model, cfg)
    original_data = copy.deepcopy(data)
    original_weights = copy.deepcopy(model.state_dict())
    metrics = ppo.update(data)
    assert model.explicit_names == SUPERVISION[variant]
    assert set(name for name in SUPERVISION["fullest"] if name + "_loss" in metrics) == set(SUPERVISION[variant])
    assert ("velocity_rmse" in metrics) == model.has_velocity_estimate
    assert metrics["kl"] == metrics["policy_kl"]
    assert metrics["learning_rate"] == 5e-4
    assert metrics["optimizer_steps"] == metrics["gradient_steps"] == 16
    assert not any("accepted" in k or "rejected" in k or "retained" in k for k in metrics)
    assert len([event for event in events if event["event"] == "ppo_minibatch_update"]) == 16
    assert len([event for event in events if event["event"] == "ppo_epoch_kl"]) == 4
    assert all(event["learning_rate"] == 5e-4 for event in events)
    assert all(e["policy_kl"] == e["kl"] for e in events if e["event"] == "ppo_epoch_kl")
    if variant != "estnet":
        assert metrics["latent_kl"] > 0 and metrics["prediction_loss"] > 0
        assert metrics["latent_kl_weighted"] == pytest.approx(50 * metrics["latent_kl"], rel=1e-6)
        expected_aux = 2 * metrics["prediction_loss"] + 50 * metrics["latent_kl"]
        assert any(not torch.equal(value, original_weights[name]) for name, value in model.state_dict().items() if name.startswith("decoder."))
    else:
        assert "latent_kl" not in metrics and "prediction_loss" not in metrics
        expected_aux = 0.
    expected_aux += sum(getattr(cfg, name + "_coef") * metrics[name + "_loss"] for name in SUPERVISION[variant])
    assert metrics["auxiliary_loss"] == pytest.approx(expected_aux, rel=1e-6)
    assert metrics["loss"] == pytest.approx(metrics["policy_loss"] + metrics["value_loss"]
                                            - .008 * metrics["entropy"] + metrics["auxiliary_loss"], abs=1e-6)
    equal_tree(data, original_data)
    state, weights = copy.deepcopy(ppo.state_dict()), copy.deepcopy(model.state_dict())
    assert state["updates"] == 1 and state["total_optimizer_steps"] == 16
    assert len(state["optimizer"]["state"]) == len(list(model.parameters()))
    assert all(float(moment["step"]) == 16 for moment in state["optimizer"]["state"].values())
    assert all(torch.isfinite(moment[key]).all() for moment in state["optimizer"]["state"].values() for key in ("exp_avg", "exp_avg_sq"))
    # 保留真实decoder重参数采样；恢复同一CPU RNG后下一轮逐张量相同。
    second = batch_for(model, cfg)
    rng = torch.get_rng_state().clone()
    expected = ppo.update(second)
    expected_weights, expected_state = copy.deepcopy(model.state_dict()), copy.deepcopy(ppo.state_dict())
    restored_model = factory.build_model(cfg)
    restored_model.load_state_dict(weights)
    restored = factory.build_ppo(restored_model, cfg)
    restored.load_state_dict(state)
    torch.set_rng_state(rng)
    actual = restored.update(second)
    assert actual == expected
    equal_tree(restored_model.state_dict(), expected_weights)
    equal_tree(restored.state_dict(), expected_state)
    assert restored.updates == 2 and restored.total_optimizer_steps == 32
    json.dumps(events, allow_nan=False)


@pytest.mark.parametrize("variant", VARIANTS)
def test_actual_auxiliary_graph_updates_decoder_encoder_and_only_existing_heads(variant):
    cfg = config(variant)
    model = factory.build_model(cfg).eval()  # Deterministic decoder for analytic comparison.
    ppo = factory.build_ppo(model, cfg)
    data = batch_for(model, cfg)
    indices = torch.arange(17)
    data["next_obs"] = data["obs"] + 100.  # Must never become this experiment's target.
    auxiliary = model.auxiliary(data["history"], data["obs"], data["command"])
    weighted, metrics = ppo.auxiliary_loss(data, indices, None)
    assert torch.allclose(metrics["prediction_loss"], F.mse_loss(auxiliary["prediction"], data["obs"]))
    assert not torch.allclose(metrics["prediction_loss"], F.mse_loss(auxiliary["prediction"], data["next_obs"]))
    for name in SUPERVISION[variant]:
        assert torch.allclose(metrics[name + "_loss"], F.mse_loss(auxiliary[name], data[name]))
    weighted.backward()
    for module in (model.encoder, model.decoder, model.mu_head, model.logvar_head):
        assert nonzero_grads(module)
    for name in SUPERVISION[variant]:
        assert nonzero_grads(getattr(model, name + "_head"))
    for name in set(SUPERVISION["fullest"]) - set(SUPERVISION[variant]):
        assert not hasattr(model, name + "_head") and name + "_loss" not in metrics


@pytest.mark.parametrize("variant", VARIANTS)
def test_latent_kl_beta50_mean_reduction_has_real_mu_and_logvar_gradients(variant):
    cfg = config(variant, prediction_coef=0., velocity_coef=0., heightmap_coef=0., body_height_coef=0.)
    model = factory.build_model(cfg).eval()
    ppo = factory.build_ppo(model, cfg)
    with torch.no_grad():
        model.mu_head.weight.zero_(); model.mu_head.bias.fill_(.4)
        model.logvar_head.weight.zero_(); model.logvar_head.bias.fill_(.2)
    data = batch_for(model, cfg)
    weighted, metrics = ppo.auxiliary_loss(data, torch.arange(17), None)
    expected_kl = .5 * (.4**2 + math.exp(.2) - 1 - .2)
    assert metrics["latent_kl"].item() == pytest.approx(expected_kl, abs=1e-6)
    assert weighted.item() == pytest.approx(50 * expected_kl, abs=1e-5)
    weighted.backward()
    assert torch.allclose(model.mu_head.bias.grad, torch.full_like(model.mu_head.bias, 50*.4/16), atol=1e-6)
    assert torch.allclose(model.logvar_head.bias.grad, torch.full_like(model.logvar_head.bias, 50*.5*(math.exp(.2)-1)/16), atol=1e-6)


def test_real_latent_kl_gradient_can_move_policy_above_point02_without_any_skip_or_lr_change(monkeypatch):
    cfg = config("key1", encoder_hidden=(4,), actor_hidden=(4,), critic_hidden=(4,), decoder_hidden=(4,),
                 prediction_coef=0., velocity_coef=0., entropy_coef=0., value_coef=0.)
    model = factory.build_model(cfg)
    with torch.no_grad():
        for p in model.encoder.parameters(): p.zero_()
        for head in (model.velocity_head, model.mu_head, model.logvar_head):
            head.weight.zero_(); head.bias.zero_()
        model.mu_head.bias.fill_(.4)
        model.logvar_head.bias.fill_(.2)
        for p in model.actor.parameters(): p.zero_()
        model.actor[0].weight[0, cfg.obs_dim + cfg.command_dim + model.explicit_dim] = 1.
        model.actor[-1].weight[:, 0] = 100.
    ppo = factory.build_ppo(model, cfg)
    data = batch_for(model, cfg)
    data["advantages"].zero_()  # No policy, value, entropy or explicit/reconstruction gradients.
    calls = []
    actual_kl = ppo_module.full_rollout_kl
    def measured(*args, **kwargs):
        result = actual_kl(*args, **kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(ppo_module, "full_rollout_kl", measured)
    result = ppo.update(data)
    assert len(calls) == 4 and all(kl["total"] > .02 for kl in calls)
    assert result["policy_kl"] > 1. and result["latent_kl"] > 0.
    assert model.mu_head.bias[0] < .393
    assert result["optimizer_steps"] == 16 and ppo.total_optimizer_steps == 16
    assert ppo.learning_rate == 5e-4


@pytest.mark.parametrize("variant,field,bad_shape", [
    ("key2", "heightmap", (17, 1)), ("fullest", "heightmap", (17, 81)),
    ("fullest", "body_height", (17,)), ("irrest", "body_height", (17, 81)),
    ("key1", "velocity", (17, 1)),
])
def test_wrong_supervision_shapes_never_broadcast_or_apply_an_adam_step(variant, field, bad_shape):
    cfg = config(variant)
    model = factory.build_model(cfg)
    ppo = factory.build_ppo(model, cfg)
    data = batch_for(model, cfg)
    data[field] = torch.zeros(bad_shape)
    with pytest.raises(ValueError):
        ppo.update(data)
    assert ppo.updates == ppo.total_optimizer_steps == 0
    assert not ppo.optimizer.state_dict()["state"]


def test_joint_auxiliary_gradient_diagnostics_never_mislabels_latent_or_decoder_as_velocity():
    cfg = config("implicit", gradient_diagnostics=True)
    model = factory.build_model(cfg)
    ppo = factory.build_ppo(model, cfg)
    events = []
    ppo.diagnostic_callback = events.append
    metrics = ppo.update(batch_for(model, cfg))
    assert "velocity_loss" not in metrics
    for event in events:
        if event["event"] != "ppo_minibatch_update": continue
        d = event["gradient_diagnostics"]
        assert "velocity" not in d["objective_weights"]
        assert d["objective_weights"]["auxiliary"] == 1.
        assert d["raw_term_preclip_norms"]["auxiliary"]["estimator"] > 0
        assert d["raw_term_preclip_norms"]["auxiliary"]["other"] > 0  # Actual decoder.
        assert "auxiliary_vs_ppo" in d and "auxiliary_norm" in d["auxiliary_vs_ppo"]["estimator"]
    json.dumps(events, allow_nan=False)


@pytest.mark.parametrize("corruption", ("legacy", "wrong_total", "missing_decoder", "wrong_step", "wrong_lr"))
def test_variant_resume_requires_all_adam_states_and_exact_fixed_lr_counters(corruption):
    cfg = config("key2")
    model = factory.build_model(cfg)
    ppo = factory.build_ppo(model, cfg)
    ppo.update(batch_for(model, cfg))
    state = copy.deepcopy(ppo.state_dict())
    raw = state["optimizer"]
    decoder_index = next(i for i, (name, _) in enumerate(model.named_parameters()) if name.startswith("decoder."))
    decoder_id = raw["param_groups"][0]["params"][decoder_index]
    if corruption == "legacy": state["total_accepted_steps"] = state.pop("total_optimizer_steps")
    elif corruption == "wrong_total": state["total_optimizer_steps"] = 15
    elif corruption == "missing_decoder": del raw["state"][decoder_id]
    elif corruption == "wrong_step": raw["state"][decoder_id]["step"].sub_(1)
    else: raw["param_groups"][0]["lr"] *= .5
    with pytest.raises(ValueError):
        factory.build_ppo(factory.build_model(cfg), cfg).load_state_dict(state)
