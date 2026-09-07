"""Read-only full-policy KL and loss-gradient diagnostics for a short PPO audit."""
from __future__ import annotations

import copy
import math
from typing import Any

import torch


def cpu_clone_tree(value: Any) -> Any:
    """Independent CPU snapshot, retaining mapping metadata where present."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        result = copy.copy(value)
        for key, item in value.items():
            result[key] = cpu_clone_tree(item)
        if hasattr(value, "_metadata"):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, list):
        return [cpu_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def finite_tensor_tree(value: Any) -> bool:
    """Check candidate Adam moments/counters too, without copying them to CPU."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(finite_tensor_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensor_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


@torch.no_grad()
def full_rollout_kl(model, data: dict[str, torch.Tensor], chunk_size: int = 2048) -> dict[str, Any]:
    """Exact KL(old || current estimator->actor), all samples, bounded forward chunks.

    Accumulation and Gaussian arithmetic use float64; no rollout tensor is changed.
    A non-finite candidate is represented by JSON nulls, never NaN/Infinity.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("KL chunk_size must be a positive integer")
    count, dims = data["old_mean"].shape
    if count < 1 or data["old_std"].shape != (count, dims):
        raise ValueError("KL expects nonempty old_mean/old_std of shape [N, action_dim]")

    def invalid(reason):
        return {"finite": False, "total": None, "mean_per_joint": None,
                "variance_per_joint": None, "per_joint": None, "samples": count,
                "reason": reason}

    # Covers candidate critic/estimator parameters and buffers as well as policy outputs.
    for tensor in model.state_dict().values():
        if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all().item():
            return invalid("nonfinite_model_state")
    mean_sum = torch.zeros(dims, dtype=torch.float64, device=data["old_mean"].device)
    variance_sum = torch.zeros_like(mean_sum)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        try:
            current = model.distribution(data["history"][start:stop], data["obs"][start:stop],
                                         data["command"][start:stop])
        except ValueError as error:
            # torch.distributions.Normal validates a non-finite mean/scale by raising.
            return invalid("invalid_candidate_distribution: " + str(error)[:160])
        old_mean = data["old_mean"][start:stop].double()
        old_std = data["old_std"][start:stop].double()
        new_mean, new_std = current.mean.double(), current.stddev.double()
        if new_mean.shape != old_mean.shape or new_std.shape != old_std.shape:
            raise ValueError("Candidate policy has an incompatible Gaussian shape")
        if (not all(torch.isfinite(t).all().item() for t in (old_mean, old_std, new_mean, new_std))
                or not (old_std > 0).all().item() or not (new_std > 0).all().item()):
            return invalid("nonfinite_or_nonpositive_policy_distribution")
        mean_term = (old_mean - new_mean).square() / (2.0 * new_std.square())
        variance_term = (torch.log(new_std / old_std) + old_std.square() /
                         (2.0 * new_std.square()) - 0.5).clamp_min(0.0)
        if not torch.isfinite(mean_term).all().item() or not torch.isfinite(variance_term).all().item():
            return invalid("nonfinite_gaussian_kl")
        mean_sum += mean_term.sum(0)
        variance_sum += variance_term.sum(0)
    mean_joint = mean_sum / count
    variance_joint = variance_sum / count
    joint = mean_joint + variance_joint
    total = joint.sum().item()
    if not math.isfinite(total):
        return invalid("nonfinite_accumulated_kl")
    return {"finite": True, "total": total, "mean_per_joint": mean_joint.cpu().tolist(),
            "variance_per_joint": variance_joint.cpu().tolist(), "per_joint": joint.cpu().tolist(),
            "samples": count, "reason": None, "coordinate_order": "policy_action_columns"}


def loss_gradient_diagnostics(model, terms: dict[str, torch.Tensor],
                              weights: dict[str, float]) -> dict[str, Any]:
    """Autograd probes do not populate/modify .grad or change the loss graph.

    Norms describe gradient scale, not causal attribution or Adam parameter steps.
    Entropy is reported as positive entropy; its objective weight is negative.
    """
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters = [parameter for _, parameter in named]
    if not parameters:
        raise ValueError("Gradient diagnostics require trainable parameters")
    groups = {name: [] for name in ("actor", "critic", "estimator", "log_std", "other")}
    for index, (name, _) in enumerate(named):
        group = next((group for group in groups if name == group or name.startswith(group + ".")), "other")
        groups[group].append(index)

    gradients = {}
    for name, term in terms.items():
        gradients[name] = (torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
                           if term.requires_grad else (None,) * len(parameters))

    def norm(grads, indices):
        squares = [grads[index].detach().double().square().sum() for index in indices if grads[index] is not None]
        return torch.stack(squares).sum().sqrt().item() if squares else 0.0

    all_indices = list(range(len(parameters)))
    reported_groups = {**groups, "all": all_indices}
    raw_norms = {term: {group: norm(grads, indices) for group, indices in reported_groups.items()}
                 for term, grads in gradients.items()}
    weighted_norms = {term: {group: abs(weights[term]) * value for group, value in values.items()}
                      for term, values in raw_norms.items()}
    ppo_grads = []
    supervised_grads = []
    for index in all_indices:
        pieces = [weights[name] * gradients[name][index] for name in ("policy", "value", "entropy")
                  if gradients[name][index] is not None]
        ppo_grads.append(sum(pieces) if pieces else None)
        grad = gradients["velocity"][index]
        supervised_grads.append(None if grad is None else weights["velocity"] * grad)

    alignment = {}
    for group in ("estimator", "all"):
        indices = reported_groups[group]
        ppo_norm, supervised_norm = norm(ppo_grads, indices), norm(supervised_grads, indices)
        dots = [(ppo_grads[i].detach().double() * supervised_grads[i].detach().double()).sum()
                for i in indices if ppo_grads[i] is not None and supervised_grads[i] is not None]
        dot = torch.stack(dots).sum().item() if dots else 0.0
        cosine = (max(-1.0, min(1.0, dot / (ppo_norm * supervised_norm)))
                  if ppo_norm > 0 and supervised_norm > 0 else None)
        alignment[group] = {"ppo_norm": ppo_norm, "supervised_norm": supervised_norm,
                            "cosine": cosine, "angle_degrees": None if cosine is None else math.degrees(math.acos(cosine))}
    result = {"raw_term_preclip_norms": raw_norms, "weighted_term_preclip_norms": weighted_norms,
              "objective_weights": weights, "supervised_vs_ppo": alignment,
              "meaning": "Gradient scales/alignment only; not causal attribution or Adam update magnitudes."}

    def json_finite(value):
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {key: json_finite(item) for key, item in value.items()}
        return value
    return json_finite(result)
