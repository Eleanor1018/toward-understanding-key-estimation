"""Phase-free contact-state helpers shared by training and evaluation."""

from __future__ import annotations

import torch


def classify_foot_landings(
    first_contact: torch.Tensor,
    last_landing_foot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify single-foot landings as first, alternating, or repeated."""

    if first_contact.ndim != 2 or first_contact.shape[1] != 2:
        raise ValueError("first_contact must have shape [N, 2]")
    if last_landing_foot.shape != first_contact.shape[:1]:
        raise ValueError("last_landing_foot must have shape [N]")

    single_landing = first_contact.to(torch.int64).sum(dim=-1) == 1
    landing_foot = first_contact.to(torch.int64).argmax(dim=-1)
    previous_landing_known = last_landing_foot >= 0
    alternating_landing = torch.logical_and(
        torch.logical_and(single_landing, previous_landing_known),
        landing_foot != last_landing_foot,
    )
    repeated_landing = torch.logical_and(
        torch.logical_and(single_landing, previous_landing_known),
        landing_foot == last_landing_foot,
    )
    next_last_landing_foot = torch.where(
        single_landing,
        landing_foot,
        last_landing_foot,
    )
    return (
        single_landing,
        alternating_landing,
        repeated_landing,
        next_last_landing_foot,
    )
