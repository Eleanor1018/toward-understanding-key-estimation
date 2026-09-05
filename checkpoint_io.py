"""Restricted checkpoint loading for repository-generated PyTorch states."""

from __future__ import annotations

from pathlib import Path, PosixPath
from typing import Any

import torch


def load_checkpoint(
    path: Path | str,
    map_location: torch.device | str,
) -> dict[str, Any]:
    """Load tensors and basic containers without executing arbitrary pickle code."""

    with torch.serialization.safe_globals([PosixPath]):
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint root must be a dictionary")
    return checkpoint
