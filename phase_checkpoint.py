"""Migrate legacy 3-D-command checkpoints to the visible-phase contract.

The two phase features are inserted after the physical ``(vx, vy, yaw_rate)``
command. Their first-layer weights start at zero, so migration preserves the
legacy actor and critic functions while making room for phase learning.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from checkpoint_io import load_checkpoint
from normalization import LEGACY_NORMALIZATION_TYPE, NORMALIZATION_TYPE


MIGRATION_TYPE = "command_phase_3_to_5_v1"
LEGACY_COMMAND_DIM = 3
TARGET_COMMAND_DIM = 5
PHASE_INSERTION_INDEX = 96
ACTOR_FIRST_LAYER_SHAPE = (2048, 115)
CRITIC_FIRST_LAYER_SHAPE = (2048, 199)
FIRST_LAYER_KEY = "trunk.0.weight"
REQUIRED_MODEL_KEYS = (
    "encoder",
    "decoder",
    "actor",
    "critic",
    "action_distribution",
)
EXPECTED_OPTIMIZER_GROUPS = ("policy", "critic", "decoder")
EXPECTED_ACTION_DISTRIBUTION_TYPE = "tanh_diagonal_gaussian_v1"
EXPECTED_AUXILIARY_OBJECTIVE_TYPE = "current_observation_reconstruction_detached_v1"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a checkpoint without loading it."""

    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_identity(stat_result: os.stat_result) -> tuple[int, ...]:
    """Exclude access time, which this migration's own reads may update."""

    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dictionary")
    return value


def _validate_legacy_contract(checkpoint: dict[str, Any]) -> None:
    for key in REQUIRED_MODEL_KEYS:
        _require_mapping(checkpoint.get(key), key)

    model_config = _require_mapping(checkpoint.get("model_config"), "model_config")
    expected_config = {
        "obs_dim": 93,
        "command_dim": LEGACY_COMMAND_DIM,
        "privileged_dim": 103,
        "latent_dim": 16,
        "action_dim": 29,
        "explicit_dim": 3,
    }
    for name, expected in expected_config.items():
        actual = model_config.get(name)
        if actual != expected:
            raise ValueError(
                f"Legacy model_config[{name!r}] must be {expected}, got {actual!r}"
            )

    normalization_type = checkpoint.get("input_normalization_type")
    if normalization_type != LEGACY_NORMALIZATION_TYPE:
        raise ValueError(
            "Source checkpoint must use legacy normalization "
            f"{LEGACY_NORMALIZATION_TYPE!r}, got {normalization_type!r}"
        )

    action_distribution_type = checkpoint.get("action_distribution_type")
    if action_distribution_type != EXPECTED_ACTION_DISTRIBUTION_TYPE:
        raise ValueError(
            "Source checkpoint uses an incompatible action distribution: "
            f"{action_distribution_type!r}"
        )
    auxiliary_objective_type = checkpoint.get("auxiliary_objective_type")
    if auxiliary_objective_type != EXPECTED_AUXILIARY_OBJECTIVE_TYPE:
        raise ValueError(
            "Source checkpoint uses an incompatible auxiliary objective: "
            f"{auxiliary_objective_type!r}"
        )

    log_std = checkpoint["action_distribution"].get("log_std")
    if not isinstance(log_std, torch.Tensor) or tuple(log_std.shape) != (29,):
        raise ValueError("action_distribution.log_std must have shape (29,)")

    train_args = _require_mapping(checkpoint.get("train_args"), "train_args")
    for name in ("min_action_std", "max_action_std"):
        value = train_args.get(name)
        if not isinstance(value, (float, int)) or not 0.0 < float(value):
            raise ValueError(f"train_args[{name!r}] must be positive")
    if train_args["min_action_std"] > train_args["max_action_std"]:
        raise ValueError("min_action_std must not exceed max_action_std")

    actor_weight = checkpoint["actor"].get(FIRST_LAYER_KEY)
    critic_weight = checkpoint["critic"].get(FIRST_LAYER_KEY)
    _validate_weight_shape(actor_weight, ACTOR_FIRST_LAYER_SHAPE, "actor")
    _validate_weight_shape(critic_weight, CRITIC_FIRST_LAYER_SHAPE, "critic")
    expected_group_sizes = {
        "policy": (
            len(checkpoint["encoder"])
            + len(checkpoint["actor"])
            + len(checkpoint["action_distribution"])
        ),
        "critic": len(checkpoint["critic"]),
        "decoder": len(checkpoint["decoder"]),
    }
    _validate_optimizer(
        checkpoint.get("optimizer"),
        expected_group_sizes=expected_group_sizes,
    )


def _validate_weight_shape(
    weight: Any,
    expected_shape: tuple[int, int],
    model_name: str,
) -> None:
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"{model_name}.{FIRST_LAYER_KEY} must be a tensor")
    if tuple(weight.shape) != expected_shape:
        raise ValueError(
            f"{model_name}.{FIRST_LAYER_KEY} must have shape {expected_shape}, "
            f"got {tuple(weight.shape)}"
        )
    if not weight.is_floating_point():
        raise TypeError(f"{model_name}.{FIRST_LAYER_KEY} must be floating point")


def _validate_optimizer(
    optimizer: Any,
    *,
    expected_group_sizes: dict[str, int] | None = None,
) -> None:
    optimizer = _require_mapping(optimizer, "optimizer")
    state = optimizer.get("state")
    parameter_groups = optimizer.get("param_groups")
    if not isinstance(state, dict):
        raise TypeError("optimizer['state'] must be a dictionary")
    if not isinstance(parameter_groups, list):
        raise TypeError("optimizer['param_groups'] must be a list")

    groups_by_name: dict[str, dict[str, Any]] = {}
    for group in parameter_groups:
        group = _require_mapping(group, "optimizer parameter group")
        name = group.get("name")
        if not isinstance(name, str) or name in groups_by_name:
            raise ValueError("Optimizer groups must have unique string names")
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise TypeError(f"Optimizer group {name!r} has no parameter list")
        if expected_group_sizes is not None:
            expected_size = expected_group_sizes.get(name)
            if expected_size is None or len(parameters) != expected_size:
                raise ValueError(
                    f"Optimizer group {name!r} must contain {expected_size} "
                    f"parameters, got {len(parameters)}"
                )
        groups_by_name[name] = group

    if tuple(groups_by_name) != EXPECTED_OPTIMIZER_GROUPS:
        raise ValueError(
            "Optimizer groups must be ordered as "
            f"{EXPECTED_OPTIMIZER_GROUPS}, got {tuple(groups_by_name)}"
        )


def _expand_first_layer(
    legacy_weight: torch.Tensor,
    expected_shape: tuple[int, int],
) -> torch.Tensor:
    _validate_weight_shape(legacy_weight, expected_shape, "network")
    expanded = legacy_weight.new_zeros(
        (legacy_weight.shape[0], legacy_weight.shape[1] + 2)
    )
    expanded[:, :PHASE_INSERTION_INDEX].copy_(legacy_weight[:, :PHASE_INSERTION_INDEX])
    expanded[:, PHASE_INSERTION_INDEX + 2 :].copy_(
        legacy_weight[:, PHASE_INSERTION_INDEX:]
    )
    return expanded


def migrate_checkpoint_data(
    checkpoint: dict[str, Any],
    *,
    source_sha256: str,
    source_path: str,
) -> dict[str, Any]:
    """Return a migrated checkpoint without mutating the source dictionary."""

    _validate_legacy_contract(checkpoint)
    migrated = copy.deepcopy(checkpoint)

    migrated["actor"][FIRST_LAYER_KEY] = _expand_first_layer(
        checkpoint["actor"][FIRST_LAYER_KEY],
        ACTOR_FIRST_LAYER_SHAPE,
    )
    migrated["critic"][FIRST_LAYER_KEY] = _expand_first_layer(
        checkpoint["critic"][FIRST_LAYER_KEY],
        CRITIC_FIRST_LAYER_SHAPE,
    )
    migrated["model_config"]["command_dim"] = TARGET_COMMAND_DIM
    migrated["input_normalization_type"] = NORMALIZATION_TYPE

    # Parameter count and ordering do not change, so the serialized group
    # layout remains compatible. Dropping every moment makes the next load a
    # genuinely fresh Adam optimization state.
    migrated["optimizer"]["state"] = {}
    migrated["phase_checkpoint_migration"] = {
        "type": MIGRATION_TYPE,
        "migrated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_path": source_path,
        "source_checkpoint_sha256": source_sha256,
        "source_iteration": checkpoint.get("iteration"),
        "source_command_dim": LEGACY_COMMAND_DIM,
        "target_command_dim": TARGET_COMMAND_DIM,
        "source_normalization_type": LEGACY_NORMALIZATION_TYPE,
        "target_normalization_type": NORMALIZATION_TYPE,
        "phase_weight_columns": [
            PHASE_INSERTION_INDEX,
            PHASE_INSERTION_INDEX + 1,
        ],
        "optimizer_state": "fresh_empty_adam_state",
    }
    _validate_migrated_contract(migrated)
    return migrated


def _validate_migrated_contract(checkpoint: dict[str, Any]) -> None:
    model_config = _require_mapping(checkpoint.get("model_config"), "model_config")
    if model_config.get("command_dim") != TARGET_COMMAND_DIM:
        raise ValueError("Migrated command_dim is not 5")
    if checkpoint.get("input_normalization_type") != NORMALIZATION_TYPE:
        raise ValueError("Migrated normalization metadata is not phase v2")

    actor_weight = checkpoint["actor"][FIRST_LAYER_KEY]
    critic_weight = checkpoint["critic"][FIRST_LAYER_KEY]
    _validate_weight_shape(actor_weight, (2048, 117), "actor")
    _validate_weight_shape(critic_weight, (2048, 201), "critic")
    if torch.count_nonzero(
        actor_weight[:, PHASE_INSERTION_INDEX : PHASE_INSERTION_INDEX + 2]
    ):
        raise ValueError("Migrated actor phase columns are not zero")
    if torch.count_nonzero(
        critic_weight[:, PHASE_INSERTION_INDEX : PHASE_INSERTION_INDEX + 2]
    ):
        raise ValueError("Migrated critic phase columns are not zero")

    expected_group_sizes = {
        "policy": (
            len(checkpoint["encoder"])
            + len(checkpoint["actor"])
            + len(checkpoint["action_distribution"])
        ),
        "critic": len(checkpoint["critic"]),
        "decoder": len(checkpoint["decoder"]),
    }
    _validate_optimizer(
        checkpoint.get("optimizer"),
        expected_group_sizes=expected_group_sizes,
    )
    if checkpoint["optimizer"]["state"]:
        raise ValueError("Migrated optimizer state must be empty")


def migrate_checkpoint_file(
    source: Path | str,
    destination: Path | str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Safely migrate one checkpoint and atomically publish the result."""

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if source_path == destination_path:
        raise ValueError("Source and destination must be different paths")
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source_path}")
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination_path}")

    source_identity = _stable_file_identity(source_path.stat())
    source_digest = sha256_file(source_path)
    checkpoint = load_checkpoint(source_path, map_location="cpu")
    if _stable_file_identity(source_path.stat()) != source_identity:
        raise RuntimeError("Source checkpoint changed while it was being migrated")

    migrated = migrate_checkpoint_data(
        checkpoint,
        source_sha256=source_digest,
        source_path=str(source_path),
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(migrated, temporary_path)
        verified = load_checkpoint(temporary_path, map_location="cpu")
        _validate_migrated_contract(verified)
        if overwrite:
            os.replace(temporary_path, destination_path)
        else:
            os.link(temporary_path, destination_path)
            temporary_path.unlink()
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "source": str(source_path),
        "destination": str(destination_path),
        "source_sha256": source_digest,
        "destination_sha256": sha256_file(destination_path),
        "iteration": migrated.get("iteration"),
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate a command_dim=3 / normalization-v1 checkpoint to the "
            "command_dim=5 visible-phase contract."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination; the source is never overwritten.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    try:
        result = migrate_checkpoint_file(
            args.source,
            args.destination,
            overwrite=args.overwrite,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        parser.error(str(error))

    print(f"Migrated iteration {result['iteration']} to {result['destination']}")
    print(f"source_sha256={result['source_sha256']}")
    print(f"destination_sha256={result['destination_sha256']}")
    print("actor_first_layer=(2048, 117), critic_first_layer=(2048, 201)")
    print("optimizer_state=fresh_empty_adam_state")


if __name__ == "__main__":
    main()
