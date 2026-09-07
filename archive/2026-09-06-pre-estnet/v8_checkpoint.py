"""Migrate V7 checkpoints to the compact V8 future-reference contract."""

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
from config import (
    FULL_REFERENCE_REWARD_PROFILE,
    FUTURE_REFERENCE_DIM,
    FUTURE_REFERENCE_NORMALIZATION_TYPE,
)
from normalization import NORMALIZATION_TYPE


MIGRATION_TYPE = "future_reference_0_to_63_v1"
SOURCE_COMMAND_DIM = 5
SOURCE_FUTURE_REFERENCE_DIM = 0
TARGET_FUTURE_REFERENCE_DIM = FUTURE_REFERENCE_DIM
FUTURE_INSERTION_INDEX = 98
FIRST_LAYER_KEY = "trunk.0.weight"
SOURCE_ACTOR_FIRST_LAYER_SHAPE = (2048, 117)
SOURCE_CRITIC_FIRST_LAYER_SHAPE = (2048, 201)
TARGET_ACTOR_FIRST_LAYER_SHAPE = (2048, 180)
TARGET_CRITIC_FIRST_LAYER_SHAPE = (2048, 264)
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
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_identity(stat_result: os.stat_result) -> tuple[int, ...]:
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
    expected_group_sizes: dict[str, int],
) -> None:
    optimizer = _require_mapping(optimizer, "optimizer")
    if not isinstance(optimizer.get("state"), dict):
        raise TypeError("optimizer['state'] must be a dictionary")
    parameter_groups = optimizer.get("param_groups")
    if not isinstance(parameter_groups, list):
        raise TypeError("optimizer['param_groups'] must be a list")
    group_names = []
    for group_value in parameter_groups:
        group = _require_mapping(group_value, "optimizer parameter group")
        name = group.get("name")
        if not isinstance(name, str) or name in group_names:
            raise ValueError("Optimizer groups must have unique string names")
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise TypeError(f"Optimizer group {name!r} has no parameter list")
        expected_size = expected_group_sizes.get(name)
        if expected_size is None or len(parameters) != expected_size:
            raise ValueError(
                f"Optimizer group {name!r} must contain {expected_size} parameters, "
                f"got {len(parameters)}"
            )
        group_names.append(name)
    if tuple(group_names) != EXPECTED_OPTIMIZER_GROUPS:
        raise ValueError(
            "Optimizer groups must be ordered as "
            f"{EXPECTED_OPTIMIZER_GROUPS}, got {tuple(group_names)}"
        )


def _expected_optimizer_group_sizes(checkpoint: dict[str, Any]) -> dict[str, int]:
    return {
        "policy": (
            len(checkpoint["encoder"])
            + len(checkpoint["actor"])
            + len(checkpoint["action_distribution"])
        ),
        "critic": len(checkpoint["critic"]),
        "decoder": len(checkpoint["decoder"]),
    }


def _validate_v7_contract(checkpoint: dict[str, Any]) -> None:
    for key in REQUIRED_MODEL_KEYS:
        _require_mapping(checkpoint.get(key), key)
    model_config = _require_mapping(checkpoint.get("model_config"), "model_config")
    expected_config = {
        "obs_dim": 93,
        "command_dim": SOURCE_COMMAND_DIM,
        "privileged_dim": 103,
        "latent_dim": 16,
        "action_dim": 29,
        "explicit_dim": 3,
    }
    for name, expected in expected_config.items():
        if model_config.get(name) != expected:
            raise ValueError(
                f"V7 model_config[{name!r}] must be {expected}, "
                f"got {model_config.get(name)!r}"
            )
    if model_config.get("future_reference_dim", 0) != SOURCE_FUTURE_REFERENCE_DIM:
        raise ValueError("Source checkpoint already has a future-reference input")
    if checkpoint.get("input_normalization_type") != NORMALIZATION_TYPE:
        raise ValueError("Source checkpoint uses incompatible input normalization")
    if checkpoint.get("future_reference_normalization_type") is not None:
        raise ValueError("Source checkpoint already declares future normalization")
    if checkpoint.get("action_distribution_type") != EXPECTED_ACTION_DISTRIBUTION_TYPE:
        raise ValueError("Source checkpoint uses an incompatible action distribution")
    if checkpoint.get("auxiliary_objective_type") != EXPECTED_AUXILIARY_OBJECTIVE_TYPE:
        raise ValueError("Source checkpoint uses an incompatible auxiliary objective")
    train_args = _require_mapping(checkpoint.get("train_args"), "train_args")
    if train_args.get("reward_profile") != FULL_REFERENCE_REWARD_PROFILE:
        raise ValueError("Source checkpoint must be a V7 full-reference checkpoint")
    log_std = checkpoint["action_distribution"].get("log_std")
    if not isinstance(log_std, torch.Tensor) or tuple(log_std.shape) != (29,):
        raise ValueError("action_distribution.log_std must have shape (29,)")
    _validate_weight_shape(
        checkpoint["actor"].get(FIRST_LAYER_KEY),
        SOURCE_ACTOR_FIRST_LAYER_SHAPE,
        "actor",
    )
    _validate_weight_shape(
        checkpoint["critic"].get(FIRST_LAYER_KEY),
        SOURCE_CRITIC_FIRST_LAYER_SHAPE,
        "critic",
    )
    _validate_optimizer(
        checkpoint.get("optimizer"),
        _expected_optimizer_group_sizes(checkpoint),
    )


def _expand_first_layer(
    source_weight: torch.Tensor,
    expected_shape: tuple[int, int],
) -> torch.Tensor:
    _validate_weight_shape(source_weight, expected_shape, "network")
    expanded = source_weight.new_zeros(
        source_weight.shape[0],
        source_weight.shape[1] + TARGET_FUTURE_REFERENCE_DIM,
    )
    expanded[:, :FUTURE_INSERTION_INDEX].copy_(
        source_weight[:, :FUTURE_INSERTION_INDEX]
    )
    expanded[:, FUTURE_INSERTION_INDEX + TARGET_FUTURE_REFERENCE_DIM :].copy_(
        source_weight[:, FUTURE_INSERTION_INDEX:]
    )
    return expanded


def _validate_migrated_contract(checkpoint: dict[str, Any]) -> None:
    model_config = _require_mapping(checkpoint.get("model_config"), "model_config")
    if model_config.get("command_dim") != SOURCE_COMMAND_DIM:
        raise ValueError("Migrated command_dim is not 5")
    if model_config.get("future_reference_dim") != TARGET_FUTURE_REFERENCE_DIM:
        raise ValueError("Migrated future_reference_dim is not 63")
    if (
        checkpoint.get("future_reference_normalization_type")
        != FUTURE_REFERENCE_NORMALIZATION_TYPE
    ):
        raise ValueError("Migrated future-reference normalization metadata is invalid")
    actor_weight = checkpoint["actor"][FIRST_LAYER_KEY]
    critic_weight = checkpoint["critic"][FIRST_LAYER_KEY]
    _validate_weight_shape(actor_weight, TARGET_ACTOR_FIRST_LAYER_SHAPE, "actor")
    _validate_weight_shape(critic_weight, TARGET_CRITIC_FIRST_LAYER_SHAPE, "critic")
    phase_slice = slice(
        FUTURE_INSERTION_INDEX,
        FUTURE_INSERTION_INDEX + TARGET_FUTURE_REFERENCE_DIM,
    )
    if torch.count_nonzero(actor_weight[:, phase_slice]).item() != 0:
        raise ValueError("Migrated actor future-reference columns are not zero")
    if torch.count_nonzero(critic_weight[:, phase_slice]).item() != 0:
        raise ValueError("Migrated critic future-reference columns are not zero")
    _validate_optimizer(
        checkpoint.get("optimizer"),
        _expected_optimizer_group_sizes(checkpoint),
    )
    if checkpoint["optimizer"]["state"]:
        raise ValueError("Migrated optimizer state must be empty")


def migrate_checkpoint_data(
    checkpoint: dict[str, Any],
    *,
    source_sha256: str,
    source_path: str,
) -> dict[str, Any]:
    """Return a V8 checkpoint without mutating the V7 source dictionary."""

    _validate_v7_contract(checkpoint)
    migrated = copy.deepcopy(checkpoint)
    migrated["actor"][FIRST_LAYER_KEY] = _expand_first_layer(
        checkpoint["actor"][FIRST_LAYER_KEY],
        SOURCE_ACTOR_FIRST_LAYER_SHAPE,
    )
    migrated["critic"][FIRST_LAYER_KEY] = _expand_first_layer(
        checkpoint["critic"][FIRST_LAYER_KEY],
        SOURCE_CRITIC_FIRST_LAYER_SHAPE,
    )
    migrated["model_config"]["future_reference_dim"] = TARGET_FUTURE_REFERENCE_DIM
    migrated["future_reference_normalization_type"] = (
        FUTURE_REFERENCE_NORMALIZATION_TYPE
    )
    migrated["optimizer"]["state"] = {}
    migrated["v8_checkpoint_migration"] = {
        "type": MIGRATION_TYPE,
        "migrated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_path": source_path,
        "source_checkpoint_sha256": source_sha256,
        "source_iteration": checkpoint.get("iteration"),
        "source_future_reference_dim": SOURCE_FUTURE_REFERENCE_DIM,
        "target_future_reference_dim": TARGET_FUTURE_REFERENCE_DIM,
        "future_weight_columns": [
            FUTURE_INSERTION_INDEX,
            FUTURE_INSERTION_INDEX + TARGET_FUTURE_REFERENCE_DIM - 1,
        ],
        "future_reference_normalization_type": FUTURE_REFERENCE_NORMALIZATION_TYPE,
        "optimizer_state": "fresh_empty_adam_state",
    }
    _validate_migrated_contract(migrated)
    return migrated


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
        description="Migrate a V7 checkpoint to the 63-D V8 future reference."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--overwrite", action="store_true")
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
    print("actor_first_layer=(2048, 180), critic_first_layer=(2048, 264)")
    print("optimizer_state=fresh_empty_adam_state")


if __name__ == "__main__":
    main()
