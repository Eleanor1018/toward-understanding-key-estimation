import math
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from checkpoint_io import load_checkpoint
from config import ModelConfig
from normalization import LEGACY_NORMALIZATION_TYPE, NORMALIZATION_TYPE
from phase_checkpoint import (
    FIRST_LAYER_KEY,
    MIGRATION_TYPE,
    PHASE_INSERTION_INDEX,
    migrate_checkpoint_file,
    sha256_file,
)
from policy import Actor, Critic, Decoder, Encoder
from ppo import DiagonalGaussian
from train import AUXILIARY_OBJECTIVE_TYPE
from train_ddp import load_training_state


def _small_config(command_dim: int) -> ModelConfig:
    return replace(
        ModelConfig(),
        command_dim=command_dim,
        encoder_hidden_dims=(8,),
        decoder_hidden_dims=(8,),
        actor_hidden_dims=(2048, 8),
        critic_hidden_dims=(2048, 8),
    )


def _create_optimizer(
    encoder: Encoder,
    decoder: Decoder,
    actor: Actor,
    critic: Critic,
    action_distribution: DiagonalGaussian,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        (
            {
                "params": (
                    list(encoder.parameters())
                    + list(actor.parameters())
                    + list(action_distribution.parameters())
                ),
                "lr": 5.0e-5,
                "name": "policy",
            },
            {
                "params": list(critic.parameters()),
                "lr": 2.0e-4,
                "name": "critic",
            },
            {
                "params": list(decoder.parameters()),
                "lr": 2.0e-4,
                "name": "decoder",
            },
        )
    )


def _legacy_checkpoint() -> dict[str, object]:
    torch.manual_seed(17)
    config = _small_config(command_dim=3)
    encoder = Encoder(config)
    decoder = Decoder(config)
    actor = Actor(config)
    critic = Critic(config)
    action_distribution = DiagonalGaussian(
        config.action_dim,
        initial_std=0.12,
        min_std=0.05,
        max_std=0.6,
    )
    optimizer = _create_optimizer(
        encoder,
        decoder,
        actor,
        critic,
        action_distribution,
    )

    optimizer.zero_grad(set_to_none=True)
    loss = sum(
        parameter.square().mean()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    loss.backward()
    optimizer.step()
    if not optimizer.state:
        raise AssertionError("Synthetic legacy optimizer must contain Adam moments")

    return {
        "iteration": 123,
        "action_distribution_type": action_distribution.distribution_type,
        "input_normalization_type": LEGACY_NORMALIZATION_TYPE,
        "auxiliary_objective_type": AUXILIARY_OBJECTIVE_TYPE,
        "model_config": asdict(config),
        "train_args": {
            "log_dir": Path("logs/legacy"),
            "min_action_std": 0.05,
            "max_action_std": 0.6,
        },
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "action_distribution": action_distribution.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


class PhaseCheckpointMigrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.source = Path(self.directory.name) / "legacy.pt"
        self.destination = Path(self.directory.name) / "phase.pt"
        self.legacy = _legacy_checkpoint()
        torch.save(self.legacy, self.source)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_exact_column_mapping_and_preservation(self) -> None:
        source_digest = sha256_file(self.source)
        result = migrate_checkpoint_file(self.source, self.destination)
        migrated = load_checkpoint(self.destination, map_location="cpu")

        self.assertEqual(sha256_file(self.source), source_digest)
        self.assertEqual(result["source_sha256"], source_digest)
        self.assertEqual(migrated["model_config"]["command_dim"], 5)
        self.assertEqual(migrated["input_normalization_type"], NORMALIZATION_TYPE)
        self.assertEqual(
            migrated["phase_checkpoint_migration"]["type"],
            MIGRATION_TYPE,
        )
        self.assertEqual(migrated["optimizer"]["state"], {})
        self.assertEqual(
            migrated["optimizer"]["param_groups"],
            self.legacy["optimizer"]["param_groups"],
        )

        for model_name in ("encoder", "decoder", "action_distribution"):
            self.assertEqual(
                tuple(migrated[model_name]),
                tuple(self.legacy[model_name]),
            )
            for name, legacy_value in self.legacy[model_name].items():
                torch.testing.assert_close(migrated[model_name][name], legacy_value)

        for model_name in ("actor", "critic"):
            old_weight = self.legacy[model_name][FIRST_LAYER_KEY]
            new_weight = migrated[model_name][FIRST_LAYER_KEY]
            self.assertEqual(new_weight.shape[1], old_weight.shape[1] + 2)
            torch.testing.assert_close(
                new_weight[:, :PHASE_INSERTION_INDEX],
                old_weight[:, :PHASE_INSERTION_INDEX],
            )
            torch.testing.assert_close(
                new_weight[:, PHASE_INSERTION_INDEX : PHASE_INSERTION_INDEX + 2],
                torch.zeros_like(
                    new_weight[:, PHASE_INSERTION_INDEX : PHASE_INSERTION_INDEX + 2]
                ),
            )
            torch.testing.assert_close(
                new_weight[:, PHASE_INSERTION_INDEX + 2 :],
                old_weight[:, PHASE_INSERTION_INDEX:],
            )
            for name, legacy_value in self.legacy[model_name].items():
                if name != FIRST_LAYER_KEY:
                    torch.testing.assert_close(
                        migrated[model_name][name],
                        legacy_value,
                    )

    def test_functions_match_and_strict_loader_can_take_an_adam_step(self) -> None:
        migrate_checkpoint_file(self.source, self.destination)
        migrated = load_checkpoint(self.destination, map_location="cpu")
        old_config = _small_config(command_dim=3)
        new_config = _small_config(command_dim=5)

        old_actor = Actor(old_config)
        old_critic = Critic(old_config)
        old_actor.load_state_dict(self.legacy["actor"], strict=True)
        old_critic.load_state_dict(self.legacy["critic"], strict=True)
        new_actor = Actor(new_config)
        new_critic = Critic(new_config)
        new_actor.load_state_dict(migrated["actor"], strict=True)
        new_critic.load_state_dict(migrated["critic"], strict=True)

        torch.manual_seed(23)
        obs = torch.randn(4, 93)
        physical_command = torch.randn(4, 3)
        phase = torch.randn(4, 2)
        command_with_phase = torch.cat((physical_command, phase), dim=-1)
        latent = torch.randn(4, 16)
        explicit = torch.randn(4, 3)
        privileged = torch.randn(4, 103)
        torch.testing.assert_close(
            new_actor(obs, command_with_phase, latent, explicit),
            old_actor(obs, physical_command, latent, explicit),
        )
        torch.testing.assert_close(
            new_critic(obs, command_with_phase, privileged),
            old_critic(obs, physical_command, privileged),
        )

        encoder = Encoder(new_config)
        decoder = Decoder(new_config)
        actor = Actor(new_config)
        critic = Critic(new_config)
        action_distribution = DiagonalGaussian(
            new_config.action_dim,
            initial_std=0.12,
            min_std=0.05,
            max_std=0.6,
        )
        optimizer = _create_optimizer(
            encoder,
            decoder,
            actor,
            critic,
            action_distribution,
        )
        next_iteration = load_training_state(
            self.destination,
            torch.device("cpu"),
            new_config,
            encoder,
            decoder,
            actor,
            critic,
            action_distribution,
            optimizer,
        )
        self.assertEqual(next_iteration, 124)
        self.assertEqual(optimizer.state, {})
        torch.testing.assert_close(
            action_distribution.log_std,
            self.legacy["action_distribution"]["log_std"],
        )

        history = torch.randn(4, new_config.history_steps, new_config.obs_dim)
        optimizer.zero_grad(set_to_none=True)
        learned_latent, learned_explicit = encoder(history)
        action_mean = actor(
            obs,
            command_with_phase,
            learned_latent,
            learned_explicit,
        )
        value = critic(obs, command_with_phase, privileged)
        reconstruction = decoder(learned_latent, learned_explicit)
        loss = (
            action_mean.square().mean()
            + value.square().mean()
            + reconstruction.square().mean()
            + action_distribution.log_std.square().mean()
        )
        self.assertTrue(math.isfinite(loss.item()))
        loss.backward()
        optimizer.step()
        self.assertGreater(len(optimizer.state), 0)

    def test_refuses_incompatible_or_destructive_migration(self) -> None:
        with self.assertRaises(ValueError):
            migrate_checkpoint_file(self.source, self.source, overwrite=True)

        self.destination.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            migrate_checkpoint_file(self.source, self.destination)
        self.assertEqual(self.destination.read_bytes(), b"keep")

        invalid = _legacy_checkpoint()
        invalid["input_normalization_type"] = NORMALIZATION_TYPE
        invalid_source = Path(self.directory.name) / "invalid.pt"
        torch.save(invalid, invalid_source)
        with self.assertRaises(ValueError):
            migrate_checkpoint_file(
                invalid_source,
                Path(self.directory.name) / "invalid-output.pt",
            )


if __name__ == "__main__":
    unittest.main()
