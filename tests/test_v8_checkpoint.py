import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from checkpoint_io import load_checkpoint
from config import (
    FULL_REFERENCE_REWARD_PROFILE,
    FUTURE_REFERENCE_DIM,
    FUTURE_REFERENCE_NORMALIZATION_TYPE,
    ModelConfig,
)
from normalization import NORMALIZATION_TYPE
from policy import Actor, Critic, Decoder, Encoder
from ppo import DiagonalGaussian
from train import AUXILIARY_OBJECTIVE_TYPE
from v8_checkpoint import (
    FIRST_LAYER_KEY,
    FUTURE_INSERTION_INDEX,
    MIGRATION_TYPE,
    migrate_checkpoint_file,
    sha256_file,
)


def _small_config(future_reference_dim: int) -> ModelConfig:
    return replace(
        ModelConfig(command_dim=5, future_reference_dim=future_reference_dim),
        encoder_hidden_dims=(8,),
        decoder_hidden_dims=(8,),
        actor_hidden_dims=(2048, 8),
        critic_hidden_dims=(2048, 8),
    )


def _optimizer(
    encoder: Encoder,
    decoder: Decoder,
    actor: Actor,
    critic: Critic,
    distribution: DiagonalGaussian,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        (
            {
                "params": (
                    list(encoder.parameters())
                    + list(actor.parameters())
                    + list(distribution.parameters())
                ),
                "lr": 1.0e-6,
                "name": "policy",
            },
            {
                "params": list(critic.parameters()),
                "lr": 5.0e-5,
                "name": "critic",
            },
            {
                "params": list(decoder.parameters()),
                "lr": 5.0e-5,
                "name": "decoder",
            },
        )
    )


def _v7_checkpoint() -> dict[str, object]:
    torch.manual_seed(31)
    config = _small_config(0)
    encoder = Encoder(config)
    decoder = Decoder(config)
    actor = Actor(config)
    critic = Critic(config)
    distribution = DiagonalGaussian(
        config.action_dim,
        initial_std=0.10,
        min_std=0.05,
        max_std=0.6,
    )
    optimizer = _optimizer(encoder, decoder, actor, critic, distribution)
    optimizer.zero_grad(set_to_none=True)
    loss = sum(
        parameter.square().mean()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    loss.backward()
    optimizer.step()
    return {
        "iteration": 4900,
        "action_distribution_type": distribution.distribution_type,
        "input_normalization_type": NORMALIZATION_TYPE,
        "auxiliary_objective_type": AUXILIARY_OBJECTIVE_TYPE,
        "model_config": asdict(config),
        "train_args": {
            "reward_profile": FULL_REFERENCE_REWARD_PROFILE,
            "min_action_std": 0.05,
            "max_action_std": 0.6,
        },
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "action_distribution": distribution.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


class V8CheckpointMigrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.source = Path(self.directory.name) / "v7.pt"
        self.destination = Path(self.directory.name) / "v8.pt"
        self.v7 = _v7_checkpoint()
        torch.save(self.v7, self.source)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_exact_mapping_provenance_and_fresh_adam(self) -> None:
        source_digest = sha256_file(self.source)
        result = migrate_checkpoint_file(self.source, self.destination)
        migrated = load_checkpoint(self.destination, map_location="cpu")
        self.assertEqual(result["source_sha256"], source_digest)
        self.assertEqual(sha256_file(self.source), source_digest)
        self.assertEqual(
            migrated["model_config"]["future_reference_dim"],
            FUTURE_REFERENCE_DIM,
        )
        self.assertEqual(
            migrated["future_reference_normalization_type"],
            FUTURE_REFERENCE_NORMALIZATION_TYPE,
        )
        self.assertEqual(
            migrated["v8_checkpoint_migration"]["type"],
            MIGRATION_TYPE,
        )
        self.assertEqual(migrated["optimizer"]["state"], {})
        self.assertEqual(
            migrated["optimizer"]["param_groups"],
            self.v7["optimizer"]["param_groups"],
        )

        for model_name in ("actor", "critic"):
            old_weight = self.v7[model_name][FIRST_LAYER_KEY]
            new_weight = migrated[model_name][FIRST_LAYER_KEY]
            torch.testing.assert_close(
                new_weight[:, :FUTURE_INSERTION_INDEX],
                old_weight[:, :FUTURE_INSERTION_INDEX],
            )
            torch.testing.assert_close(
                new_weight[
                    :,
                    FUTURE_INSERTION_INDEX : FUTURE_INSERTION_INDEX
                    + FUTURE_REFERENCE_DIM,
                ],
                torch.zeros(new_weight.shape[0], FUTURE_REFERENCE_DIM),
            )
            torch.testing.assert_close(
                new_weight[:, FUTURE_INSERTION_INDEX + FUTURE_REFERENCE_DIM :],
                old_weight[:, FUTURE_INSERTION_INDEX:],
            )
            for name, old_value in self.v7[model_name].items():
                if name != FIRST_LAYER_KEY:
                    torch.testing.assert_close(migrated[model_name][name], old_value)

        for model_name in ("encoder", "decoder", "action_distribution"):
            for name, old_value in self.v7[model_name].items():
                torch.testing.assert_close(migrated[model_name][name], old_value)

    def test_migrated_actor_and_critic_exactly_preserve_v7_functions(self) -> None:
        migrate_checkpoint_file(self.source, self.destination)
        migrated = load_checkpoint(self.destination, map_location="cpu")
        old_config = _small_config(0)
        new_config = _small_config(FUTURE_REFERENCE_DIM)
        old_actor = Actor(old_config)
        old_critic = Critic(old_config)
        new_actor = Actor(new_config)
        new_critic = Critic(new_config)
        old_actor.load_state_dict(self.v7["actor"], strict=True)
        old_critic.load_state_dict(self.v7["critic"], strict=True)
        new_actor.load_state_dict(migrated["actor"], strict=True)
        new_critic.load_state_dict(migrated["critic"], strict=True)

        torch.manual_seed(37)
        obs = torch.randn(5, 93)
        command = torch.randn(5, 5)
        latent = torch.randn(5, 16)
        explicit = torch.randn(5, 3)
        privileged = torch.randn(5, 103)
        future = torch.randn(5, 3, 21)
        torch.testing.assert_close(
            new_actor(obs, command, latent, explicit, future),
            old_actor(obs, command, latent, explicit),
        )
        torch.testing.assert_close(
            new_critic(obs, command, privileged, future),
            old_critic(obs, command, privileged),
        )

        encoder = Encoder(new_config)
        decoder = Decoder(new_config)
        distribution = DiagonalGaussian(
            new_config.action_dim,
            initial_std=0.10,
            min_std=0.05,
            max_std=0.6,
        )
        optimizer = _optimizer(
            encoder,
            decoder,
            new_actor,
            new_critic,
            distribution,
        )
        optimizer.load_state_dict(migrated["optimizer"])
        self.assertEqual(optimizer.state, {})
        optimizer.zero_grad(set_to_none=True)
        learned_latent, learned_explicit = encoder(torch.randn(5, 50, 93))
        loss = (
            new_actor(
                obs,
                command,
                learned_latent,
                learned_explicit,
                future,
            )
            .square()
            .mean()
            + new_critic(obs, command, privileged, future).square().mean()
            + decoder(learned_latent, learned_explicit).square().mean()
            + distribution.log_std.square().mean()
        )
        loss.backward()
        optimizer.step()
        self.assertGreater(len(optimizer.state), 0)

    def test_refuses_wrong_source_or_destructive_destination(self) -> None:
        with self.assertRaises(ValueError):
            migrate_checkpoint_file(self.source, self.source, overwrite=True)
        invalid = _v7_checkpoint()
        invalid["train_args"]["reward_profile"] = "p1_walk_stable_v6_phase_rsi"
        invalid_path = Path(self.directory.name) / "invalid.pt"
        torch.save(invalid, invalid_path)
        with self.assertRaisesRegex(ValueError, "V7"):
            migrate_checkpoint_file(
                invalid_path,
                Path(self.directory.name) / "invalid-v8.pt",
            )


if __name__ == "__main__":
    unittest.main()
