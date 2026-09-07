import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config import (
    FUTURE_REFERENCE_DIM,
    FUTURE_REFERENCE_REWARD_PROFILE,
    ModelConfig,
    model_config_for_reward_profile,
)
from motion_reference import (
    G1_WALK_ARCHIVE_SHA256,
    G1_WALK_SOURCE_SHA256,
    V8_ANKLE_POSITION_CENTER,
    V8_ANKLE_POSITION_SCALES,
    V8_FUTURE_HORIZONS_S,
    V8_ROOT_HEIGHT_CENTER_M,
    V8_ROOT_HEIGHT_SCALE_M,
    CyclicJointReference,
    build_g1_future_reference_features,
)
from normalization import DEFAULT_JOINT_POSITIONS
from policy import Actor, Critic


class V8FutureFeatureBuilderTest(unittest.TestCase):
    def test_exact_feature_order_and_normalization(self) -> None:
        center = torch.tensor(DEFAULT_JOINT_POSITIONS)
        action_scale = torch.full((1, 3, 29), 0.5)
        expected_leg = torch.linspace(-0.9, 0.9, 36).reshape(1, 3, 12)
        joint_position = center.expand(1, 3, -1).clone()
        joint_position[..., :12] += expected_leg * action_scale[..., :12]

        expected_ankle = torch.tensor(
            [
                [
                    [[-0.5, 0.25, 0.75], [0.5, -0.25, -0.75]],
                    [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]],
                    [[0.0, 0.4, -0.4], [0.6, -0.6, 0.2]],
                ]
            ]
        )
        ankle_center = torch.tensor(V8_ANKLE_POSITION_CENTER)
        ankle_scales = torch.tensor(V8_ANKLE_POSITION_SCALES)
        ankle_position = ankle_center + expected_ankle * ankle_scales
        contact_target = torch.tensor([[[1.0, 0.0], [0.75, 0.25], [0.5, 0.5]]])
        expected_contact = 2.0 * contact_target - 1.0
        expected_height = torch.tensor([[-0.5, 0.0, 0.75]])
        root_height = V8_ROOT_HEIGHT_CENTER_M + V8_ROOT_HEIGHT_SCALE_M * expected_height

        features = build_g1_future_reference_features(
            joint_position,
            action_scale,
            center,
            ankle_position,
            contact_target,
            root_height,
        )
        expected = torch.cat(
            (
                expected_leg,
                expected_ankle.flatten(start_dim=-2),
                expected_contact,
                expected_height.unsqueeze(-1),
            ),
            dim=-1,
        )
        self.assertEqual(features.shape, (1, 3, 21))
        torch.testing.assert_close(features, expected)


class V8FutureReferenceSamplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "motions"
            / "g1_walk_mimickit.npz"
        )
        with np.load(path, allow_pickle=False) as data:
            names = tuple(str(name) for name in data["joint_names"].tolist())
        cls.reference = CyclicJointReference(
            path,
            names,
            "cpu",
            expected_archive_sha256=G1_WALK_ARCHIVE_SHA256,
            expected_source_sha256=G1_WALK_SOURCE_SHA256,
        )
        cls.reference.retarget_to_position_envelope(
            torch.tensor(DEFAULT_JOINT_POSITIONS),
            max_offset=0.22,
        )

    def test_uses_mimickit_30hz_future_horizons(self) -> None:
        current_phase = torch.tensor([0.95, 0.20])
        phase_rate = torch.tensor([1.5, 0.5])
        sample = self.reference.sample_future_reference(
            current_phase,
            phase_rate,
        )
        horizon = torch.tensor(V8_FUTURE_HORIZONS_S)
        expected_phase = torch.remainder(
            current_phase.unsqueeze(-1)
            + phase_rate.unsqueeze(-1) * horizon / self.reference.duration_s,
            1.0,
        )
        self.assertEqual(sample.features.shape, (2, 3, 21))
        torch.testing.assert_close(sample.phase, expected_phase)
        self.assertAlmostEqual(V8_FUTURE_HORIZONS_S[-1], 0.1)

    def test_default_is_full_progress_and_matches_feature_builder(self) -> None:
        sample = self.reference.sample_future_reference(
            torch.tensor([0.1, 0.6]),
            phase_rate=torch.tensor([0.8, 1.2]),
        )
        assert self.reference.trajectory_center is not None
        rebuilt = build_g1_future_reference_features(
            sample.joint_position,
            sample.action_scale,
            self.reference.trajectory_center,
            sample.ankle_position_pelvis,
            sample.contact_target,
            sample.root_height,
        )
        torch.testing.assert_close(sample.features, rebuilt)
        assert self.reference.source_joint_positions is not None
        source_leg, _, _ = self.reference.sample_phase(sample.phase.reshape(-1))
        # sample_phase is the compressed V6 alias, so full V8 legs must differ.
        self.assertGreater(
            torch.max(
                torch.abs(
                    sample.joint_position[..., :12]
                    - source_leg.reshape(2, 3, 29)[..., :12]
                )
            ).item(),
            0.01,
        )


class V8PolicyContractTest(unittest.TestCase):
    def test_profile_and_model_dimensions(self) -> None:
        config = model_config_for_reward_profile(FUTURE_REFERENCE_REWARD_PROFILE)
        self.assertEqual(config.command_dim, 5)
        self.assertEqual(config.future_reference_dim, FUTURE_REFERENCE_DIM)
        self.assertEqual(config.actor_input_dim, 180)
        self.assertEqual(config.critic_input_dim, 264)

    def test_actor_and_critic_accept_structured_or_flat_targets(self) -> None:
        config = replace(
            model_config_for_reward_profile(FUTURE_REFERENCE_REWARD_PROFILE),
            actor_hidden_dims=(16,),
            critic_hidden_dims=(16,),
        )
        actor = Actor(config)
        critic = Critic(config)
        obs = torch.randn(4, 93)
        command = torch.randn(4, 5)
        latent = torch.randn(4, 16)
        explicit = torch.randn(4, 3)
        privileged = torch.randn(4, 103)
        future = torch.randn(4, 3, 21)
        torch.testing.assert_close(
            actor(obs, command, latent, explicit, future),
            actor(obs, command, latent, explicit, future.reshape(4, 63)),
        )
        torch.testing.assert_close(
            critic(obs, command, privileged, future),
            critic(obs, command, privileged, future.reshape(4, 63)),
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            actor(obs, command, latent, explicit)

    def test_legacy_models_keep_optional_target_absent(self) -> None:
        config = replace(
            ModelConfig(command_dim=5, future_reference_dim=0),
            actor_hidden_dims=(16,),
            critic_hidden_dims=(16,),
        )
        actor = Actor(config)
        critic = Critic(config)
        obs = torch.randn(2, 93)
        command = torch.randn(2, 5)
        latent = torch.randn(2, 16)
        explicit = torch.randn(2, 3)
        privileged = torch.randn(2, 103)
        self.assertEqual(actor(obs, command, latent, explicit).shape, (2, 29))
        self.assertEqual(critic(obs, command, privileged).shape, (2,))
        with self.assertRaisesRegex(ValueError, "does not accept"):
            actor(obs, command, latent, explicit, torch.zeros(2, 3, 21))


if __name__ == "__main__":
    unittest.main()
