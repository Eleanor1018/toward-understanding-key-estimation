import unittest
from argparse import Namespace

import torch

from config import FUTURE_REFERENCE_REWARD_PROFILE
from evaluate import resolve_v8_evaluation_curriculum
from normalization import normalize_future_reference, normalize_observation_batch
from train import (
    V8_CURRICULUM_STEPS,
    collect_rollout,
    normalized_input_clip_counts,
    prepare_v8_curriculum,
    v8_reference_state_initialization_probability,
    v8_task_mix_beta,
)


class FutureReferenceNormalizationTest(unittest.TestCase):
    def test_identity_inside_bounds_and_copy(self) -> None:
        future_reference = torch.linspace(-4.5, 4.5, 2 * 3 * 21).reshape(2, 3, 21)
        original = future_reference.clone()

        normalized = normalize_future_reference(future_reference)

        torch.testing.assert_close(normalized, original)
        torch.testing.assert_close(future_reference, original)
        self.assertNotEqual(normalized.data_ptr(), future_reference.data_ptr())

    def test_clips_builder_output_and_checks_shape(self) -> None:
        future_reference = torch.zeros(1, 3, 21)
        future_reference[0, 0, 0] = 6.0
        future_reference[0, 0, 1] = -6.0
        normalized = normalize_future_reference(future_reference)
        self.assertEqual(normalized[0, 0, 0].item(), 5.0)
        self.assertEqual(normalized[0, 0, 1].item(), -5.0)

        with self.assertRaisesRegex(ValueError, r"\(3, 21\)"):
            normalize_future_reference(torch.zeros(1, 63))
        with self.assertRaisesRegex(TypeError, "floating point"):
            normalize_future_reference(torch.zeros(1, 3, 21, dtype=torch.int64))

    def test_batch_keeps_future_reference_optional(self) -> None:
        observation = {
            "history": torch.zeros(2, 50, 93),
            "obs": torch.zeros(2, 93),
            "command": torch.zeros(2, 5),
            "privileged": torch.zeros(2, 103),
            "explicit_target": torch.zeros(2, 3),
            "future_reference": torch.ones(2, 3, 21),
        }
        normalized = normalize_observation_batch(observation)
        self.assertEqual(tuple(normalized), tuple(observation))
        self.assertEqual(normalized["future_reference"].shape, (2, 3, 21))

        without_future = dict(observation)
        without_future.pop("future_reference")
        self.assertNotIn(
            "future_reference",
            normalize_observation_batch(without_future),
        )

    def test_clip_count_includes_optional_future_reference(self) -> None:
        rollout = {
            "history": torch.zeros(1, 2, 50, 93),
            "obs": torch.zeros(1, 2, 93),
            "command": torch.zeros(1, 2, 5),
            "privileged": torch.zeros(1, 2, 103),
            "explicit_target": torch.zeros(1, 2, 3),
            "future_reference": torch.zeros(1, 2, 3, 21),
        }
        rollout["future_reference"][0, 0, 0, 0] = 5.0
        clipped, total = normalized_input_clip_counts(rollout)
        self.assertEqual(clipped.item(), 1)
        expected_per_sample = 50 * 93 + 93 + 5 + 103 + 3 + 3 * 21
        self.assertEqual(total.item(), 2 * expected_per_sample)


class V8CurriculumTest(unittest.TestCase):
    def test_exact_task_mix_boundaries_and_absolute_offset(self) -> None:
        origin = 4001
        expected = {
            1: 0.0,
            1000: 0.0,
            2000: 0.45,
            3000: 0.9,
            3250: 0.95,
            3500: 1.0,
            5000: 1.0,
        }
        for new_step, beta in expected.items():
            self.assertAlmostEqual(
                v8_task_mix_beta(origin + new_step - 1, origin),
                beta,
            )

    def test_exact_rsi_boundaries_and_absolute_offset(self) -> None:
        origin = 9001
        expected = {
            1: 1.0,
            500: 1.0,
            1000: 0.7,
            2500: 0.3,
            3500: 0.0,
            5000: 0.0,
        }
        for new_step, probability in expected.items():
            self.assertAlmostEqual(
                v8_reference_state_initialization_probability(
                    origin + new_step - 1,
                    origin,
                ),
                probability,
            )

    def test_warm_start_and_crash_resume_keep_absolute_bounds(self) -> None:
        warm_args = Namespace(
            reward_profile=FUTURE_REFERENCE_REWARD_PROFILE,
            reference_state_initialization_probability=0.0,
            v8_curriculum_origin_iteration=None,
            v8_curriculum_end_iteration=None,
        )
        prepare_v8_curriculum(
            warm_args,
            4001,
            {"reward_profile": "p1_walk_stable_v7_full_reference"},
        )
        self.assertEqual(warm_args.v8_curriculum_origin_iteration, 4001)
        self.assertEqual(
            warm_args.v8_curriculum_end_iteration,
            4001 + V8_CURRICULUM_STEPS - 1,
        )
        self.assertEqual(warm_args.initial_task_mix_beta, 0.0)
        self.assertEqual(
            warm_args.initial_reference_state_initialization_probability,
            1.0,
        )

        checkpoint_iteration = 5500
        source_args = {
            "reward_profile": FUTURE_REFERENCE_REWARD_PROFILE,
            "v8_curriculum_origin_iteration": 4001,
            "v8_curriculum_end_iteration": 9000,
            "current_task_mix_beta": v8_task_mix_beta(
                checkpoint_iteration,
                4001,
            ),
            "current_reference_state_initialization_probability": (
                v8_reference_state_initialization_probability(
                    checkpoint_iteration,
                    4001,
                )
            ),
        }
        resumed_args = Namespace(
            reward_profile=FUTURE_REFERENCE_REWARD_PROFILE,
            reference_state_initialization_probability=0.0,
            v8_curriculum_origin_iteration=None,
            v8_curriculum_end_iteration=None,
        )
        prepare_v8_curriculum(
            resumed_args,
            checkpoint_iteration + 1,
            source_args,
        )
        self.assertEqual(resumed_args.v8_curriculum_origin_iteration, 4001)
        self.assertEqual(resumed_args.v8_curriculum_end_iteration, 9000)
        self.assertAlmostEqual(
            resumed_args.initial_task_mix_beta,
            v8_task_mix_beta(checkpoint_iteration + 1, 4001),
        )

        source_args["current_task_mix_beta"] = 0.99
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            prepare_v8_curriculum(
                resumed_args,
                checkpoint_iteration + 1,
                source_args,
            )

    def test_evaluation_restores_beta_but_reports_training_rsi(self) -> None:
        self.assertEqual(
            resolve_v8_evaluation_curriculum(
                True,
                {
                    "reward_profile": FUTURE_REFERENCE_REWARD_PROFILE,
                    "current_task_mix_beta": 0.4,
                    "current_reference_state_initialization_probability": 0.7,
                },
            ),
            (0.4, 0.7),
        )
        self.assertEqual(
            resolve_v8_evaluation_curriculum(
                True,
                {
                    "reward_profile": "p1_walk_stable_v7_full_reference",
                    "reference_state_initialization_probability": 0.7,
                },
            ),
            (0.0, 0.7),
        )


def _observation(count: int) -> dict[str, torch.Tensor]:
    return {
        "history": torch.zeros(count, 50, 93),
        "obs": torch.zeros(count, 93),
        "command": torch.zeros(count, 5),
        "privileged": torch.zeros(count, 103),
        "explicit_target": torch.zeros(count, 3),
        "future_reference": torch.ones(count, 3, 21),
    }


class _Encoder:
    def __call__(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = history.shape[0]
        return history.new_zeros(count, 16), history.new_zeros(count, 3)


class _Actor:
    def __init__(self) -> None:
        self.future_shapes: list[tuple[int, ...]] = []

    def __call__(
        self,
        obs: torch.Tensor,
        command: torch.Tensor,
        latent: torch.Tensor,
        explicit: torch.Tensor,
        future_reference: torch.Tensor | None,
    ) -> torch.Tensor:
        del command, latent, explicit
        if future_reference is None:
            raise AssertionError("Actor did not receive future reference")
        self.future_shapes.append(tuple(future_reference.shape))
        return obs.new_zeros(obs.shape[0], 29)


class _Critic:
    def __init__(self) -> None:
        self.future_shapes: list[tuple[int, ...]] = []

    def __call__(
        self,
        obs: torch.Tensor,
        command: torch.Tensor,
        privileged: torch.Tensor,
        future_reference: torch.Tensor | None,
    ) -> torch.Tensor:
        del command, privileged
        if future_reference is None:
            raise AssertionError("Critic did not receive future reference")
        self.future_shapes.append(tuple(future_reference.shape))
        return obs.new_zeros(obs.shape[0])


class _ActionDistribution:
    def sample_for_ppo(
        self,
        action_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            action_mean,
            action_mean,
            action_mean.new_zeros(action_mean.shape[0]),
        )


class _OneStepTimeoutEnv:
    def step(self, actions: torch.Tensor):
        count = actions.shape[0]
        timeout_observation = _observation(1)
        timeout_observation["env_ids"] = torch.tensor([0])
        return (
            _observation(count),
            torch.zeros(count),
            torch.zeros(count, dtype=torch.bool),
            torch.tensor([True, False]),
            {"time_out_critic_observation": timeout_observation},
        )


class FutureReferenceRolloutTest(unittest.TestCase):
    def test_actor_critic_and_timeout_receive_flattened_future(self) -> None:
        actor = _Actor()
        critic = _Critic()
        rollout, _ = collect_rollout(
            env=_OneStepTimeoutEnv(),
            observation=_observation(2),
            encoder=_Encoder(),
            actor=actor,
            critic=critic,
            action_distribution=_ActionDistribution(),
            rollout_steps=1,
            gamma=0.99,
            gae_lambda=0.95,
        )

        self.assertEqual(rollout["future_reference"].shape, (1, 2, 3, 21))
        self.assertEqual(actor.future_shapes, [(2, 63)])
        self.assertEqual(critic.future_shapes, [(2, 63), (1, 63), (2, 63)])
        self.assertTrue(rollout["truncated"][0, 0].item())


if __name__ == "__main__":
    unittest.main()
