import unittest

from evaluate import resolve_v7_evaluation_curriculum


class V7EvaluationCurriculumTest(unittest.TestCase):
    def test_restores_each_checkpoint_progress_and_weight(self) -> None:
        progress, weight = resolve_v7_evaluation_curriculum(
            True,
            {
                "reward_profile": "p1_walk_stable_v7_full_reference",
                "current_reference_motion_progress": 0.5,
                "current_imitation_reward_weight": 0.6,
            },
            0.75,
        )
        self.assertEqual(progress, 0.5)
        self.assertEqual(weight, 0.6)

    def test_v6_warm_start_uses_compressed_semantics(self) -> None:
        progress, weight = resolve_v7_evaluation_curriculum(
            True,
            {"reward_profile": "p1_walk_stable_v6_phase_rsi_imitation"},
            0.75,
        )
        self.assertEqual(progress, 0.0)
        self.assertEqual(weight, 0.75)

    def test_rejects_missing_v7_curriculum(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "omitted"):
            resolve_v7_evaluation_curriculum(
                True,
                {"reward_profile": "p1_walk_stable_v7_full_reference"},
                0.75,
            )


if __name__ == "__main__":
    unittest.main()
