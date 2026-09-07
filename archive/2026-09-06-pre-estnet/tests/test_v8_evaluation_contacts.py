import unittest

import torch

from evaluate import force_contact_state_counts


class V8ForceContactEvaluationTest(unittest.TestCase):
    def test_counts_mutually_exclusive_force_contact_states(self) -> None:
        actual_contact = torch.tensor(
            [
                [False, False],
                [True, False],
                [False, True],
                [True, True],
                [True, True],
            ]
        )
        torch.testing.assert_close(
            force_contact_state_counts(actual_contact),
            torch.tensor([1, 2, 2]),
        )

    def test_flattens_leading_dimensions_but_rejects_wrong_contract(self) -> None:
        actual_contact = torch.tensor(
            [
                [[False, False], [True, False]],
                [[False, True], [True, True]],
            ]
        )
        torch.testing.assert_close(
            force_contact_state_counts(actual_contact),
            torch.tensor([1, 2, 1]),
        )
        with self.assertRaisesRegex(ValueError, r"\[\.\.\., 2\]"):
            force_contact_state_counts(torch.zeros(3, 3, dtype=torch.bool))
        with self.assertRaisesRegex(TypeError, "boolean"):
            force_contact_state_counts(torch.zeros(3, 2))


if __name__ == "__main__":
    unittest.main()
