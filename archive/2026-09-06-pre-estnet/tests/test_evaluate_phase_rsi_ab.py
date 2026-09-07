import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.evaluate_phase_rsi_ab import (
    evaluate_ab,
    evaluate_arm,
    main,
    select_final_checkpoint_result,
)


def checkpoint_result(iteration: int) -> dict:
    return {
        "checkpoint": f"/tmp/checkpoint_{iteration:05d}.pt",
        "checkpoint_iteration": iteration,
        "survival_fraction": 0.80,
        "survival_adjusted_command_bin_tracking": {
            "forward": {"mean_directional_velocity": 0.15}
        },
        "gait": {
            "landing_rate_hz": 1.0,
            "alternating_landing_fraction": 0.80,
        },
        "raw_action_saturation_fraction": 0.019,
    }


def report_with(result: dict) -> dict:
    return {
        "results": [
            {
                "checkpoint": None,
                "checkpoint_iteration": None,
                "label": "zero_action",
            },
            result,
        ]
    }


class FinalCheckpointSelectionTest(unittest.TestCase):
    def test_selects_greatest_iteration_not_last_result(self) -> None:
        report = {
            "results": [
                checkpoint_result(250),
                checkpoint_result(100),
                {"checkpoint": None, "checkpoint_iteration": None},
            ]
        }

        selected = select_final_checkpoint_result(report)

        self.assertEqual(selected["checkpoint_iteration"], 250)

    def test_missing_checkpoint_results_fail_every_gate(self) -> None:
        arm = evaluate_arm(
            {"results": [{"checkpoint": None, "checkpoint_iteration": None}]},
            "empty.json",
        )

        self.assertFalse(arm["passed"])
        self.assertIn("no checkpoint results", arm["error"])
        self.assertTrue(all(not gate["passed"] for gate in arm["gates"].values()))


class PhaseRsiGateTest(unittest.TestCase):
    def test_threshold_boundaries_pass_except_saturation_is_strict(self) -> None:
        passing = checkpoint_result(250)
        report = evaluate_ab(report_with(passing), report_with(copy.deepcopy(passing)))
        self.assertTrue(report["passed"])

        at_saturation_limit = checkpoint_result(250)
        at_saturation_limit["raw_action_saturation_fraction"] = 0.02
        failed = evaluate_arm(report_with(at_saturation_limit), "treatment.json")

        self.assertFalse(failed["passed"])
        self.assertFalse(failed["gates"]["action_saturation"]["passed"])

    def test_missing_and_null_metrics_are_explicit_failures(self) -> None:
        missing = checkpoint_result(250)
        del missing["gait"]["landing_rate_hz"]
        missing_arm = evaluate_arm(report_with(missing), "missing.json")
        self.assertEqual(
            missing_arm["gates"]["valid_landing_rate"]["reason"],
            "missing metric: gait.landing_rate_hz",
        )

        null = checkpoint_result(250)
        null["survival_adjusted_command_bin_tracking"]["forward"][
            "mean_directional_velocity"
        ] = None
        null_arm = evaluate_arm(report_with(null), "null.json")
        self.assertEqual(
            null_arm["gates"]["fixed_horizon_forward_directional_velocity"]["reason"],
            "null metric: "
            "survival_adjusted_command_bin_tracking.forward.mean_directional_velocity",
        )

    def test_non_numeric_and_non_finite_metrics_fail(self) -> None:
        invalid = checkpoint_result(250)
        invalid["survival_fraction"] = True
        invalid["gait"]["alternating_landing_fraction"] = float("nan")

        arm = evaluate_arm(report_with(invalid), "invalid.json")

        self.assertFalse(arm["gates"]["survival"]["passed"])
        self.assertIn("non-numeric", arm["gates"]["survival"]["reason"])
        self.assertFalse(arm["gates"]["alternating_landing_fraction"]["passed"])
        self.assertIn(
            "non-finite",
            arm["gates"]["alternating_landing_fraction"]["reason"],
        )

    def test_combined_report_requires_both_arms_to_pass(self) -> None:
        control = checkpoint_result(250)
        treatment = checkpoint_result(250)
        control["survival_fraction"] = 0.79

        report = evaluate_ab(report_with(control), report_with(treatment))

        self.assertFalse(report["passed"])
        self.assertFalse(report["control"]["passed"])
        self.assertTrue(report["treatment"]["passed"])


class PhaseRsiCliTest(unittest.TestCase):
    def test_cli_prints_and_writes_machine_readable_report(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            control_path = root / "control.json"
            treatment_path = root / "treatment.json"
            output_path = root / "gate_report.json"
            input_report = report_with(checkpoint_result(250))
            control_path.write_text(json.dumps(input_report), encoding="utf-8")
            treatment_path.write_text(json.dumps(input_report), encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(control_path),
                        str(treatment_path),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            emitted = json.loads(stdout.getvalue())
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(emitted, written)
            self.assertTrue(emitted["passed"])


if __name__ == "__main__":
    unittest.main()
