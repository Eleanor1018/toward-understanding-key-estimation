import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


G1_ENV_PATH = Path(__file__).resolve().parents[1] / "g1_env.py"
PURE_HELPERS = {
    "_signed_similarity",
    "_v8_force_contact_score",
    "_v8_clearance_score",
    "_v8_support_score",
    "_v8_landing_score",
    "_v8_imitation_score",
    "_v8_task_score",
    "_v8_core_reward",
}


def load_g1_env_functions(names: set[str]) -> dict[str, object]:
    tree = ast.parse(G1_ENV_PATH.read_text(encoding="utf-8"))
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
        ):
            functions.append(node)
        if isinstance(node, ast.ClassDef):
            functions.extend(
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name in names
            )
    found = {function.name for function in functions}
    if found != names:
        raise AssertionError(f"missing g1_env functions: {sorted(names - found)}")
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace: dict[str, object] = {
        "G1_FOOT_SOLE_OFFSET_M": 0.035,
        "FUTURE_REFERENCE_REWARD_PROFILE": ("p1_walk_stable_v8_future_reference"),
        "math": math,
        "torch": torch,
    }
    exec(compile(module, str(G1_ENV_PATH), "exec"), namespace)
    return namespace


class V8RewardMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.functions = load_g1_env_functions(PURE_HELPERS)

    def test_force_contact_is_signed_and_double_support_is_zero(self) -> None:
        score, actual = self.functions["_v8_force_contact_score"](
            torch.tensor(
                [
                    [100.0, 100.0],
                    [100.0, 0.0],
                    [0.0, 100.0],
                ]
            ),
            torch.full((3,), 1_000.0),
            torch.tensor(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                ]
            ),
        )
        torch.testing.assert_close(score, torch.tensor([0.0, 1.0, -1.0]))
        torch.testing.assert_close(
            actual,
            torch.tensor(
                [
                    [True, True],
                    [True, False],
                    [False, True],
                ]
            ),
        )

    def test_clearance_is_signed_and_double_support_is_not_half_rewarded(self) -> None:
        score, actual, target = self.functions["_v8_clearance_score"](
            torch.tensor(
                [
                    [0.035, 0.035],
                    [0.035, 0.235],
                ]
            ),
            torch.zeros(2),
            torch.tensor(
                [
                    [-0.75, -0.55],
                    [-0.75, -0.55],
                ]
            ),
        )
        torch.testing.assert_close(actual[0], torch.zeros(2), atol=1.0e-7, rtol=0.0)
        torch.testing.assert_close(target[0], torch.tensor([0.0, 0.2]))
        self.assertLess(abs(score[0].item()), 1.0e-5)
        self.assertAlmostEqual(score[1].item(), 1.0, places=6)

    def test_signed_scores_span_full_range_and_mix_endpoints(self) -> None:
        imitation = self.functions["_v8_imitation_score"]
        task = self.functions["_v8_task_score"]
        core = self.functions["_v8_core_reward"]
        ones = torch.ones(2)
        zeros = torch.zeros(2)

        torch.testing.assert_close(
            imitation(ones, ones, ones, ones, ones, ones, ones),
            ones,
        )
        torch.testing.assert_close(
            imitation(zeros, zeros, zeros, zeros, zeros, -ones, -ones),
            -ones,
        )
        torch.testing.assert_close(
            task(ones, ones, ones, ones, ones, ones, ones),
            ones,
        )
        torch.testing.assert_close(
            task(zeros, -ones, zeros, zeros, zeros, -ones, -ones),
            -ones,
        )

        imitation_value = torch.tensor([0.4])
        task_value = torch.tensor([-0.2])
        reward, mixed = core(imitation_value, task_value, 0.0)
        torch.testing.assert_close(mixed, imitation_value)
        torch.testing.assert_close(reward, 3.0 * imitation_value)
        reward, mixed = core(imitation_value, task_value, 1.0)
        torch.testing.assert_close(mixed, task_value)
        torch.testing.assert_close(reward, 3.0 * task_value)

    def test_support_and_landing_scores_are_symmetric(self) -> None:
        support = self.functions["_v8_support_score"](
            torch.tensor(
                [
                    [True, False],
                    [True, True],
                    [False, False],
                ]
            )
        )
        torch.testing.assert_close(support, torch.tensor([1.0, -0.25, -1.0]))
        landing = self.functions["_v8_landing_score"](
            torch.tensor([True, False, False]),
            torch.tensor([False, True, False]),
        )
        torch.testing.assert_close(landing, torch.tensor([1.0, -1.0, 0.0]))


class V8EnvironmentContractTest(unittest.TestCase):
    def test_runtime_setters_validate_and_update_cfg(self) -> None:
        namespace = load_g1_env_functions(
            {
                "set_task_mix_beta",
                "set_reference_state_initialization_probability",
            }
        )
        cfg = SimpleNamespace(
            reward_profile="p1_walk_stable_v8_future_reference",
            task_mix_beta=0.0,
            reference_state_initialization_probability=0.7,
        )
        environment = SimpleNamespace(
            cfg=cfg,
            _phase_rsi_enabled=True,
            _task_mix_beta=0.0,
        )

        namespace["set_task_mix_beta"](environment, 0.65)
        namespace["set_reference_state_initialization_probability"](
            environment,
            0.25,
        )
        self.assertEqual(environment._task_mix_beta, 0.65)
        self.assertEqual(cfg.task_mix_beta, 0.65)
        self.assertEqual(cfg.reference_state_initialization_probability, 0.25)
        with self.assertRaises(ValueError):
            namespace["set_task_mix_beta"](environment, 1.01)
        with self.assertRaises(ValueError):
            namespace["set_reference_state_initialization_probability"](
                environment,
                -0.01,
            )

    def test_future_sampler_uses_live_phase_rate_and_full_progress(self) -> None:
        namespace = load_g1_env_functions({"_future_reference_observation"})
        expected = torch.arange(2 * 3 * 21, dtype=torch.float32).reshape(2, 3, 21)
        calls: list[tuple[torch.Tensor, torch.Tensor, float]] = []

        class Reference:
            def sample_future_reference(
                self,
                phase: torch.Tensor,
                phase_rate: torch.Tensor,
                progress: float,
            ) -> SimpleNamespace:
                calls.append((phase.clone(), phase_rate.clone(), progress))
                return SimpleNamespace(features=expected)

        environment = SimpleNamespace(
            _future_reference_enabled=True,
            _walk_reference=Reference(),
            _current_reference_phase=lambda: torch.tensor([0.2, 0.8]),
            _reference_phase_rate=lambda: torch.tensor([0.5, 1.5]),
        )
        actual = namespace["_future_reference_observation"](environment)

        self.assertIs(actual, expected)
        torch.testing.assert_close(calls[0][0], torch.tensor([0.2, 0.8]))
        torch.testing.assert_close(calls[0][1], torch.tensor([0.5, 1.5]))
        self.assertEqual(calls[0][2], 1.0)

    def test_source_contract_covers_normal_and_terminal_future_targets(self) -> None:
        source = G1_ENV_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        classes = {
            node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
        }
        cfg_assignments = {
            target.id
            for node in classes["G1KeyEstimationEnvCfg"].body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        env_methods = {
            node.name
            for node in classes["G1KeyEstimationEnv"].body
            if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("task_mix_beta", cfg_assignments)
        self.assertIn("reference_state_initialization_probability", cfg_assignments)
        self.assertIn("set_task_mix_beta", env_methods)
        self.assertIn("set_reference_state_initialization_probability", env_methods)
        self.assertGreaterEqual(source.count('["future_reference"]'), 4)
        get_dones = ast.get_source_segment(
            source,
            next(
                node
                for node in classes["G1KeyEstimationEnv"].body
                if isinstance(node, ast.FunctionDef) and node.name == "_get_dones"
            ),
        )
        assert get_dones is not None
        self.assertLess(
            get_dones.index("self._advance_reference_phase()"),
            get_dones.index("self._future_reference_observation()"),
        )


if __name__ == "__main__":
    unittest.main()
